------------------------------ MODULE SlabWord ------------------------------
\* Layout and shader-decode model for the 64-bit packed GPU block-table word
\* (`src/blocks.rs::BlockPool::gpu_table_entry`, `src/engine.rs::bind_pool_slabs`,
\* and `src/kernels.metal::kv_blk_ptr`):
\*   word == (slab_index << 62) | within_slab_byte_offset
\*   si   == word >> 62                    (in 0..3)
\*   off  == word & ((1 << 62) - 1)        (62-bit unsigned byte offset)
\*   base == KV0 (buf 3) | KV1 (buf 10) | KV2 (buf 11) | KV3 (buf 12)
\*
\* TLC integers are signed 32-bit, so each 64-bit word is represented exact to
\* the bit as four non-negative limbs:
\*   top2 \in 0..3           (bits 63..62)
\*   hi30 \in 0..1073741823  (bits 61..32, i.e. 4 GiB units mod 2^30)
\*   mib  \in 0..4095        (bits 31..20, i.e. 1 MiB units mod 4096)
\*   sub  \in 0..1048575     (bits 19..0,  i.e. bytes mod 1 MiB)

EXTENDS Integers, FiniteSets, Sequences

CONSTANTS
    Kvb,            \* Element byte width: 2 (fp16) or 1 (--kv8)
    SlotsPerSlab,   \* Maximum slots per GPU slab buffer (>= 1)
    MaxSlots,       \* Maximum pool slots to grow during exploration
    MaxBlocks,      \* Logical block-table length
    BaseGiB4,       \* Added 4-GiB units on within-slab offset (0, 1, or 2^30-1)
    Bug             \* "none" | "unmasked" | "wrong_bind" | "no_overflow_check" | "fifth_slab"

ASSUME
    /\ Kvb \in {1, 2}
    /\ SlotsPerSlab >= 1
    /\ MaxSlots >= 1
    /\ MaxBlocks >= 1
    /\ BaseGiB4 >= 0
    /\ Bug \in {"none", "unmasked", "wrong_bind", "no_overflow_check", "fifth_slab"}

MaxSlabs == 4
Top30Lim == 1073741824  \* 2^30 (4 GiB * 2^30 == 2^62 bytes)
GiB4MiB  == 4096        \* 2^12 MiB == 4 GiB
MiBBytes == 1048576     \* 2^20 bytes == 1 MiB

Kinds == {"full", "window"}
StrideMiB(kind) == (IF kind = "full" THEN 1 ELSE 5) * Kvb

\* Buffer indices bound by `Engine::bind_pool_slabs`: KV0=3, KV1=10, KV2=11, KV3=12.
BufferIndices == <<3, 10, 11, 12>>

ZeroWord == [top2 |-> 0, hi30 |-> 0, mib |-> 0, sub |-> 0]

BitOr1(a, b) == IF a = 1 \/ b = 1 THEN 1 ELSE 0
BitOr2(a, b) == BitOr1(a \div 2, b \div 2) * 2 + BitOr1(a % 2, b % 2)

\* 64-bit representation of within-slab byte offset `(BaseGiB4 * 4GiB) + withinIdx * StrideMiB`.
WithinSlabWord(kind, withinIdx) ==
    LET totalMiB == withinIdx * StrideMiB(kind)
        g4       == BaseGiB4 + (totalMiB \div GiB4MiB)
        remMiB   == totalMiB % GiB4MiB
    IN [top2 |-> g4 \div Top30Lim,
        hi30 |-> g4 % Top30Lim,
        mib  |-> remMiB,
        sub  |-> 0]

FitsIn62Bits(w) == w.top2 = 0

\* `(slab_bits << 62) | off_word`
PackWord(slab, offW) ==
    [top2 |-> BitOr2(slab % 4, offW.top2),
     hi30 |-> offW.hi30,
     mib  |-> offW.mib,
     sub  |-> offW.sub]

\* Add an in-slot byte offset (< 1 MiB or multi-MiB) to a 64-bit word.
AddInSlotBytes(w, addMiB, addSub) ==
    LET sTotal == w.sub + addSub
        mTotal == w.mib + addMiB + (sTotal \div MiBBytes)
        hTotal == w.hi30 + (mTotal \div GiB4MiB)
    IN [top2 |-> (w.top2 + (hTotal \div Top30Lim)) % 4,
        hi30 |-> hTotal % Top30Lim,
        mib  |-> mTotal % GiB4MiB,
        sub  |-> sTotal % MiBBytes]

WordLt(a, b) ==
    \/ a.top2 < b.top2
    \/ (a.top2 = b.top2 /\ a.hi30 < b.hi30)
    \/ (a.top2 = b.top2 /\ a.hi30 = b.hi30 /\ a.mib < b.mib)
    \/ (a.top2 = b.top2 /\ a.hi30 = b.hi30 /\ a.mib = b.mib /\ a.sub < b.sub)

WordLe(a, b) == a = b \/ WordLt(a, b)

\* `Engine::bind_pool_slabs` binds slabs[0..3] to buffer indices 3, 10, 11, 12,
\* falling back to slab 0 for unallocated higher slab slots.
EncoderBind(slabCnt) ==
    IF Bug = "wrong_bind"
    THEN [i \in 0..3 |->
            IF i = 0 THEN 0
            ELSE IF i = 1 THEN (IF 2 < slabCnt THEN 2 ELSE 0)
            ELSE IF i = 2 THEN (IF 1 < slabCnt THEN 1 ELSE 0)
            ELSE (IF 3 < slabCnt THEN 3 ELSE 0)]
    ELSE [i \in 0..3 |-> IF i < slabCnt THEN i ELSE 0]

\* `kv_blk_ptr` in `src/kernels.metal:1269-1278`:
\*   si  = uint(entry >> 62);
\*   off = entry & KV_SLAB_MASK;
ShaderDecode(entry, slabCnt, inMiB, inSub) ==
    LET si       == entry.top2
        masked   == IF Bug = "unmasked"
                    THEN entry
                    ELSE [entry EXCEPT !.top2 = 0]
        byteAddr == AddInSlotBytes(masked, inMiB, inSub)
        bindMap  == EncoderBind(slabCnt)
    IN [bufIdx  |-> BufferIndices[si + 1],
        slab    |-> bindMap[si],
        addr    |-> byteAddr]

VARIABLES
    kind,       \* Current pool kind under check ("full" or "window")
    slotCount,  \* Number of allocated slots in pool (0..MaxSlots)
    slabCount,  \* Number of open slabs in pool
    slotTaken,  \* [1..MaxSlots -> BOOLEAN]
    tableSlot,  \* [0..(MaxBlocks - 1) -> 0..MaxSlots] (0 = empty)
    gpuTable    \* [0..(MaxBlocks - 1) -> Word]

vars == <<kind, slotCount, slabCount, slotTaken, tableSlot, gpuTable>>

SlotSlab(s)   == (s - 1) \div SlotsPerSlab
SlotWithin(s) == (s - 1) % SlotsPerSlab

Init ==
    /\ kind \in Kinds
    /\ slotCount = 0
    /\ slabCount = 0
    /\ slotTaken = [s \in 1..MaxSlots |-> FALSE]
    /\ tableSlot = [b \in 0..(MaxBlocks - 1) |-> 0]
    /\ gpuTable  = [b \in 0..(MaxBlocks - 1) |-> ZeroWord]

GrowAndTake ==
    /\ slotCount < MaxSlots
    /\ LET nextS  == slotCount + 1
           slabI  == (nextS - 1) \div SlotsPerSlab
           newCnt == IF slabI + 1 > slabCount THEN slabI + 1 ELSE slabCount
       IN /\ (Bug = "fifth_slab" \/ slabI < MaxSlabs)
          /\ slotCount' = nextS
          /\ slabCount' = newCnt
          /\ slotTaken' = [slotTaken EXCEPT ![nextS] = TRUE]
    /\ UNCHANGED <<kind, tableSlot, gpuTable>>

BindBlock(b, s) ==
    /\ b \in 0..(MaxBlocks - 1)
    /\ s \in 1..slotCount
    /\ slotTaken[s]
    /\ LET slabI == SlotSlab(s)
           wIdx  == SlotWithin(s)
           offW  == WithinSlabWord(kind, wIdx)
       IN /\ (Bug = "fifth_slab" \/ slabI < MaxSlabs)
          /\ (Bug = "no_overflow_check" \/ FitsIn62Bits(offW))
          /\ tableSlot' = [tableSlot EXCEPT ![b] = s]
          /\ gpuTable'  = [gpuTable EXCEPT ![b] = PackWord(slabI, offW)]
    /\ UNCHANGED <<kind, slotCount, slabCount, slotTaken>>

ClearBlock(b) ==
    /\ b \in 0..(MaxBlocks - 1)
    /\ tableSlot[b] # 0
    /\ tableSlot' = [tableSlot EXCEPT ![b] = 0]
    /\ gpuTable'  = [gpuTable EXCEPT ![b] = ZeroWord]
    /\ UNCHANGED <<kind, slotCount, slabCount, slotTaken>>

Next ==
    \/ GrowAndTake
    \/ (\E b \in 0..(MaxBlocks - 1), s \in 1..MaxSlots : BindBlock(b, s))
    \/ (\E b \in 0..(MaxBlocks - 1) : ClearBlock(b))

Spec == Init /\ [][Next]_vars

\* ---- Invariants -----------------------------------------------------------

TypeOK ==
    /\ kind \in Kinds
    /\ slotCount \in 0..MaxSlots
    /\ slabCount \in 0..MaxSlots
    /\ \A b \in 0..(MaxBlocks - 1) :
        /\ tableSlot[b] \in 0..slotCount
        /\ gpuTable[b].top2 \in 0..3
        /\ gpuTable[b].hi30 \in 0..(Top30Lim - 1)
        /\ gpuTable[b].mib  \in 0..(GiB4MiB - 1)
        /\ gpuTable[b].sub  \in 0..(MiBBytes - 1)

SlabCountBounded ==
    /\ slabCount <= MaxSlabs
    /\ \A s \in 1..slotCount : SlotSlab(s) \in 0..(MaxSlabs - 1)

\* A 62-bit overflow must never clobber the top-2 slab selector bits.
NoOffsetClobbersSlab ==
    \A b \in 0..(MaxBlocks - 1) :
        tableSlot[b] # 0 =>
            LET s    == tableSlot[b]
                offW == WithinSlabWord(kind, SlotWithin(s))
            IN /\ FitsIn62Bits(offW)
               /\ gpuTable[b].top2 = SlotSlab(s)

\* Every bound GPU block-table word decodes through `kv_blk_ptr` + `bind_pool_slabs`
\* to the exact owning slab buffer, Metal buffer slot (3, 10, 11, 12), and byte address
\* for both the slot start and the last element row inside the slot.
ShaderPtrResolves ==
    \A b \in 0..(MaxBlocks - 1) :
        tableSlot[b] # 0 =>
            LET s       == tableSlot[b]
                slabI   == SlotSlab(s)
                offW    == WithinSlabWord(kind, SlotWithin(s))
                lastMiB == StrideMiB(kind) - 1
                lastSub == MiBBytes - (128 * Kvb)
                d0      == ShaderDecode(gpuTable[b], slabCount, 0, 0)
                dLast   == ShaderDecode(gpuTable[b], slabCount, lastMiB, lastSub)
            IN /\ slabI \in 0..3
               /\ d0.bufIdx = BufferIndices[slabI + 1]
               /\ d0.slab   = slabI
               /\ d0.addr   = offW
               /\ dLast.slab = slabI
               /\ dLast.addr = AddInSlotBytes(offW, lastMiB, lastSub)
               /\ WordLt(dLast.addr, AddInSlotBytes(offW, StrideMiB(kind), 0))

\* Distinct slots in the same slab have non-overlapping 62-bit byte ranges;
\* slots in distinct slabs separate cleanly on `(slab_index << 62)`.
SlabNonOverlap ==
    \A b1, b2 \in 0..(MaxBlocks - 1) :
        (tableSlot[b1] # 0 /\ tableSlot[b2] # 0 /\ tableSlot[b1] < tableSlot[b2]) =>
            LET s1 == tableSlot[b1]
                s2 == tableSlot[b2]
                d1 == ShaderDecode(gpuTable[b1], slabCount, 0, 0)
                d2 == ShaderDecode(gpuTable[b2], slabCount, 0, 0)
            IN IF SlotSlab(s1) # SlotSlab(s2)
               THEN d1.slab < d2.slab /\ d1.bufIdx # d2.bufIdx
               ELSE WordLe(AddInSlotBytes(d1.addr, StrideMiB(kind), 0), d2.addr)

=============================================================================
