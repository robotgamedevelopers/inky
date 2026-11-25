------------------------------- MODULE SwaRing -------------------------------
\* Layout and data-flow model for the Layer 2 sliding-window ring (`pos % W`)
\* and `k_kvcopy` linear prefill scratch round-trip
\* (`src/engine.rs:3056-3091`, `src/prefill.metal:534-542`):
\*   1. `k_zero` clears scratch `[0..Scap)` before prefill;
\*   2. `k_kvcopy(to_ring = 0)` copies `min(p0, W)` prior keys `[p0 - min(p0, W) .. p0)`
\*      from `ring[p % W]` into linear scratch `scratch[p - kpos0]`, where
\*      `kpos0 = p0 - (W + KGuard)`;
\*   3. `k_kvprep_b` writes the span's `t` keys `[p0 .. p0 + t)` into `scratch[p - kpos0]`;
\*   4. `k_attn_h` / `k_attn_b` reads `[max(0, q + 1 - W) .. q]` for each query `q`
\*      in `[p0 .. p0 + t)` plus masked tail rows up to `NextMult(p0 + t, Ak)`;
\*   5. `k_kvcopy(to_ring = 1)` writes the newest `min(t, W)` keys
\*      `[p0 + t - min(t, W) .. p0 + t)` back from `scratch` into `ring[p % W]`.

EXTENDS Integers, FiniteSets

CONSTANTS
    W,          \* Sliding-window width (`EXT = 1024` in production, 3 here)
    MaxPos,     \* Maximum stream position (7 here)
    MaxSpan,    \* Maximum prefill span length `t` (4 here, > W so wrap is exercised!)
    KGuard,     \* Scratch front guard rows (`KGUARD = 64` in production, 1 here)
    Ak,         \* Attention key tile alignment (`ATT_AK = 64` in production, 2 here)
    Bug         \* "none" | "short_writeback"

ASSUME
    /\ W >= 2
    /\ MaxSpan > W
    /\ MaxPos >= W + MaxSpan
    /\ KGuard >= 1
    /\ Ak >= 1
    /\ Bug \in {"none", "short_writeback"}

Scap == W + MaxSpan + 2 * KGuard
Poison == -1
KeyVal(p) == p + 1

Min(a, b) == IF a <= b THEN a ELSE b
Max(a, b) == IF a >= b THEN a ELSE b
NextMult(n, a) == ((n + a - 1) \div a) * a

VARIABLES
    pos,        \* Number of cached stream positions (`0..MaxPos`)
    ring,       \* [0..(W - 1) -> Int] sliding-window ring buffer
    scratch,    \* [0..(Scap - 1) -> Int] batch-arena linear scratch
    attnBad     \* BOOLEAN: TRUE if any prefill or decode query read a wrong/poisoned row

vars == <<pos, ring, scratch, attnBad>>

Init ==
    /\ pos     = 0
    /\ ring    = [i \in 0..(W - 1) |-> 0]
    /\ scratch = [i \in 0..(Scap - 1) |-> Poison]
    /\ attnBad = FALSE

\* One prefill span `[pos, pos + t)` through `k_zero` -> `k_kvcopy(0)` -> `k_kvprep_b`
\* -> `k_kvcopy(1)` -> `k_kvcopy(2)` -> `k_attn_h`.
PrefillSpan(t) ==
    /\ t \in 1..MaxSpan
    /\ pos + t <= MaxPos
    /\ LET p0     == pos
           kpos0  == p0 - (W + KGuard)
           histN  == Min(p0, W)
           histLo == p0 - histN
           \* 1. `k_zero` clears scratch at start of prefill step.
           scr0   == [i \in 0..(Scap - 1) |-> 0]
           \* 2. `k_kvcopy(..., to_ring = 0)` copies `[histLo .. p0)` from `ring[p % W]`.
           scr1   == [i \in 0..(Scap - 1) |->
                        LET p == i + kpos0
                        IN IF p >= histLo /\ p < p0
                           THEN ring[p % W]
                           ELSE scr0[i]]
           \* 3. `k_kvprep_b` writes `[p0 .. p0 + t)` into `scratch[p - kpos0]`.
           scr2   == [i \in 0..(Scap - 1) |->
                        LET p == i + kpos0
                        IN IF p >= p0 /\ p < p0 + t
                           THEN KeyVal(p)
                           ELSE scr1[i]]
           \* 4. `k_kvcopy(..., to_ring = 1)` writes newest `min(t, W)` keys to `ring[p % W]`.
           wbCnt  == IF Bug = "short_writeback" /\ t > 1
                     THEN Min(t - 1, W)
                     ELSE Min(t, W)
           wbLo   == p0 + t - wbCnt
           nring  == [r \in 0..(W - 1) |->
                        IF \E p \in wbLo..(p0 + t - 1) : p % W = r
                        THEN LET p == CHOOSE q \in wbLo..(p0 + t - 1) : q % W = r
                             IN scr2[p - kpos0]
                        ELSE ring[r]]
           \* 5. `k_kvcopy(..., to_ring = 2)` zeroes tail `[p0 + t .. NextMult(p0 + t, Ak))`.
           padEnd == NextMult(p0 + t, Ak)
           scr3   == [i \in 0..(Scap - 1) |->
                        LET p == i + kpos0
                        IN IF p >= p0 + t /\ p < padEnd
                           THEN 0
                           ELSE scr2[i]]
           \* 6. Verify every query `q \in p0..(p0 + t - 1)` reads exact keys in its window
           \*    and finite (non-Poison) guard/tail rows across `[kstart .. padEnd)`.
           qFirst == p0
           kStart == IF qFirst + 1 > W THEN qFirst + 1 - W ELSE 0
           tileLo == (kStart \div Ak) * Ak
           readsOk ==
               /\ \A q \in p0..(p0 + t - 1) :
                    LET wLo == IF q + 1 > W THEN q + 1 - W ELSE 0
                    IN \A kp \in wLo..q :
                        /\ kp - kpos0 \in 0..(Scap - 1)
                        /\ scr3[kp - kpos0] = KeyVal(kp)
               /\ \A kp \in tileLo..(padEnd - 1) :
                    /\ kp - kpos0 \in 0..(Scap - 1)
                    /\ scr3[kp - kpos0] # Poison
       IN /\ pos'     = p0 + t
          /\ ring'    = nring
          /\ scratch' = scr3
          /\ attnBad' = (attnBad \/ ~readsOk)

\* Single-token decode step at `pos`: writes `ring[pos % W]` directly and attends
\* over `k \in max(0, pos + 1 - W) .. pos` in `ring[k % W]`.
DecodeOne ==
    /\ pos < MaxPos
    /\ LET nring == [ring EXCEPT ![pos % W] = KeyVal(pos)]
           wLo   == IF pos + 1 > W THEN pos + 1 - W ELSE 0
           ok    == \A kp \in wLo..pos : nring[kp % W] = KeyVal(kp)
       IN /\ ring'    = nring
          /\ pos'     = pos + 1
          /\ attnBad' = (attnBad \/ ~ok)
          \* Poison scratch after decode so any later prefill that forgot `k_zero` / `k_kvcopy` fails!
          /\ scratch' = [i \in 0..(Scap - 1) |-> Poison]

Next ==
    \/ (\E t \in 1..MaxSpan : PrefillSpan(t))
    \/ DecodeOne

Spec == Init /\ [][Next]_vars

\* ---- Invariants -----------------------------------------------------------

TypeOK ==
    /\ pos \in 0..MaxPos
    /\ attnBad \in BOOLEAN

\* At every stream position `pos`, `ring[p % W]` holds exact `KeyVal(p)` for the
\* active sliding window `[max(0, pos - W) .. pos - 1]` (covering both subsequent
\* decode steps keeping `W - 1` prior + 1 current AND `k_kvcopy(to_ring = 0)`
\* copying `min(pos, W)` prior positions), and no attention query ever misreads.
RingHoldsExactWindow ==
    /\ ~attnBad
    /\ LET lo == IF pos > W THEN pos - W ELSE 0
       IN \A p \in lo..(pos - 1) : ring[p % W] = KeyVal(p)

=============================================================================
