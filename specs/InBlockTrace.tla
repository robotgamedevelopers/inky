------------------------------ MODULE InBlockTrace -----------------------
\* Trace conformance spec binding BOTH `src/blocks.rs::key_row_at` (small bounds)
\* AND the Metal shader in-slot address formula parsed from `src/kernels.metal`
\* and `src/prefill.metal` (`KV_HEAD_BYTES`, `KV_PLANE_BYTES`, `KV_PASS_BYTES`,
\* `KV_BLK_MASK`) at production `Full` (8 passes) and `Window` (40 passes)
\* geometry across fp16 (`Kvb = 2`) and `--kv8` (`Kvb = 1`).
EXTENDS Integers, Sequences, InBlockTraceData

CONSTANTS Kvb
ASSUME Kvb \in {1, 2}

SmallHeads == 3
SmallB     == 4
SmallHD    == 2

ProdHeads      == 8
ProdB          == 64
ProdHD         == 128
ProdFullPasses == 8
ProdWinPasses  == 40

SmallFormula(p, k, h, i) ==
  ((p * 2 + k) * SmallHeads + h) * SmallB * SmallHD * Kvb + i * SmallHD * Kvb

ProdFormula(p, k, h, i) ==
  ((p * 2 + k) * ProdHeads + h) * ProdB * ProdHD * Kvb + i * ProdHD * Kvb

ProdStride(passes) == passes * 2 * ProdHeads * ProdB * ProdHD * Kvb

SmallTrace == IF Kvb = 2 THEN Trace2     ELSE Trace1
FullTrace  == IF Kvb = 2 THEN MetalFull2 ELSE MetalFull1
WinTrace   == IF Kvb = 2 THEN MetalWin2  ELSE MetalWin1

TotalLen == Len(SmallTrace) + Len(FullTrace) + Len(WinTrace)

EntryAt(idx) ==
  IF idx <= Len(SmallTrace)
  THEN <<"small", SmallTrace[idx]>>
  ELSE IF idx <= Len(SmallTrace) + Len(FullTrace)
       THEN <<"full", FullTrace[idx - Len(SmallTrace)]>>
       ELSE <<"window", WinTrace[idx - Len(SmallTrace) - Len(FullTrace)]>>

VARIABLES n, mode, p, k, h, i, addr
vars == <<n, mode, p, k, h, i, addr>>

Init ==
  /\ n = 1
  /\ LET cur == EntryAt(1)
         row == cur[2]
     IN /\ mode = cur[1]
        /\ p = row[1] /\ k = row[2] /\ h = row[3] /\ i = row[4]
        /\ addr = row[5]

Next ==
  \/ /\ n < TotalLen
     /\ n' = n + 1
     /\ LET cur == EntryAt(n')
            row == cur[2]
        IN /\ mode' = cur[1]
           /\ p' = row[1] /\ k' = row[2] /\ h' = row[3] /\ i' = row[4]
           /\ addr' = row[5]
           /\ <<mode', p', k', h', i'>> # <<mode, p, k, h, i>>
  \/ /\ n = TotalLen
     /\ UNCHANGED vars

Ordered ==
  IF mode = "small"
  THEN addr = SmallFormula(p, k, h, i)
  ELSE IF mode = "full"
       THEN /\ p \in 0..(ProdFullPasses - 1)
            /\ addr = ProdFormula(p, k, h, i)
            /\ addr + ProdHD * Kvb <= ProdStride(ProdFullPasses)
       ELSE /\ p \in 0..(ProdWinPasses - 1)
            /\ addr = ProdFormula(p, k, h, i)
            /\ addr + ProdHD * Kvb <= ProdStride(ProdWinPasses)

Spec == Init /\ [][Next]_vars
=============================================================================
