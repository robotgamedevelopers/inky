--------------------------- MODULE VerifyRollback ---------------------------
\* State and layout model for speculative verification (`Engine::encode_verify`,
\* `keep_state`, and `rollback_verify` in `src/engine.rs:2353-2470`):
\*   - A verify step runs span `[p0, p0 + 1 + k)` (`k` draft tokens after the
\*     re-encoded committed token at `p0`), snapshotting prior KV rows `[p0..p0+k]`
\*     and recording one convolution image per position `0..k` BEFORE `k_state_upd`
\*     overwrites `cs` with the end-of-span state.
\*   - `rollback_verify(n_keep)` (`n_keep \in 1..(1 + k)`) restores `cs` from image
\*     `n_keep - 1`, restores overwritten KV rows at `(p0 + n_keep)..(p0 + k)` from
\*     `kv_undo` (initialized to distinct non-zero sentinels `-(p + 1)` so clearing
\*     to zero instead of restoring pre-verify bytes is caught), advances committed
\*     `pos` to `p0 + n_keep`, and NEVER seals any block extending into a rejected tail.

EXTENDS Integers, FiniteSets, Sequences

CONSTANTS
    BlockSize,  \* Positions per block (2 here, 64 in production)
    MaxPos,     \* Maximum stream positions (4 or 6 here)
    MaxK,       \* Maximum draft count `k` per verify step (2 here -> span up to 3)
    Tokens,     \* Token alphabet, e.g. {1, 2}
    Bug         \* "none" | "off_by_one_cs" | "keep_rejected_kv" | "seal_rejected"

ASSUME
    /\ BlockSize >= 2
    /\ MaxPos >= 2 * BlockSize
    /\ MaxPos % BlockSize = 0
    /\ MaxK >= 1
    /\ Tokens # {} /\ 0 \notin Tokens
    /\ Bug \in {"none", "off_by_one_cs", "keep_rejected_kv", "seal_rejected"}

Blocks == 0..((MaxPos \div BlockSize) - 1)
InitSentinel(p) == -(p + 1)

\* Exact causal 3-tap shift register step matching `hh = <<h1, h2, x>>` (`src/kernels.metal:1319`).
ConvStep(cs, tok) == <<cs[2], cs[3], tok>>

RECURSIVE ConvAfterSeq(_, _)
ConvAfterSeq(cs, seq) ==
    IF Len(seq) = 0 THEN cs
    ELSE ConvAfterSeq(ConvStep(cs, Head(seq)), Tail(seq))

\* KV row signature written at position `p` given incoming `cs` and token `tok`.
KvRowVal(p, cs, tok) == ((p + 1) * 1000) + (cs[1] * 100) + (cs[2] * 10) + cs[3] + tok

VARIABLES
    phase,          \* "ready" | "verifying"
    pos,            \* Committed token count (`0..MaxPos`), waiting to encode `toks[pos]`
    committedToks,  \* Seq(Tokens) of length `pos` (tokens at `0..(pos - 1)`)
    cs,             \* Live 3-tap convolution state `<<h0, h1, h2>>`
    kv,             \* [0..(MaxPos - 1) -> Int] live KV rows
    sealed,         \* [Blocks -> BOOLEAN] sealed blocks in block table
    verifyP0,       \* `p0` of the in-flight verify span
    verifyK,        \* `k` of the in-flight verify span (`t = 1 + k`)
    verifyToks,     \* Seq(Tokens) of length `1 + k` run through `encode_verify`
    csSnap,         \* Seq of `1 + k` conv images recorded by `k_state_keep`
    kvUndo          \* [0..(MaxPos - 1) -> Int] KV snapshot before `encode_verify`

vars == <<
    phase, pos, committedToks, cs, kv, sealed,
    verifyP0, verifyK, verifyToks, csSnap, kvUndo
>>

InitCs == <<0, 0, 0>>

Init ==
    /\ phase         = "ready"
    /\ pos           = 0
    /\ committedToks = << >>
    /\ cs            = InitCs
    /\ kv            = [p \in 0..(MaxPos - 1) |-> InitSentinel(p)]
    /\ sealed        = [b \in Blocks |-> FALSE]
    /\ verifyP0      = 0
    /\ verifyK       = 0
    /\ verifyToks    = << >>
    /\ csSnap        = << >>
    /\ kvUndo        = [p \in 0..(MaxPos - 1) |-> InitSentinel(p)]

\* Normal 1-token sequential encode at `pos`: updates `kv[pos]`, advances `cs`,
\* and seals block `pos \div BlockSize` iff `pos + 1` completes the block.
EncodeSingle(tok) ==
    /\ phase = "ready"
    /\ pos < MaxPos
    /\ tok \in Tokens
    /\ LET row   == KvRowVal(pos, cs, tok)
           ncs   == ConvStep(cs, tok)
           npos  == pos + 1
           nseal == [b \in Blocks |-> sealed[b] \/ ((b + 1) * BlockSize <= npos)]
       IN /\ kv'            = [kv EXCEPT ![pos] = row]
          /\ cs'            = ncs
          /\ pos'           = npos
          /\ committedToks' = Append(committedToks, tok)
          /\ sealed'        = nseal
          /\ verifyK'       = 0
    /\ UNCHANGED <<phase, verifyP0, verifyToks, csSnap, kvUndo>>

\* `encode_verify` over `[pos, pos + 1 + k)` with candidate tokens `spanToks` (`Len = 1 + k`):
\* snapshots `kv`, writes `csSnap[1..(1 + k)]` via `k_state_keep`, updates `cs` to the end
\* of the span via `k_state_upd`, and writes draft KV rows without sealing unverified blocks.
EncodeVerify(k, spanToks) ==
    /\ phase = "ready"
    /\ k \in 1..MaxK
    /\ pos + 1 + k <= MaxPos
    /\ Len(spanToks) = 1 + k
    /\ LET snaps == [d \in 1..(1 + k) |-> ConvAfterSeq(cs, SubSeq(spanToks, 1, d))]
           inCs  == [d \in 1..(1 + k) |->
                        IF d = 1 THEN cs ELSE snaps[d - 1]]
           nkv   == [p \in 0..(MaxPos - 1) |->
                        IF p >= pos /\ p < pos + 1 + k
                        THEN KvRowVal(p, inCs[(p - pos) + 1], spanToks[(p - pos) + 1])
                        ELSE kv[p]]
       IN /\ phase'      = "verifying"
          /\ verifyP0'   = pos
          /\ verifyK'    = k
          /\ verifyToks' = spanToks
          /\ csSnap'     = snaps
          /\ kvUndo'     = kv
          /\ kv'         = nkv
          /\ cs'         = snaps[1 + k]
          /\ sealed'     = IF Bug = "seal_rejected"
                           THEN [b \in Blocks |-> sealed[b] \/ ((b + 1) * BlockSize <= pos + 1 + k)]
                           ELSE sealed
    /\ UNCHANGED <<pos, committedToks>>

\* `rollback_verify(n_keep)` (`n_keep \in 1..(1 + verifyK)`):
\*   - copies `csSnap[n_keep]` (0-based `n_keep - 1` in Rust) back into `cs`,
\*   - restores `kv[p]` from `kvUndo[p]` for rejected positions `p >= verifyP0 + n_keep`,
\*   - commits `accepted_end = verifyP0 + n_keep`, sealing ONLY blocks `<= accepted_end`.
RollbackVerify(nKeep) ==
    /\ phase = "verifying"
    /\ nKeep \in 1..(1 + verifyK)
    /\ LET endPos  == verifyP0 + nKeep
           snapIdx == IF Bug = "off_by_one_cs" /\ nKeep < 1 + verifyK
                      THEN nKeep + 1
                      ELSE nKeep
           nkv     == IF Bug = "keep_rejected_kv"
                      THEN kv
                      ELSE [p \in 0..(MaxPos - 1) |->
                                IF p >= endPos /\ p < verifyP0 + 1 + verifyK
                                THEN kvUndo[p]
                                ELSE kv[p]]
           nseal   == [b \in Blocks |-> sealed[b] \/ ((b + 1) * BlockSize <= endPos)]
       IN /\ phase'         = "ready"
          /\ cs'            = csSnap[snapIdx]
          /\ kv'            = nkv
          /\ pos'           = endPos
          /\ committedToks' = committedToks \o SubSeq(verifyToks, 1, nKeep)
          /\ sealed'        = nseal
    /\ UNCHANGED <<verifyP0, verifyK, verifyToks, csSnap, kvUndo>>

DoneStutter ==
    /\ phase = "ready" /\ pos = MaxPos
    /\ UNCHANGED vars

SpanSeqs(len) ==
    IF len = 2 THEN {<<a, b>> : a, b \in Tokens}
    ELSE {<<a, b, c>> : a, b, c \in Tokens}

Next ==
    \/ (\E tok \in Tokens : EncodeSingle(tok))
    \/ (\E k \in 1..MaxK : \E s \in SpanSeqs(1 + k) : EncodeVerify(k, s))
    \/ (\E nk \in 1..(1 + MaxK) : RollbackVerify(nk))
    \/ DoneStutter

Spec == Init /\ [][Next]_vars

\* ---- Ground-Truth Sequential Oracle & Invariants --------------------------

ExpectedKvAt(seq, p) ==
    LET prevCs == ConvAfterSeq(InitCs, SubSeq(seq, 1, p))
    IN KvRowVal(p, prevCs, seq[p + 1])

TypeOK ==
    /\ phase \in {"ready", "verifying"}
    /\ pos \in 0..MaxPos
    /\ Len(committedToks) = pos
    /\ sealed \in [Blocks -> BOOLEAN]

\* Whenever `phase = "ready"`, live `cs` and every committed KV row `[0..pos)`
\* match pure 1-token-at-a-time sequential execution bit-for-bit.
ConvMatchesSequential ==
    phase = "ready" =>
        /\ cs = ConvAfterSeq(InitCs, committedToks)
        /\ \A p \in 0..(pos - 1) : kv[p] = ExpectedKvAt(committedToks, p)

\* After `rollback_verify`, every rejected position `pos..(verifyP0 + verifyK)`
\* holds the exact pre-verify sentinel/bytes from `kvUndo` (never draft K/V or zeros).
RejectedTailRestored ==
    (phase = "ready" /\ verifyK > 0 /\ pos <= verifyP0 + verifyK) =>
        \A p \in pos..(verifyP0 + verifyK) : kv[p] = kvUndo[p]

\* No block past the committed `pos` is ever sealed (neither during `encode_verify` nor after rollback).
NoRejectedSeal ==
    \A b \in Blocks :
        sealed[b] => (b + 1) * BlockSize <= pos

=============================================================================
