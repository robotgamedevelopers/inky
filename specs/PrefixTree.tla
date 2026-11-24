----------------------------- MODULE PrefixTree -----------------------------
\* The Layer 3 prefix matcher (src/prefix.rs, PrefixTree and LiveTree):
\* a request matches the deepest retained start point in a tree of
\* sealed blocks.
\*
\* Unlike the first version of this model, blocks have token content:
\* a block is BlockLen tokens over Alphabet, so a block that differs
\* in its second token is a different block and MatchExact is a real
\* property, not a construction. A prefix is a token sequence whose
\* length is a whole number of blocks.
\*
\* Each sealed block also carries the pool facts the live matcher
\* checks: the slot it was sealed into and that slot's generation at
\* seal time (slotOf, genOf), against the slot's current generation
\* (poolGen). Reuse bumps a slot's generation; a block sealed at an
\* older generation is stale and must never be returned. refs counts
\* the one ref the tree holds per sealed block. starts counts the
\* holders of a start point (a lane head and a prompt end are two).
\*
\* The tree stops changing once a seek has run (~sought on every
\* mutating action): the checked property is about one seek against
\* the tree as the caller sees it at call time, as in the code.
\*
\* Bug values are the reversed witnesses, one per config:
\*   "shallow"     Seek returns the shallowest live start (DeepestStart)
\*   "stale"       Seek ignores generations (ResultLive)
\*   "first-token" a block matches on its first token only (MatchExact)
\*   "any-parent"  blocks match anywhere, not on one chain (ResultIsStart)
\*   "partial"     the result may end mid-block (StartOnBoundary)
\*   "leak"        Seek evicts stale blocks but drops no refs
\*                 (RefsSettledAfterSeek)
\*   "reuse-held"  a slot is reused while the tree holds its ref
\*                 (HeldNeverStale)
\*
\* A seek also evicts: every stale sealed block (sealed at an older
\* generation of its slot) leaves the tree with its subtree, its
\* start holds released and its ref dropped — the model's version of
\* LiveTree evicting on a match and re-seeking. The code evicts only
\* the stale block on the matched path; evicting every stale block
\* here is stronger and the checked properties hold under both.
\* The request being matched is fixed in the module because a TLC
\* config file assigns only literals and cannot write a sequence.

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Alphabet, BlockLen, MaxBlocks, Slots, MaxGen, MaxHolders, MaxSealed, Bug

ASSUME /\ BlockLen \in Nat \ {0}
       /\ MaxBlocks \in Nat \ {0}
       /\ MaxGen \in Nat
       /\ MaxHolders \in Nat \ {0}
       /\ MaxSealed \in Nat

\* Two blocks of two tokens; must satisfy Len(Req) = MaxBlocks * BlockLen.
Req == <<1, 2, 1, 2>>

VARIABLES sealed, starts, slotOf, genOf, poolGen, refs, result, sought
vars == <<sealed, starts, slotOf, genOf, poolGen, refs, result, sought>>

L0 == {<<>>}
L1 == {<<a>> : a \in Alphabet}
L2 == {<<a, b>> : a, b \in Alphabet}
L3 == {<<a, b, c>> : a, b, c \in Alphabet}
L4 == {<<a, b, c, d>> : a, b, c, d \in Alphabet}
\* Every token sequence the model can name (results may end mid-block).
AllTok == L0 \cup L1 \cup L2 \cup L3 \cup L4
\* Sequences at block boundaries: the only things ever sealed or held.
BlockSeqs == L0 \cup L2 \cup L4

IsPrefix(p, s) ==
    /\ Len(p) <= Len(s)
    /\ \A i \in 1..Len(p) : p[i] = s[i]

DropBlock(s) == SubSeq(s, 1, Len(s) - BlockLen)

BlockAt(s, k) == SubSeq(s, (k - 1) * BlockLen + 1, k * BlockLen)

\* Every block-boundary prefix of s is sealed at its slot's current
\* generation. The empty prefix needs nothing and is always live.
Live(s) ==
    \A q \in BlockSeqs :
        (IsPrefix(q, s) /\ Len(q) > 0) =>
            q \in sealed /\ genOf[q] = poolGen[slotOf[q]]

Held == {s \in BlockSeqs : starts[s] > 0}

Longest(cands, r) ==
    CHOOSE p \in cands : \A q \in cands : Len(q) <= Len(p)

DeepestLive(r) ==
    Longest({p \in Held \cup {<<>>} : IsPrefix(p, r) /\ Live(p)}, r)

Init ==
    /\ sealed = {<<>>}
    /\ starts = [s \in BlockSeqs |-> 0]
    /\ slotOf = [s \in BlockSeqs |-> 0]
    /\ genOf = [s \in BlockSeqs |-> 0]
    /\ poolGen = [s \in Slots |-> 0]
    /\ refs = [s \in BlockSeqs |-> 0]
    /\ result = <<>>
    /\ sought = FALSE

\* A block is published only sealed, only onto a sealed parent, and
\* only within the modelled depth. It takes the tree's one ref and
\* records the slot's generation at seal time.
Seal(s, slot) ==
    /\ ~sought
    /\ s \in BlockSeqs
    /\ Len(s) \in {BlockLen, 2 * BlockLen}
    /\ Len(s) <= MaxBlocks * BlockLen
    /\ DropBlock(s) \in sealed
    /\ s \notin sealed
    /\ slot \in Slots
    /\ sealed' = sealed \cup {s}
    /\ slotOf' = [slotOf EXCEPT ![s] = slot]
    /\ genOf' = [genOf EXCEPT ![s] = poolGen[slot]]
    /\ refs' = [refs EXCEPT ![s] = 1]
    /\ UNCHANGED <<starts, poolGen, result, sought>>

Retain(s) ==
    /\ ~sought
    /\ s \in sealed \ {<<>>}
    /\ starts[s] < MaxHolders
    /\ starts' = [starts EXCEPT ![s] = starts[s] + 1]
    /\ UNCHANGED <<sealed, slotOf, genOf, poolGen, refs, result, sought>>

Release(s) ==
    /\ ~sought
    /\ starts[s] > 0
    /\ starts' = [starts EXCEPT ![s] = starts[s] - 1]
    /\ UNCHANGED <<sealed, slotOf, genOf, poolGen, refs, result, sought>>

\* The pool reuses a slot only once the tree holds no ref on it:
\* the generation moves, and any block still naming the slot at the
\* old generation would be stale. Three witness configs lift that
\* guard (stale, leak, reuse-held) so staleness can arise at all.
Reuse(slot) ==
    /\ ~sought
    /\ poolGen[slot] < MaxGen
    /\ Bug \in {"stale", "leak", "reuse-held"}
        \/ \A s \in sealed : slotOf[s] = slot => refs[s] = 0
    /\ poolGen' = [poolGen EXCEPT ![slot] = poolGen[slot] + 1]
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, refs, result, sought>>

\* Removal takes a block and everything below it out of the tree and
\* drops their refs and start holds.
Evict(s) ==
    /\ ~sought
    /\ s \in sealed \ {<<>>}
    /\ sealed' = {t \in sealed : ~IsPrefix(s, t)}
    /\ starts' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE starts[x]]
    /\ refs' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE refs[x]]
    /\ slotOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE slotOf[x]]
    /\ genOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE genOf[x]]
    /\ UNCHANGED <<poolGen, result, sought>>

FirstTokenMatch(p, r) ==
    /\ Len(p) <= Len(r)
    /\ \A k \in 1..(Len(p) \div BlockLen) :
           p[(k - 1) * BlockLen + 1] = r[(k - 1) * BlockLen + 1]

\* The longest request prefix whose blocks each appear as some sealed
\* block's last block, anywhere in the tree (the parent chain ignored).
AnyParentResult ==
    LET ok(n) == \A k \in 1..(n \div BlockLen) :
            \E t \in sealed :
                Len(t) >= BlockLen /\ BlockAt(t, Len(t) \div BlockLen) = BlockAt(Req, k)
        ns == {n \in {0, BlockLen, 2 * BlockLen} : n <= Len(Req) /\ ok(n)}
    IN  SubSeq(Req, 1, CHOOSE n \in ns : \A m \in ns : m <= n)

SeekResult ==
    CASE Bug = "shallow" ->
            LET c == {p \in Held \cup {<<>>} : IsPrefix(p, Req) /\ Live(p)}
            IN  CHOOSE p \in c : \A q \in c : Len(p) <= Len(q)
      [] Bug = "stale" ->
            Longest({p \in Held \cup {<<>>} : IsPrefix(p, Req)}, Req)
      [] Bug = "first-token" ->
            Longest({p \in Held \cup {<<>>} : FirstTokenMatch(p, Req)}, Req)
      [] Bug = "any-parent" -> AnyParentResult
      [] Bug = "partial" ->
            LET d == DeepestLive(Req)
            IN  IF Len(d) < Len(Req) THEN SubSeq(Req, 1, Len(d) + 1) ELSE d
      [] OTHER -> DeepestLive(Req)

\* Stale blocks and everything below them: what a seek evicts.
StaleSet == {s \in sealed : Len(s) > 0 /\ genOf[s] # poolGen[slotOf[s]]}
Evicted == {t \in sealed : \E s \in StaleSet : IsPrefix(s, t)}

Seek ==
    /\ result' = SeekResult
    /\ sought' = TRUE
    /\ sealed' = sealed \ Evicted
    /\ starts' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE starts[x]]
    /\ refs' = [x \in BlockSeqs |->
           IF Bug = "leak" \/ x \notin Evicted THEN refs[x] ELSE 0]
    /\ slotOf' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE slotOf[x]]
    /\ genOf' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE genOf[x]]
    /\ UNCHANGED <<poolGen>>

Next ==
    \/ \E s \in BlockSeqs, slot \in Slots : Seal(s, slot)
    \/ \E s \in BlockSeqs : Retain(s) \/ Release(s) \/ Evict(s)
    \/ \E slot \in Slots : Reuse(slot)
    \/ Seek

\* A state bound for the small configs: the per-PR depth-2 check keeps
\* every rule but only explores trees of at most MaxSealed blocks.
SealedBound == Cardinality(sealed) <= MaxSealed

\* The per-PR bound further: one chain only. A match walks a single
\* chain from the root, so the deepest-start properties live there;
\* branching is covered by the Rust tests and the any-parent witness.
ChainBound ==
    \A a, b \in sealed : IsPrefix(a, b) \/ IsPrefix(b, a)

FastBound == SealedBound /\ ChainBound

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ sealed \subseteq BlockSeqs
    /\ starts \in [BlockSeqs -> 0..MaxHolders]
    /\ slotOf \in [BlockSeqs -> Slots \cup {0}]
    /\ genOf \in [BlockSeqs -> 0..MaxGen]
    /\ poolGen \in [Slots -> 0..MaxGen]
    /\ refs \in [BlockSeqs -> 0..1]
    /\ result \in AllTok
    /\ sought \in BOOLEAN

\* The tree is prefix-closed at block granularity.
TreeClosed ==
    \A s \in sealed : Len(s) >= BlockLen => DropBlock(s) \in sealed

\* Only sealed blocks are ever held as start points.
StartsSealed == \A s \in BlockSeqs : starts[s] > 0 => s \in sealed

\* A ref is only ever held on a sealed block.
RefsOnSealed == \A s \in BlockSeqs : refs[s] = 1 => s \in sealed \ {<<>>}

\* After a seek the books are exact: one ref per sealed block, none
\* for anything evicted. The "leak" witness breaks exactly this.
RefsSettledAfterSeek ==
    sought => \A s \in BlockSeqs :
        refs[s] = (IF s \in sealed \ {<<>>} THEN 1 ELSE 0)

\* A block the tree holds a ref on is never stale: its slot has not
\* been reused under it. The "reuse-held" witness breaks exactly this.
HeldNeverStale ==
    \A s \in sealed \ {<<>>} :
        refs[s] = 1 => genOf[s] = poolGen[slotOf[s]]

\* The result is a held start point (or position 0).
ResultIsStart == sought => result \in Held \cup {<<>>}

\* The result is an exact token prefix of the request.
MatchExact == sought => IsPrefix(result, Req)

\* The result never ends in the middle of a block.
StartOnBoundary == sought => Len(result) = BlockLen * (Len(result) \div BlockLen)

\* No block on the returned path is stale or unsealed.
ResultLive == sought => Live(result)

\* The result is the deepest live held start point.
DeepestStart == sought => result = DeepestLive(Req)
=============================================================================
