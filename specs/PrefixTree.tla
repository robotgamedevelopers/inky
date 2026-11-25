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
\* checks: the slots it was sealed into and those slots' generations
\* at seal time (slotOf, genOf for the Full entry; winSlotOf, winGenOf
\* for the Window entry), against each kind's own current slot
\* generation (poolGen for Full, winPoolGen for Window). ReuseFull and
\* ReuseWindow bump Full and Window slot generations independently; a
\* block sealed at an older generation of either kind is stale and must
\* never be returned. refs is the pin: a cached block sits at refs = 0,
\* evictable, and lookup validates it by generation; a lane pins a block
\* it steps from (refs = 1), and only then does the pool hold a ref for
\* it. starts counts the holders of a start point (a lane head and a
\* prompt end are two).
\*
\* Two requests exercise both boundary cases within depth MaxBlocks = 2:
\*   Req     = <<1, 2, 1, 2, 1>> (two full blocks plus a trailing token,
\*             so both 1-block <<1, 2>> and 2-block <<1, 2, 1, 2>> are
\*             strict prefixes and deepest (2 blocks) is distinct from
\*             shallowest non-empty held start (1 block));
\*   ReqFull = <<1, 2, 1, 2>> (exact full-block match, which steps back
\*             one block so at least one token remains to prefill).
\*
\* Bug values are the reversed witnesses, one per config:
\*   "shallow"      Seek returns the shallowest non-empty live start (DeepestStart)
\*   "stale"        Seek ignores generations (ResultLive)
\*   "kind-mixup"   Seek checks Window entries against Full's poolGen (ResultLive)
\*   "first-token"  a block matches on its first token only (MatchExact)
\*   "any-parent"   blocks match anywhere, not on one chain (ResultIsStart)
\*   "partial"      the result may end mid-block (StartOnBoundary)
\*   "leak"         Seek evicts stale blocks but drops no refs
\*                  (RefsSettledAfterSeek)
\*   "reuse-held"   a slot is reused while a block on it is pinned
\*                  (HeldNeverStale)
\*   "stale-insert" an insert is accepted at a generation its slot has
\*                  moved past (NoBadInsert; SealClaimed below)
\*   "evict"        not a bug: the Reuse guard is lifted (the pool's
\*                  bump_generation has no such guard) but Seek stays
\*                  correct, so a passing config exercises eviction
\*                  itself (PrefixTreeEvictFast, NoStaleAfterSeek)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Alphabet, BlockLen, MaxBlocks, Slots, MaxGen, MaxHolders, MaxSealed, Bug

SlotSym == Permutations(Slots)

ASSUME /\ BlockLen \in Nat \ {0}
       /\ MaxBlocks \in Nat \ {0}
       /\ MaxGen \in Nat
       /\ MaxHolders \in Nat \ {0}
       /\ MaxSealed \in Nat

\* Exact 2-block request (exercises full-match step-back) and 2-block+1 request
\* (distinguishes 2-block deepest held start from 1-block shallowest held start).
ReqFull == <<1, 2, 1, 2>>
Req == <<1, 2, 1, 2, 1>>
Reqs == {ReqFull, Req}

VARIABLES sealed, starts, slotOf, genOf, winSlotOf, winGenOf, poolGen, winPoolGen,
          refs, result, sought, req, badInsert
vars == <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf, poolGen, winPoolGen,
          refs, result, sought, req, badInsert>>

L0 == {<<>>}
L1 == {<<a>> : a \in Alphabet}
L2 == {<<a, b>> : a, b \in Alphabet}
L3 == {<<a, b, c>> : a, b, c \in Alphabet}
L4 == {<<a, b, c, d>> : a, b, c, d \in Alphabet}
L5 == {Req}
\* Every token sequence the model can name (results may end mid-block).
AllTok == L0 \cup L1 \cup L2 \cup L3 \cup L4 \cup L5
\* Sequences at block boundaries: the only things ever sealed or held.
BlockSeqs == L0 \cup L2 \cup L4

IsPrefix(p, s) ==
    /\ Len(p) <= Len(s)
    /\ \A i \in 1..Len(p) : p[i] = s[i]

DropBlock(s) == SubSeq(s, 1, Len(s) - BlockLen)

BlockAt(s, k) == SubSeq(s, (k - 1) * BlockLen + 1, k * BlockLen)

WinLive(s) == IF winSlotOf[s] = 0 THEN TRUE ELSE winGenOf[s] = winPoolGen[winSlotOf[s]]

\* Under Bug = "kind-mixup", Window staleness mistakenly reads Full's poolGen.
WinLiveChecked(s) ==
    IF winSlotOf[s] = 0 THEN TRUE
    ELSE IF Bug = "kind-mixup" THEN winGenOf[s] = poolGen[winSlotOf[s]]
    ELSE winGenOf[s] = winPoolGen[winSlotOf[s]]

\* One sealed block's own entries are at their slots' current generations.
NodeLive(s) ==
    s \in sealed /\ genOf[s] = poolGen[slotOf[s]] /\ WinLive(s)

EntriesLive(s) == genOf[s] = poolGen[slotOf[s]] /\ WinLive(s)

\* Every block-boundary prefix of s is sealed at its slots' current
\* generations (Full in poolGen, Window in winPoolGen).
Live(s) ==
    \A q \in BlockSeqs :
        (IsPrefix(q, s) /\ Len(q) > 0) => NodeLive(q)

LiveChecked(s) ==
    \A q \in BlockSeqs :
        (IsPrefix(q, s) /\ Len(q) > 0) =>
            q \in sealed /\ genOf[q] = poolGen[slotOf[q]] /\ WinLiveChecked(q)

Held == {s \in BlockSeqs : starts[s] > 0}

Longest(cands, r) ==
    CHOOSE p \in cands : \A q \in cands : Len(q) <= Len(p)

DeepestLive(r) ==
    Longest({p \in Held \cup {<<>>} : IsPrefix(p, r) /\ Live(p)}, r)

DeepestLiveChecked(r) ==
    Longest({p \in Held \cup {<<>>} : IsPrefix(p, r) /\ LiveChecked(p)}, r)

\* The correct seek result for r: the deepest live held start, stepped
\* back one block when it covers the whole request, so a token remains
\* to prefill.
ExpectedResultFor(r) ==
    LET d == DeepestLive(r)
    IN  IF d = r THEN DeepestLive(DropBlock(r)) ELSE d

ExpectedResult == ExpectedResultFor(req)

Init ==
    /\ sealed = {<<>>}
    /\ starts = [s \in BlockSeqs |-> 0]
    /\ slotOf = [s \in BlockSeqs |-> 0]
    /\ genOf = [s \in BlockSeqs |-> 0]
    /\ winSlotOf = [s \in BlockSeqs |-> 0]
    /\ winGenOf = [s \in BlockSeqs |-> 0]
    /\ poolGen = [s \in Slots |-> 0]
    /\ winPoolGen = [s \in Slots |-> 0]
    /\ refs = [s \in BlockSeqs |-> 0]
    /\ result = <<>>
    /\ sought = FALSE
    /\ req = ReqFull
    /\ badInsert = FALSE

\* A block is published only sealed, only onto a sealed parent, and
\* only within the modelled depth. It records its Full slot's poolGen
\* and its Window slot's winPoolGen at seal time and takes no ref.
Seal(s, slot, wslot) ==
    /\ s \in BlockSeqs
    /\ Len(s) \in {BlockLen, 2 * BlockLen}
    /\ Len(s) <= MaxBlocks * BlockLen
    /\ DropBlock(s) \in sealed
    /\ s \notin sealed
    /\ slot \in Slots
    /\ wslot \in Slots \cup {0}
    /\ sealed' = sealed \cup {s}
    /\ slotOf' = [slotOf EXCEPT ![s] = slot]
    /\ genOf' = [genOf EXCEPT ![s] = poolGen[slot]]
    /\ winSlotOf' = [winSlotOf EXCEPT ![s] = wslot]
    /\ winGenOf' = [winGenOf EXCEPT ![s] = IF wslot = 0 THEN 0 ELSE winPoolGen[wslot]]
    /\ refs' = [refs EXCEPT ![s] = 0]
    /\ UNCHANGED <<starts, poolGen, winPoolGen, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

\* The stale-insert bug: an insert arrives claiming a generation its
\* Full or Window slot has moved past, and is accepted anyway.
SealClaimed(s, slot, gen) ==
    /\ Bug = "stale-insert"
    /\ s \in BlockSeqs
    /\ Len(s) \in {BlockLen, 2 * BlockLen}
    /\ Len(s) <= MaxBlocks * BlockLen
    /\ DropBlock(s) \in sealed
    /\ slot \in Slots
    /\ gen \in 0..MaxGen
    /\ gen # poolGen[slot]
    /\ sealed' = sealed \cup {s}
    /\ slotOf' = [slotOf EXCEPT ![s] = slot]
    /\ genOf' = [genOf EXCEPT ![s] = gen]
    /\ winSlotOf' = [winSlotOf EXCEPT ![s] = 0]
    /\ winGenOf' = [winGenOf EXCEPT ![s] = 0]
    /\ refs' = [refs EXCEPT ![s] = 0]
    /\ badInsert' = TRUE
    /\ UNCHANGED <<starts, poolGen, winPoolGen>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

\* Inserting a block that is already in the tree, exactly.
Dedup(s) ==
    /\ s \in sealed \ {<<>>}
    /\ EntriesLive(s)
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf,
                    poolGen, winPoolGen, refs, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

\* Inserting a fresh entry for a block whose twin in the tree is stale.
ReSeal(s, slot, wslot) ==
    /\ s \in sealed \ {<<>>}
    /\ ~NodeLive(s)
    /\ slot \in Slots
    /\ wslot \in Slots \cup {0}
    /\ sealed' = {u \in sealed : ~IsPrefix(s, u)} \cup {s}
    /\ starts' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE starts[x]]
    /\ slotOf' = [x \in BlockSeqs |->
            IF x = s THEN slot ELSE IF IsPrefix(s, x) THEN 0 ELSE slotOf[x]]
    /\ genOf' = [x \in BlockSeqs |->
            IF x = s THEN poolGen[slot] ELSE IF IsPrefix(s, x) THEN 0 ELSE genOf[x]]
    /\ winSlotOf' = [x \in BlockSeqs |->
            IF x = s THEN wslot ELSE IF IsPrefix(s, x) THEN 0 ELSE winSlotOf[x]]
    /\ winGenOf' = [x \in BlockSeqs |->
            IF x = s THEN (IF wslot = 0 THEN 0 ELSE winPoolGen[wslot])
            ELSE IF IsPrefix(s, x) THEN 0 ELSE winGenOf[x]]
    /\ refs' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE refs[x]]
    /\ UNCHANGED <<poolGen, winPoolGen, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

Pin(s) ==
    /\ s \in sealed \ {<<>>}
    /\ refs[s] = 0
    /\ NodeLive(s)
    /\ refs' = [refs EXCEPT ![s] = 1]
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf,
                    poolGen, winPoolGen, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

Unpin(s) ==
    /\ s \in sealed \ {<<>>}
    /\ refs[s] = 1
    /\ refs' = [refs EXCEPT ![s] = 0]
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf,
                    poolGen, winPoolGen, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

Retain(s) ==
    /\ s \in sealed \ {<<>>}
    /\ starts[s] < MaxHolders
    /\ starts' = [starts EXCEPT ![s] = starts[s] + 1]
    /\ UNCHANGED <<sealed, slotOf, genOf, winSlotOf, winGenOf, poolGen, winPoolGen, refs, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

Release(s) ==
    /\ starts[s] > 0
    /\ starts' = [starts EXCEPT ![s] = starts[s] - 1]
    /\ UNCHANGED <<sealed, slotOf, genOf, winSlotOf, winGenOf, poolGen, winPoolGen, refs, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

\* Full and Window pools reuse slots independently when no pinned block
\* names that kind's slot. Witness configs lift the guard so staleness
\* is reachable.
ReuseFull(slot) ==
    /\ poolGen[slot] < MaxGen
    /\ Bug \in {"stale", "kind-mixup", "leak", "reuse-held", "evict"}
        \/ \A s \in sealed : slotOf[s] = slot => refs[s] = 0
    /\ poolGen' = [poolGen EXCEPT ![slot] = poolGen[slot] + 1]
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf, winPoolGen, refs, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

ReuseWindow(wslot) ==
    /\ winPoolGen[wslot] < MaxGen
    /\ Bug \in {"stale", "kind-mixup", "leak", "reuse-held", "evict"}
        \/ \A s \in sealed : winSlotOf[s] = wslot => refs[s] = 0
    /\ winPoolGen' = [winPoolGen EXCEPT ![wslot] = winPoolGen[wslot] + 1]
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf, poolGen, refs, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

Evict(s) ==
    /\ s \in sealed \ {<<>>}
    /\ sealed' = {t \in sealed : ~IsPrefix(s, t)}
    /\ starts' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE starts[x]]
    /\ refs' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE refs[x]]
    /\ slotOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE slotOf[x]]
    /\ genOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE genOf[x]]
    /\ winSlotOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE winSlotOf[x]]
    /\ winGenOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE winGenOf[x]]
    /\ UNCHANGED <<poolGen, winPoolGen, badInsert>>
    /\ sought' = FALSE
    /\ req' = ReqFull
    /\ result' = <<>>

FirstTokenMatch(p, r) ==
    /\ Len(p) <= Len(r)
    /\ \A k \in 1..(Len(p) \div BlockLen) :
           p[(k - 1) * BlockLen + 1] = r[(k - 1) * BlockLen + 1]

AnyParentResult(r) ==
    LET ok(n) == \A k \in 1..(n \div BlockLen) :
            \E t \in sealed :
                Len(t) >= BlockLen /\ BlockAt(t, Len(t) \div BlockLen) = BlockAt(r, k)
        ns == {n \in {0, BlockLen, 2 * BlockLen} : n < Len(r) /\ ok(n)}
    IN  SubSeq(r, 1, CHOOSE n \in ns : \A m \in ns : m <= n)

SeekResult(r) ==
    CASE Bug = "shallow" ->
            LET nonEmpty == {p \in Held : Len(p) > 0 /\ p # r /\ IsPrefix(p, r) /\ Live(p)}
            IN  IF nonEmpty # {}
                THEN CHOOSE p \in nonEmpty : \A q \in nonEmpty : Len(p) <= Len(q)
                ELSE <<>>
      [] Bug = "stale" ->
            Longest({p \in Held \cup {<<>>} : p # r /\ IsPrefix(p, r)}, r)
      [] Bug = "kind-mixup" ->
            LET d == DeepestLiveChecked(r)
            IN  IF d = r THEN DeepestLiveChecked(DropBlock(r)) ELSE d
      [] Bug = "first-token" ->
            Longest({p \in Held \cup {<<>>} : p # r /\ FirstTokenMatch(p, r)}, r)
      [] Bug = "any-parent" -> AnyParentResult(r)
      [] Bug = "partial" ->
            LET d == DeepestLive(r)
            IN  IF Len(d) < Len(r) THEN SubSeq(r, 1, Len(d) + 1) ELSE d
      [] OTHER -> ExpectedResultFor(r)

\* Stale blocks and everything below them: what a seek evicts.
StaleSet == {s \in sealed : Len(s) > 0 /\ ~NodeLive(s)}
StaleSetChecked == {s \in sealed : Len(s) > 0 /\ (genOf[s] # poolGen[slotOf[s]] \/ ~WinLiveChecked(s))}
Evicted == {t \in sealed : \E s \in StaleSetChecked : IsPrefix(s, t)}

Seek(r) ==
    /\ r \in Reqs
    /\ result' = SeekResult(r)
    /\ req' = r
    /\ sought' = TRUE
    /\ sealed' = sealed \ Evicted
    /\ starts' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE starts[x]]
    /\ refs' = [x \in BlockSeqs |->
           IF Bug = "leak" \/ x \notin Evicted THEN refs[x] ELSE 0]
    /\ slotOf' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE slotOf[x]]
    /\ genOf' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE genOf[x]]
    /\ winSlotOf' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE winSlotOf[x]]
    /\ winGenOf' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE winGenOf[x]]
    /\ UNCHANGED <<poolGen, winPoolGen, badInsert>>

Next ==
    \/ \E s \in BlockSeqs, slot \in Slots, wslot \in Slots \cup {0} : Seal(s, slot, wslot)
    \/ \E s \in BlockSeqs, slot \in Slots, gen \in 0..MaxGen : SealClaimed(s, slot, gen)
    \/ \E s \in BlockSeqs : Dedup(s)
    \/ \E s \in BlockSeqs, slot \in Slots, wslot \in Slots \cup {0} : ReSeal(s, slot, wslot)
    \/ \E s \in BlockSeqs : Pin(s) \/ Unpin(s)
    \/ \E s \in BlockSeqs : Retain(s) \/ Release(s) \/ Evict(s)
    \/ \E slot \in Slots : ReuseFull(slot) \/ ReuseWindow(slot)
    \/ \E r \in Reqs : Seek(r)

SealedBound == Cardinality(sealed) <= MaxSealed

\* The per-PR bound further: one chain only. A match walks a single
\* chain from the root, so the deepest-start properties live there;
\* the nightly bound also explores siblings.
ChainBound ==
    \A a, b \in sealed : IsPrefix(a, b) \/ IsPrefix(b, a)

FastBound == SealedBound /\ ChainBound

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ sealed \subseteq BlockSeqs
    /\ starts \in [BlockSeqs -> 0..MaxHolders]
    /\ slotOf \in [BlockSeqs -> Slots \cup {0}]
    /\ genOf \in [BlockSeqs -> 0..MaxGen]
    /\ winSlotOf \in [BlockSeqs -> Slots \cup {0}]
    /\ winGenOf \in [BlockSeqs -> 0..MaxGen]
    /\ poolGen \in [Slots -> 0..MaxGen]
    /\ winPoolGen \in [Slots -> 0..MaxGen]
    /\ refs \in [BlockSeqs -> 0..1]
    /\ result \in AllTok
    /\ sought \in BOOLEAN
    /\ req \in Reqs
    /\ badInsert \in BOOLEAN

TreeClosed ==
    \A s \in sealed : Len(s) >= BlockLen => DropBlock(s) \in sealed

StartsSealed == \A s \in BlockSeqs : starts[s] > 0 => s \in sealed

RefsOnSealed == \A s \in BlockSeqs : refs[s] = 1 => s \in sealed \ {<<>>}

RefsSettledAfterSeek ==
    sought => \A s \in BlockSeqs : refs[s] = 1 => s \in sealed \ {<<>>}

HeldNeverStale ==
    \A s \in sealed \ {<<>>} :
        refs[s] = 1 => NodeLive(s)

\* A pin covers one node's entries. LiveTree::pin does not recursively
\* pin its ancestors. The serving lane binds the entire matched chain.
\* A completely pinned chain must remain suitable for reuse.
PinnedPathLive ==
    \A s \in sealed :
        (\A q \in sealed \ {<<>>} : IsPrefix(q, s) => refs[q] = 1) => Live(s)

NoBadInsert == ~badInsert

\* After a seek no stale block remains sealed: the seek evicted every
\* one of them, wherever they were in the tree.
NoStaleAfterSeek == sought => StaleSet = {}

ResultIsStart == sought => result \in Held \cup {<<>>}

MatchExact == sought => IsPrefix(result, req)

StartOnBoundary == sought => Len(result) = BlockLen * (Len(result) \div BlockLen)

ResultLive == sought => Live(result)

DeepestStart == sought => result = ExpectedResult
=============================================================================
