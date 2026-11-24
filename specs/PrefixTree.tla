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
\* for the Window entry), against the slots' current generation
\* (poolGen). Reuse bumps a slot's generation; a block sealed at an
\* older generation is stale and must never be returned. refs is the
\* pin: a cached block sits at refs = 0, evictable, and lookup
\* validates it by generation; a lane pins a block it steps from
\* (refs = 1), and only then does the pool hold a ref for it. starts
\* counts the holders of a start point (a lane head and a prompt end
\* are two).
\*
\* A mutation after a seek consumes that seek's result (sought and
\* result reset): the checked properties speak about the tree as the
\* caller sees it at the moment of a seek, and the code's tree keeps
\* changing after one, as does the model's.
\*
\* Bug values are the reversed witnesses, one per config:
\*   "shallow"      Seek returns the shallowest live start (DeepestStart)
\*   "stale"        Seek ignores generations (ResultLive)
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
\*
\* A seek also evicts: every stale sealed block (sealed at an older
\* generation of either of its slots) leaves the tree with its
\* subtree, its start holds released and any pin dropped. The code
\* does the same: LiveTree's seek evicts every stale block before
\* matching. A seek never returns the whole request: the final token
\* must remain to prefill, so an exact full match steps back one
\* block (ExpectedResult).
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

VARIABLES sealed, starts, slotOf, genOf, winSlotOf, winGenOf, poolGen,
          refs, result, sought, badInsert
vars == <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf, poolGen,
          refs, result, sought, badInsert>>

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

WinLive(s) == IF winSlotOf[s] = 0 THEN TRUE ELSE winGenOf[s] = poolGen[winSlotOf[s]]

\* Every block-boundary prefix of s is sealed at its slots' current
\* generations. The empty prefix needs nothing and is always live.
Live(s) ==
    \A q \in BlockSeqs :
        (IsPrefix(q, s) /\ Len(q) > 0) =>
            q \in sealed /\ genOf[q] = poolGen[slotOf[q]] /\ WinLive(q)

Held == {s \in BlockSeqs : starts[s] > 0}

Longest(cands, r) ==
    CHOOSE p \in cands : \A q \in cands : Len(q) <= Len(p)

DeepestLive(r) ==
    Longest({p \in Held \cup {<<>>} : IsPrefix(p, r) /\ Live(p)}, r)

\* The correct seek result: the deepest live held start, stepped back
\* one block when it is the whole request, so a token remains to
\* prefill.
ExpectedResult ==
    LET d == DeepestLive(Req)
    IN  IF d = Req THEN DeepestLive(DropBlock(Req)) ELSE d

Init ==
    /\ sealed = {<<>>}
    /\ starts = [s \in BlockSeqs |-> 0]
    /\ slotOf = [s \in BlockSeqs |-> 0]
    /\ genOf = [s \in BlockSeqs |-> 0]
    /\ winSlotOf = [s \in BlockSeqs |-> 0]
    /\ winGenOf = [s \in BlockSeqs |-> 0]
    /\ poolGen = [s \in Slots |-> 0]
    /\ refs = [s \in BlockSeqs |-> 0]
    /\ result = <<>>
    /\ sought = FALSE
    /\ badInsert = FALSE

\* A block is published only sealed, only onto a sealed parent, and
\* only within the modelled depth. It records its slots' generations
\* at seal time and takes no ref: cached, it is evictable until a
\* lane pins it. wslot = 0 seals a block with no Window entry.
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
    /\ winGenOf' = [winGenOf EXCEPT ![s] = IF wslot = 0 THEN 0 ELSE poolGen[wslot]]
    /\ refs' = [refs EXCEPT ![s] = 0]
    /\ UNCHANGED <<starts, poolGen, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

\* The stale-insert bug, expressible only as its own action: an insert
\* arrives claiming a generation its slot has moved past, and is
\* accepted anyway — for fresh tokens, or for a block already in the
\* tree (the stale twin: the name is re-sealed at the claimed, older
\* generation). Seal cannot write it, because Seal records the slot's
\* current generation and refuses a block already sealed. The code
\* refuses this insert (LiveTree, StaleEntry); under Bug =
\* "stale-insert" the model accepts it and NoBadInsert falls.
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
    /\ UNCHANGED <<starts, poolGen>>
    /\ sought' = FALSE
    /\ result' = <<>>

\* Inserting a block that is already in the tree, exactly: the code
\* returns the block it has and takes no second ref. The tree keeps
\* its own entries, whichever slots the caller named.
Dedup(s) ==
    /\ s \in sealed \ {<<>>}
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf,
                    poolGen, refs, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

\* Inserting a fresh entry for a block whose twin in the tree is
\* stale: the code evicts the twin (its pin dropped, everything below
\* it gone, its start holds released) and seals the fresh entries in
\* its place, at their slots' current generations, unpinned.
ReSeal(s, slot, wslot) ==
    /\ s \in sealed \ {<<>>}
    /\ ~Live(s)
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
            IF x = s THEN (IF wslot = 0 THEN 0 ELSE poolGen[wslot])
            ELSE IF IsPrefix(s, x) THEN 0 ELSE winGenOf[x]]
    /\ refs' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE refs[x]]
    /\ UNCHANGED <<poolGen, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

\* A lane pins a live block it is stepping from; only then does the
\* pool hold a ref for it. A stale block cannot be pinned.
Pin(s) ==
    /\ s \in sealed \ {<<>>}
    /\ refs[s] = 0
    /\ Live(s)
    /\ refs' = [refs EXCEPT ![s] = 1]
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf,
                    poolGen, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

Unpin(s) ==
    /\ s \in sealed \ {<<>>}
    /\ refs[s] = 1
    /\ refs' = [refs EXCEPT ![s] = 0]
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf,
                    poolGen, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

Retain(s) ==
    /\ s \in sealed \ {<<>>}
    /\ starts[s] < MaxHolders
    /\ starts' = [starts EXCEPT ![s] = starts[s] + 1]
    /\ UNCHANGED <<sealed, slotOf, genOf, winSlotOf, winGenOf, poolGen, refs, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

Release(s) ==
    /\ starts[s] > 0
    /\ starts' = [starts EXCEPT ![s] = starts[s] - 1]
    /\ UNCHANGED <<sealed, slotOf, genOf, winSlotOf, winGenOf, poolGen, refs, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

\* The pool reuses a slot only once no pinned block names it (on
\* either entry): the generation moves, and any block still naming
\* the slot at the old generation is stale. Three witness configs
\* lift that guard (stale, leak, reuse-held) so staleness can arise
\* at all, and the "evict" config lifts it with a correct Seek so
\* eviction is exercised in a passing config.
Reuse(slot) ==
    /\ poolGen[slot] < MaxGen
    /\ Bug \in {"stale", "leak", "reuse-held", "evict"}
        \/ \A s \in sealed : (slotOf[s] = slot \/ winSlotOf[s] = slot) => refs[s] = 0
    /\ poolGen' = [poolGen EXCEPT ![slot] = poolGen[slot] + 1]
    /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf, refs, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

\* Removal takes a block and everything below it out of the tree and
\* drops their pins and start holds.
Evict(s) ==
    /\ s \in sealed \ {<<>>}
    /\ sealed' = {t \in sealed : ~IsPrefix(s, t)}
    /\ starts' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE starts[x]]
    /\ refs' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE refs[x]]
    /\ slotOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE slotOf[x]]
    /\ genOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE genOf[x]]
    /\ winSlotOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE winSlotOf[x]]
    /\ winGenOf' = [x \in BlockSeqs |-> IF IsPrefix(s, x) THEN 0 ELSE winGenOf[x]]
    /\ UNCHANGED <<poolGen, badInsert>>
    /\ sought' = FALSE
    /\ result' = <<>>

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
      [] OTHER -> ExpectedResult

\* Stale blocks and everything below them: what a seek evicts.
StaleSet == {s \in sealed : Len(s) > 0 /\ (genOf[s] # poolGen[slotOf[s]] \/ ~WinLive(s))}
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
    /\ winSlotOf' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE winSlotOf[x]]
    /\ winGenOf' = [x \in BlockSeqs |-> IF x \in Evicted THEN 0 ELSE winGenOf[x]]
    /\ UNCHANGED <<poolGen, badInsert>>

Next ==
    \/ \E s \in BlockSeqs, slot \in Slots, wslot \in Slots \cup {0} : Seal(s, slot, wslot)
    \/ \E s \in BlockSeqs, slot \in Slots, gen \in 0..MaxGen : SealClaimed(s, slot, gen)
    \/ \E s \in BlockSeqs : Dedup(s)
    \/ \E s \in BlockSeqs, slot \in Slots, wslot \in Slots \cup {0} : ReSeal(s, slot, wslot)
    \/ \E s \in BlockSeqs : Pin(s) \/ Unpin(s)
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
    /\ winSlotOf \in [BlockSeqs -> Slots \cup {0}]
    /\ winGenOf \in [BlockSeqs -> 0..MaxGen]
    /\ poolGen \in [Slots -> 0..MaxGen]
    /\ refs \in [BlockSeqs -> 0..1]
    /\ result \in AllTok
    /\ sought \in BOOLEAN
    /\ badInsert \in BOOLEAN

\* The tree is prefix-closed at block granularity.
TreeClosed ==
    \A s \in sealed : Len(s) >= BlockLen => DropBlock(s) \in sealed

\* Only sealed blocks are ever held as start points.
StartsSealed == \A s \in BlockSeqs : starts[s] > 0 => s \in sealed

\* A pin is only ever held on a sealed block.
RefsOnSealed == \A s \in BlockSeqs : refs[s] = 1 => s \in sealed \ {<<>>}

\* After a seek no pin survives on an evicted block: pins live only
\* on sealed blocks, and a seek has evicted every stale one. The
\* "leak" witness breaks exactly this.
RefsSettledAfterSeek ==
    sought => \A s \in BlockSeqs : refs[s] = 1 => s \in sealed \ {<<>>}

\* A pinned block is never stale: its slots have not been reused
\* under it. The "reuse-held" witness breaks exactly this.
HeldNeverStale ==
    \A s \in sealed \ {<<>>} :
        refs[s] = 1 => Live(s)

\* No insert was ever accepted at a generation its slot had moved
\* past. The "stale-insert" witness breaks exactly this.
NoBadInsert == ~badInsert

\* After a seek no stale block remains sealed: the seek evicted every
\* one of them, on the matched path or not. In the guarded configs
\* this is vacuous (nothing can be stale at a seek); the "evict"
\* config is where it bites.
NoStaleAfterSeek == sought => StaleSet = {}

\* The result is a held start point (or position 0).
ResultIsStart == sought => result \in Held \cup {<<>>}

\* The result is an exact token prefix of the request.
MatchExact == sought => IsPrefix(result, Req)

\* The result never ends in the middle of a block.
StartOnBoundary == sought => Len(result) = BlockLen * (Len(result) \div BlockLen)

\* No block on the returned path is stale or unsealed.
ResultLive == sought => Live(result)

\* The result is the deepest live held start point, stepped back one
\* block when that is the whole request.
DeepestStart == sought => result = ExpectedResult
=============================================================================
