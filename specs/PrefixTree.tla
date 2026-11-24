----------------------------- MODULE PrefixTree -----------------------------
\* The Layer 3 prefix matcher (src/prefix.rs): a request matches the
\* deepest retained start point in a tree of sealed blocks.
\*
\* One alphabet symbol stands for one whole 64-position block: the code
\* validates every token of a block and equality is all matching ever
\* uses, so a block is abstracted to its identity here. A prefix is a
\* sequence of blocks; position 0 (the empty prefix) is always a start
\* point and needs no state. `Req` is the request being matched.
\*
\* Bug = "shallow" is the reversed witness: Seek returns the shallowest
\* matching start point instead of the deepest, and DeepestStart must
\* fail (specs/PrefixTreeShallow.cfg).

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Alphabet, MaxDepth, Req, Bug

ASSUME MaxDepth \in Nat /\ Len(Req) = MaxDepth

VARIABLES sealed, starts, result, sought
vars == <<sealed, starts, result, sought>>

AllSeqs ==
    {<<>>}
    \cup {<<a>> : a \in Alphabet}
    \cup {<<a, b>> : a, b \in Alphabet}
    \cup {<<a, b, c>> : a, b, c \in Alphabet}

IsPrefix(p, s) ==
    /\ Len(p) <= Len(s)
    /\ \A i \in 1..Len(p) : p[i] = s[i]

DropLast(s) == SubSeq(s, 1, Len(s) - 1)

\* The longest retained start point that is a prefix of the request.
\* Prefixes of one sequence are ordered by length, so this is unique.
Deepest(r) ==
    LET matching == {p \in starts \cup {<<>>} : IsPrefix(p, r)}
    IN  CHOOSE p \in matching :
            \A q \in matching : Len(q) <= Len(p)

Shallowest(r) ==
    LET matching == {p \in starts \cup {<<>>} : IsPrefix(p, r)}
    IN  CHOOSE p \in matching :
            \A q \in matching : Len(p) <= Len(q)

Init ==
    /\ sealed = {<<>>}
    /\ starts = {}
    /\ result = <<>>
    /\ sought = FALSE

\* A block is published only sealed and only onto a sealed parent.
Seal(s) ==
    /\ s \in AllSeqs
    /\ Len(s) >= 1
    /\ Len(s) <= MaxDepth
    /\ DropLast(s) \in sealed
    /\ s \notin sealed
    /\ sealed' = sealed \cup {s}
    /\ UNCHANGED <<starts, result, sought>>

Retain(s) ==
    /\ s \in sealed \ {<<>>}
    /\ s \notin starts
    /\ starts' = starts \cup {s}
    /\ UNCHANGED <<sealed, result, sought>>

Release(s) ==
    /\ s \in starts
    /\ starts' = starts \ {s}
    /\ UNCHANGED <<sealed, result, sought>>

Seek ==
    /\ result' = IF Bug = "shallow" THEN Shallowest(Req) ELSE Deepest(Req)
    /\ sought' = TRUE
    /\ UNCHANGED <<sealed, starts>>

Next ==
    \/ \E s \in AllSeqs : Seal(s) \/ Retain(s) \/ Release(s)
    \/ Seek

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ sealed \subseteq AllSeqs
    /\ starts \subseteq AllSeqs
    /\ result \in AllSeqs
    /\ sought \in BOOLEAN

\* The tree is prefix-closed: no block without its whole parent chain.
TreeClosed ==
    \A s \in sealed : Len(s) >= 1 => DropLast(s) \in sealed

\* Only sealed blocks are ever retained as start points.
StartsSealed == starts \subseteq sealed

\* The result is a retained start point and an exact prefix.
ResultIsStart == sought => result \in starts \cup {<<>>}
MatchExact == sought => IsPrefix(result, Req)

\* The result is the deepest such start point. This is the property the
\* reversed witness (Bug = "shallow") must violate.
DeepestStart == sought => result = Deepest(Req)

\* No start point is ever deeper than the matched prefix allows: the
\* result's length never passes the request, block by block.
NoPartialBlock == sought => Len(result) <= Len(Req)
=============================================================================
