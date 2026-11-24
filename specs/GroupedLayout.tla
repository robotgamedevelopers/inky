---------------------------- MODULE GroupedLayout ----------------------------
EXTENDS Integers, FiniteSets, Sequences

\* Snapshot of one attention pass. Both row tables stay pinned for this step.
\* A physical slot is named by its logical block and its first owning row.
CONSTANT Capacity, Bug
Rows == {1, 2}
BlockSize == 2
TileSize == 2
SplitSize == 4
WindowSize == 3
ASSUME /\ Capacity \in {3, 4}
       /\ Bug \in {"none", "localMatch", "donorWindow", "tilePartial"}

VARIABLES lengths, shared, windowed, tile, loaded, visits, partials
vars == <<lengths, shared, windowed, tile, loaded, visits, partials>>
Min(a,b) == IF a < b THEN a ELSE b
Max(a,b) == IF a > b THEN a ELSE b
LastBlock(n) == (n - 1) \div BlockSize
\* shared is the number of complete shared blocks before the histories fork.
Slot(r,b) == <<b, IF b < shared THEN 1 ELSE r>>
Resident == UNION {{Slot(r,b) : b \in 0..LastBlock(lengths[r])} : r \in Rows}
First(r) == IF windowed THEN Max(0, lengths[r] - WindowSize) ELSE 0
\* The first divergent block has different token IDs. Subsequent local token
\* IDs can match even though their ancestors, and therefore their K/V, differ.
Lookup(r,b) ==
    IF Bug = "localMatch" /\ r = 2 /\ b > shared
       /\ (b+1)*BlockSize <= Min(lengths[1],lengths[2])
    THEN Slot(1,b) ELSE Slot(r,b)
MaskStart(r) == IF Bug = "donorWindow" THEN First(1) ELSE First(r)

\* The scalar reference enumerates visible keys directly without staging tiles.
KeyRef(r) ==
    [i \in 1..(lengths[r]-First(r)) |->
        LET p == First(r)+i-1 IN <<p, Slot(r,p \div BlockSize), p \div SplitSize>>]
PartRef(r) ==
    LET a == First(r) \div SplitSize
        b == (lengths[r]-1) \div SplitSize
    IN [i \in 1..(b-a+1) |->
            LET s == a+i-1 IN <<s, Min((s+1)*SplitSize, lengths[r])-1>>]

TileCount == (Max(lengths[1],lengths[2]) + TileSize - 1) \div TileSize
TileStart == tile * TileSize
Split == TileStart \div SplitSize
Start(r) == Max(TileStart, MaskStart(r))
Stop(r) == Min(TileStart+TileSize, lengths[r])
Readers == {r \in Rows : Start(r) < Stop(r)}
Requests == {Lookup(r,TileStart \div BlockSize) : r \in Readers}
Uses(r,s) == r \in Readers /\ Lookup(r,TileStart \div BlockSize) = s
Keys(r,s) == [i \in 1..(Stop(r)-Start(r)) |-> <<Start(r)+i-1,s,Split>>]
EndPartial(r) == Bug = "tilePartial" \/ Stop(r) = lengths[r]
                                      \/ Stop(r) = (Split+1)*SplitSize

Init ==
    /\ lengths \in [Rows -> 1..(Capacity*BlockSize)]
    /\ shared \in 0..(Min(lengths[1],lengths[2]) \div BlockSize)
    /\ Cardinality(Resident) <= Capacity
    /\ windowed \in BOOLEAN
    /\ tile = 0 /\ loaded = {}
    /\ visits = [r \in Rows |-> <<>>] /\ partials = [r \in Rows |-> <<>>]

\* Stage a physical tile once, serving every row that references it. The
\* group keeps its split index and row-local reduction across tile advances.
Stage(s) ==
    /\ s \in Requests \ loaded
    /\ visits' = [r \in Rows |-> IF Uses(r,s) THEN visits[r] \o Keys(r,s) ELSE visits[r]]
    /\ partials' = [r \in Rows |-> IF Uses(r,s) /\ EndPartial(r)
                           THEN Append(partials[r], <<Split,Stop(r)-1>>) ELSE partials[r]]
    /\ loaded' = loaded \cup {s}
    /\ UNCHANGED <<lengths,shared,windowed,tile>>
Advance ==
    /\ loaded = Requests
    /\ tile' = tile+1 /\ loaded' = {}
    /\ UNCHANGED <<lengths,shared,windowed,visits,partials>>
Next == \/ tile < TileCount /\ ((\E s \in Requests : Stage(s)) \/ Advance)
        \/ tile = TileCount /\ UNCHANGED vars
Spec == Init /\ [][Next]_vars

VisitedKeys == tile = TileCount => \A r \in Rows : visits[r] = KeyRef(r)
OrderedPartials == tile = TileCount => \A r \in Rows : partials[r] = PartRef(r)
=============================================================================
