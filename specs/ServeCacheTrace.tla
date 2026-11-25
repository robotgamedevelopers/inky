---- MODULE ServeCacheTrace ----
EXTENDS Integers, Sequences, FiniteSets, ServeCacheTraceData

\* Projection of the serving path for the StartTails and PrefixTree contracts.
\* The real cache seals a span at once, co-owns slots with lanes and start
\* points, and can retain a node whose old window handle has expired.
CONSTANTS B, W, Every, PromptReason, EveryReason
VARIABLE at

Init == at = 1
Next == /\ at < Len(Trace) /\ at' = at + 1

S == Trace[at]
Prev == Trace[IF at > 1 THEN at - 1 ELSE 1]
Lanes == 1..2
K == 1..2
Ids(r) == {n[1] : n \in r.tree}
Node(r, id) == CHOOSE n \in r.tree : n[1] = id
StartIds(r) == {s[1] : s \in r.starts}
Start(r, id) == CHOOSE s \in r.starts : s[1] = id
SeqSet(q) == {q[i] : i \in DOMAIN q}
Prefix(a,b) == Len(a) <= Len(b) /\ SubSeq(b, 1, Len(a)) = a
RECURSIVE Path(_, _)
Path(r, id) ==
    IF id = 0 THEN <<>> ELSE
    LET n == Node(r, id) IN
        IF n[2] \notin Ids(r) \cup {0} \/ n[2] >= id THEN <<0>> ELSE
        Append(Path(r, n[2]), n[3])

FirstWindow(pos) == IF pos < W THEN 0 ELSE (pos - W + 1) \div B
LastBlock(pos) == (pos - 1) \div B
Keys(pos) ==
    {<<1, i>> : i \in 0..LastBlock(pos)}
    \cup {<<2, i>> : i \in FirstWindow(pos)..LastBlock(pos)}
EntryKeys(entries) == {<<e[1], e[2]>> : e \in entries}
Cell(r, k, h) == r.pools[k][h[1]]
Current(r, k, h) ==
    IF h[1] \in DOMAIN r.pools[k]
    THEN Cell(r,k,h)[1] = h[2] /\ Cell(r,k,h)[3] > 0
    ELSE FALSE

Tables ==
    \A l \in Lanes :
        LET lane == S.lanes[l] IN
        /\ EntryKeys(lane[5]) = Keys(lane[1])
        /\ Cardinality(lane[5]) = Cardinality(Keys(lane[1]))
        /\ \A e \in lane[5] :
             /\ Current(S, e[1], e[3])
             /\ Cell(S, e[1], e[3])[2] =
                   IF e[2] = LastBlock(lane[1]) THEN ((lane[1] - 1) % B) + 1 ELSE B

RetainedWindows ==
    \A s \in S.starts :
        /\ s[1] \in Ids(S)
        /\ s[2] = B * Len(Path(S, s[1]))
        /\ s[4] = s[2]
        /\ EntryKeys(s[5]) = Keys(s[2])
        /\ Cardinality(s[5]) = Cardinality(Keys(s[2]))
        /\ \A e \in s[5] :
             /\ Current(S, e[1], e[3])
             /\ Cell(S, e[1], e[3])[2] = B
             /\ Cell(S, e[1], e[3])[4] = 1

Owned(entries, k, slot) == {e \in entries : e[1] = k /\ e[3][1] = slot}

PoolRefs ==
    \A k \in K : \A slot \in DOMAIN S.pools[k] :
        S.pools[k][slot][3] =
            Cardinality(UNION {{<<l, e>> : e \in Owned(S.lanes[l][5], k, slot)} : l \in Lanes})
            + Cardinality(UNION {{<<s[1], e>> : e \in Owned(s[5], k, slot)} : s \in S.starts})

TreePinsAndPruning ==
    /\ \A l \in Lanes :
           LET lane == S.lanes[l] IN
           \A i \in DOMAIN lane[4] :
               /\ lane[4][i] \in Ids(S)
               /\ Path(S, lane[4][i]) = SubSeq(lane[3], 1, i)
    /\ \A n \in S.tree :
        /\ n[2] \in Ids(S) \cup {0}
        /\ n[2] < n[1]
        /\ \A m \in S.tree : <<m[2], m[3]>> = <<n[2], n[3]>> => m = n
        /\ n[4] = Cardinality({l \in Lanes : n[1] \in SeqSet(S.lanes[l][4])})
        /\ n[5] = IF n[1] \in StartIds(S) THEN 1 ELSE 0
        /\ n[4] > 0 \/ \E s \in S.starts : Prefix(Path(S, n[1]), Path(S, s[1]))

\* Real serving requires a seed token after a match. Equal leading tokens in
\* a block cannot substitute for exact tokens with the correct parent.
Lookup ==
    IF S.op # "bind" THEN TRUE ELSE
    LET lane == S.lanes[S.lane]
        matches == {s \in Prev.starts :
            s[2] < lane[2] /\ Prefix(Path(Prev, s[1]), lane[3])}
        best == CHOOSE s \in matches : \A other \in matches : other[2] <= s[2]
    IN
    IF matches = {} THEN /\ S.arg = 0 /\ lane[1] = 0 /\ lane[4] = <<>> /\ S.cs = 0
    ELSE
        /\ S.arg = best[2] /\ lane[1] = best[2] /\ S.cs = best[4] /\ lane[5] = best[5]
        /\ Len(lane[4]) = best[2] \div B
        /\ lane[4][Len(lane[4])] = best[1]

\* New seals must install the lane's current handles. A live full handle
\* for the same node must be shared. An old window handle may be stale.
HandleRefresh ==
    IF S.op # "prefill" THEN TRUE ELSE
    LET lane == S.lanes[S.lane]
        before == Prev.lanes[S.lane]
    IN
    \A i \in (Len(before[4])+1)..Len(lane[4]) :
        LET id == lane[4][i]
            node == Node(S, id)
        IN
        /\ \A e \in lane[5] :
            e[2] = i - 1 => node[5 + e[1]] = e[3]
        /\ IF id \in Ids(Prev) /\ Current(Prev, 1, Node(Prev,id)[6])
           THEN node[6] = Node(Prev,id)[6] ELSE TRUE

\* Start points are published only at complete prompt or periodic boundaries.
\* Releasing a lane never discards a checkpoint. Evicting one drops only it.
StartPoints ==
    LET lane == S.lanes[S.lane]
        pos == lane[1]
        tags == lane[4]
        tip == IF Len(tags) = 0 THEN 0 ELSE tags[Len(tags)]
        prompt == pos = lane[2] \/ pos = ((lane[2] - 1) \div B) * B
        periodic == pos % Every = 0
        reasons == (IF prompt THEN PromptReason ELSE 0)
                 + (IF periodic THEN EveryReason ELSE 0)
    IN
    CASE S.op = "init" -> S.starts = {}
      [] S.op = "evict" ->
           /\ S.arg \in StartIds(Prev)
           /\ S.starts = {s \in Prev.starts : s[1] # S.arg}
      [] S.op = "prefill" /\ pos > 0 /\ pos % B = 0 /\ reasons > 0 ->
           /\ tip \in StartIds(S)
           /\ Start(S, tip)[3] = reasons
           /\ Start(S, tip)[5] = lane[5]
           /\ {s \in S.starts : s[1] # tip} = {s \in Prev.starts : s[1] # tip}
      [] OTHER -> S.starts = Prev.starts

LaneProgress ==
    IF S.op = "init" THEN \A l \in Lanes : S.lanes[l] = <<0,0,<<>>,<<>>,{}>>
    ELSE
        LET lane == S.lanes[S.lane]
            old == Prev.lanes[S.lane]
        IN
        /\ \A l \in Lanes \ {S.lane} : S.lanes[l] = Prev.lanes[l]
        /\ CASE S.op = "release" -> lane = <<0,0,<<>>,<<>>,{}>>
              [] S.op = "evict" -> lane = old
              [] S.op = "decode" ->
                   /\ lane[1] = old[1] + 1 /\ lane[1] = S.arg
                   /\ SubSeq(lane, 2, 4) = SubSeq(old, 2, 4)
              [] S.op = "prefill" ->
                   /\ lane[1] > old[1] /\ lane[1] = S.arg /\ lane[1] <= lane[2]
                   /\ Len(lane[4]) = lane[1] \div B
                   /\ Prefix(old[4], lane[4])
                   /\ SubSeq(lane, 2, 3) = SubSeq(old, 2, 3)
              [] S.op = "bind" -> old = <<0,0,<<>>,<<>>,{}>>
====
