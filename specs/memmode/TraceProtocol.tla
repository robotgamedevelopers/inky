--------------------------- MODULE TraceProtocol ---------------------------
EXTENDS Integers, Sequences, FiniteSets
\* The call log is produced by the injected closures, independently of Mode.
\* Result 0/1 means success/failure for set/keep; get returns the value or -1.
\* Effect records whether the fake assigned the argument or left storage unchanged.
ValueAfter(cs, n, kind, initial) ==
    LET writes == {i \in 1..n : cs[i].kind = kind /\ cs[i].effect = "assigned"}
    IN IF writes = {} THEN initial
       ELSE cs[CHOOSE i \in writes : \A j \in writes : j <= i].arg
Effects(p,s,cs) ==
    /\ s.op = "owner" \/ s.limit = ValueAfter(cs,Len(cs),"set",p.limit)
    /\ s.disk = ValueAfter(cs,Len(cs),"keep",p.disk)
    /\ \A i \in 1..Len(cs) : cs[i].kind = "get" =>
          cs[i].result \in {-1, ValueAfter(cs,i-1,"set",p.limit)}
At(cs, i) == IF i <= Len(cs) THEN cs[i] ELSE [kind |-> "missing", arg |-> -2, result |-> -2]
Call(cs, i, kind, arg) == /\ i <= Len(cs) /\ At(cs,i).kind = kind /\ At(cs,i).arg = arg
End(s, cs, n, saved, owed, clients) == /\ Len(cs) = n /\ s.saved = saved /\ s.owed = owed /\ Cardinality(s.conns) = clients
Forget(s, cs, n, clients) == /\ Call(cs,n,"keep",-1) /\ End(s,cs,n,-1,FALSE,clients)
Raise(p, s, cs, n, found, target) ==
    /\ Call(cs,n,"set",target)
    /\ IF At(cs,n).result = 0 THEN End(s,cs,n,found,TRUE,1)
       ELSE /\ Call(cs,n+1,"get",-1)
            /\ LET rollback == At(cs,n+1).result # found
                   lastGet == IF rollback THEN n+3 ELSE n+2
               IN /\ IF rollback THEN Call(cs,n+2,"set",found) ELSE TRUE
                  /\ Call(cs,lastGet,"get",-1)
                  /\ IF At(cs,lastGet).result = found
                        THEN Forget(s,cs,lastGet+1,0)
                        ELSE End(s,cs,lastGet,found,TRUE,0)
ConnectCalls(p,s,cs,target) ==
    IF p.conns # {} THEN End(s,cs,0,p.saved,p.owed,Cardinality(p.conns)+1)
    ELSE IF p.saved # -1 THEN Raise(p,s,cs,1,p.saved,target)
    ELSE /\ Call(cs,1,"get",-1)
         /\ IF At(cs,1).result = -1 THEN End(s,cs,1,p.saved,p.owed,0)
            ELSE IF At(cs,1).result >= target THEN End(s,cs,1,-1,FALSE,1)
            ELSE /\ Call(cs,2,"keep",At(cs,1).result)
                 /\ IF At(cs,2).result # 0 THEN End(s,cs,2,p.saved,p.owed,0)
                    ELSE Raise(p,s,cs,3,At(cs,1).result,target)
CloseCalls(p,s,cs) ==
    IF Cardinality(p.conns) > 1 \/ p.saved = -1
       THEN End(s,cs,0,p.saved,p.owed,Cardinality(p.conns)-1)
    ELSE /\ Call(cs,1,"set",p.saved)
         /\ IF At(cs,1).result # 0 THEN End(s,cs,1,p.saved,TRUE,0)
            ELSE /\ Call(cs,2,"get",-1)
                 /\ IF At(cs,2).result = p.saved THEN Forget(s,cs,3,0)
                    ELSE End(s,cs,2,p.saved,TRUE,0)
RestartCalls(p,s,cs) ==
    IF p.disk = -1 THEN End(s,cs,0,-1,FALSE,0)
    ELSE /\ Call(cs,1,"set",p.disk)
         /\ IF At(cs,1).result = 0 THEN Forget(s,cs,2,0)
            ELSE End(s,cs,1,p.disk,TRUE,0)
Protocol(p,s,target) ==
    /\ Effects(p,s,s.calls)
    /\ CASE s.op = "connect" -> ConnectCalls(p,s,s.calls,target)
      [] s.op = "close" -> CloseCalls(p,s,s.calls)
      [] s.op = "restart" -> RestartCalls(p,s,s.calls)
      [] s.op = "crash" -> End(s,s.calls,0,-1,FALSE,0)
      [] OTHER -> End(s,s.calls,0,p.saved,p.owed,Cardinality(p.conns))
=============================================================================
