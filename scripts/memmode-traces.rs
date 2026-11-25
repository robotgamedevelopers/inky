//! Compiled with exact production Mode and daemon startup restore source.
mod production { const LIMIT: &str = "fake"; include!("mode.rs"); include!("startup.rs"); }
use production::Mode;
use std::cell::{Cell, RefCell};
#[derive(Clone)]
struct Call { kind: &'static str, arg: i64, result: i64, effect: &'static str }
struct Fake {
    limit: Cell<u64>, disk: Cell<Option<u64>>, rng: Cell<u64>, failures: Cell<bool>,
    coverage: Cell<[u64; 6]>, unconfirmed: Cell<u64>, calls: RefCell<Vec<Call>>, path: String, confirmed_set: Cell<bool>,
}
impl Fake {
    fn draw(&self, n: u64) -> u64 { let mut x=self.rng.get(); x^=x<<13; x^=x>>7; x^=x<<17; self.rng.set(x); x%n }
    fn hit(&self,n:usize) { let mut c=self.coverage.get(); c[n]+=1; self.coverage.set(c); }
    fn log(&self,kind:&'static str,arg:i64,result:i64,effect:&'static str) { self.calls.borrow_mut().push(Call{kind,arg,result,effect}); }
    fn get(&self)->Option<u64> {
        let v=if self.failures.get() && self.draw(5)==0 {self.hit(0);None} else {Some(self.limit.get())};
        self.log("get",-1,opt(v),"read"); v
    }
    fn set(&self,v:u64)->Result<(),String> {
        // Connect and startup receive the production set_limit contract: Ok confirms the write.
        // Close additionally checks read-back, so exercise Ok-without-moving there.
        let outcome=if self.failures.get(){self.draw(if self.confirmed_set.get(){3}else{4})}else{0};
        if outcome==3 && self.limit.get()!=v {self.unconfirmed.set(self.unconfirmed.get()+1);}
        if outcome==0 || outcome==2 {self.limit.set(v);}
        if outcome>0 {self.hit(if outcome==3 {5}else{outcome as usize});}
        let ok=outcome==0 || outcome==3;
        self.log("set",v as i64,if ok{0}else{1},if outcome==0 || outcome==2 {"assigned"} else {"unchanged"});
        if ok{Ok(())}else{Err("injected set failure".into())}
    }
    fn keep(&self,v:Option<u64>)->Result<(),String> {
        let fail=self.failures.get() && self.draw(3)==0;
        self.log("keep",opt(v),if fail{1}else{0},if fail{"unchanged"}else{"assigned"});
        if fail {self.hit(if v.is_some(){3}else{4});return Err("injected keep failure".into());}
        self.disk.set(v);
        if let Some(v)=v {std::fs::write(&self.path,v.to_string()).unwrap();}
        else {let _=std::fs::remove_file(&self.path);}
        Ok(())
    }
    fn snapshot(&self,m:&Mode,owner:u64,alive:bool,owed:bool,op:&str)->String {
        let conns=(1..=m.clients).map(|v|v.to_string()).collect::<Vec<_>>().join(",");
        let calls=self.calls.borrow().iter().map(|c|format!("[kind |-> \"{}\", arg |-> {}, result |-> {}, effect |-> \"{}\"]",c.kind,c.arg,c.result,c.effect)).collect::<Vec<_>>().join(",");
        format!("[limit |-> {}, saved |-> {}, disk |-> {}, conns |-> {{{conns}}}, alive |-> {}, owed |-> {}, owner |-> {owner}, op |-> \"{op}\", calls |-> <<{calls}>>]",self.limit.get(),opt(m.saved),opt(self.disk.get()),boolean(alive),boolean(owed))
    }
}
fn opt(v:Option<u64>)->i64 {v.map_or(-1,|x|x as i64)}
fn boolean(v:bool)-> &'static str {if v{"TRUE"}else{"FALSE"}}
fn main() {
    let is_positive = std::env::args().nth(2).as_deref() == Some("positive");
    println!("---- MODULE TraceData ----\nEXTENDS Integers\nTraces == <<");
    let mut total=[0;6]; let mut restarts=0; let mut failed_restarts=0; let mut unconfirmed=0; let mut kept_higher=0;
    let mut all_traces = Vec::new();
    let max_seed = if is_positive { 32 } else { 8 };
    for seed in 1..=max_seed {
        let f=Fake{limit:Cell::new(50000),disk:Cell::new(None),rng:Cell::new(seed),failures:Cell::new(false),coverage:Cell::new([0;6]),unconfirmed:Cell::new(0),calls:RefCell::new(vec![]),path:format!("{}.{}.saved",std::env::args().nth(1).expect("temporary prefix"),seed),confirmed_set:Cell::new(false)};
        let mut m=Mode::default(); let mut owner=50000; let mut alive=true; let mut owed=false;
        let mut states=vec![f.snapshot(&m,owner,alive,owed,"init")];
        for step in 0..192 {
            f.calls.borrow_mut().clear(); f.failures.set(step>=4); f.confirmed_set.set(true);
            let action=if step<2{0}else if step<4{1}else{f.draw(4)};
            let op;
            if !alive {
                op="restart"; f.confirmed_set.set(true);
                if f.disk.get().is_some(){restarts+=1;}
                m=production::restore_on_startup(&f.path,|v|f.set(v),|v|f.keep(v)).expect("valid simulated saved limit");
                f.confirmed_set.set(false); alive=true;
                // An obligation comes from the persisted record and observed restore result,
                // independently of whether production remembered to seed Mode.saved.
                owed=f.calls.borrow().first().is_some_and(|c|c.result!=0);
                if owed {failed_restarts+=1;}
            } else {match action {
                0 if m.clients<3=>{
                    let was_idle_higher=m.clients==0 && m.saved.is_none() && owner>118000;
                    op="connect";
                    if m.connect(118000,||f.get(),|v|f.set(v),|v|f.keep(v)).is_ok() && was_idle_higher {kept_higher+=1;}
                }
                1 if m.clients>0=>{op="close";f.confirmed_set.set(false);m.close(||f.get(),|v|f.set(v),|v|f.keep(v));}
                2 if m.clients==0 && !owed && f.disk.get().is_none()=>{op="owner";owner=[0,50000,118000,130000][f.draw(4)as usize];f.limit.set(owner);}
                3=>{op="crash";m=Mode::default();alive=false;owed=false;}
                _=>{op="idle";}
            }}
            if op=="connect" || op=="close" {
                let calls=f.calls.borrow();
                for (i,c) in calls.iter().enumerate() {
                    if c.kind=="keep" && c.arg!=-1 && c.result==0 {owed=true;}
                    if c.kind=="set" && c.arg==118000 && c.result==0 && op=="connect" {owed=true;}
                    // Only an actual read-back followed by forgetting confirms release.
                    if c.kind=="keep" && c.arg==-1 && i>0 && calls[i-1].kind=="get" && calls[i-1].result==owner as i64 {owed=false;}
                }
            }
            states.push(f.snapshot(&m,owner,alive,owed,op));
        }
        unconfirmed+=f.unconfirmed.get();
        for(sum,count)in total.iter_mut().zip(f.coverage.get()){*sum+=count;}
        let _=std::fs::remove_file(&f.path);
        all_traces.push(format!("<<{}>>", states.join(",\n")));
        if !is_positive && total.iter().all(|&n| n > 0) && restarts > 0 && failed_restarts > 0 && unconfirmed > 0 && kept_higher > 0 {
            break;
        }
    }
    println!("{}\n>>\n====", all_traces.join(",\n"));
    if is_positive {
        assert!(total.iter().all(|n|*n>0),"missing failure class: {total:?}"); assert!(restarts>0 && failed_restarts>0 && unconfirmed>0 && kept_higher>0,"missing discriminating restore or keep-higher scenarios");
    }
    eprintln!("failure coverage [get, set unchanged, set moved, keep write, keep forget, Ok unmoved]: {total:?}; saved-file restarts: {restarts} ({failed_restarts} failed); Ok restores left at wrong value: {unconfirmed}; keep-higher connects: {kept_higher}");
}
