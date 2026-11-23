//! Compiled beside an exact extraction of production Mode by check-memmode-traces.sh.
mod production { const LIMIT: &str = "fake"; include!("mode.rs"); }
use production::Mode;
use std::cell::Cell;

struct Fake { limit: Cell<u64>, disk: Cell<Option<u64>>, rng: Cell<u64>, failures: Cell<bool>, coverage: Cell<[u64; 5]> }
impl Fake {
    fn draw(&self, n: u64) -> u64 {
        let mut x = self.rng.get(); x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.rng.set(x); x % n
    }
    fn hit(&self, n: usize) { let mut c = self.coverage.get(); c[n] += 1; self.coverage.set(c); }
    fn get(&self) -> Option<u64> {
        if self.failures.get() && self.draw(5) == 0 { self.hit(0); None } else { Some(self.limit.get()) }
    }
    fn set(&self, v: u64) -> Result<(), String> {
        let outcome = if self.failures.get() { self.draw(3) } else { 0 };
        if outcome != 1 { self.limit.set(v); }
        if outcome == 0 { Ok(()) } else { self.hit(outcome as usize); Err(format!("injected set outcome {outcome}")) }
    }
    fn keep(&self, v: Option<u64>) -> Result<(), String> {
        if self.failures.get() && self.draw(3) == 0 { self.hit(if v.is_some() { 3 } else { 4 }); Err("injected keep failure".into()) }
        else { self.disk.set(v); Ok(()) }
    }
    fn snapshot(&self, m: &Mode, owner: u64) -> String {
        let opt = |v: Option<u64>| v.map_or(-1, |x| x as i64);
        let conns = (1..=m.clients).map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        format!("[limit |-> {}, saved |-> {}, disk |-> {}, conns |-> {{{conns}}}, alive |-> TRUE, owed |-> {}, owner |-> {owner}]",
            self.limit.get(), opt(m.saved), opt(self.disk.get()), if m.saved.is_some() { "TRUE" } else { "FALSE" })
    }
}
fn main() {
    println!("---- MODULE TraceData ----\nEXTENDS Integers\nTraces == <<");
    let mut total = [0; 5];
    for seed in 1..=32 {
        let f = Fake { limit: Cell::new(50000), disk: Cell::new(None), rng: Cell::new(seed), failures: Cell::new(false), coverage: Cell::new([0; 5]) };
        let mut m = Mode::default(); let mut owner = 50000;
        let mut states = vec![f.snapshot(&m, owner)];
        for step in 0..128 {
            // An unfailing multi-client prefix makes the early-release mutation deterministic.
            f.failures.set(step >= 4);
            let action = if step < 2 { 0 } else if step < 4 { 1 } else { f.draw(3) };
            match action {
                0 if m.clients < 3 => { let _ = m.connect(118000, || f.get(), |v| f.set(v), |v| f.keep(v)); }
                1 if m.clients > 0 => m.close(|| f.get(), |v| f.set(v), |v| f.keep(v)),
                2 if m.clients == 0 && m.saved.is_none() && f.disk.get().is_none() => {
                    owner = [0, 50000, 118000][f.draw(3) as usize]; f.limit.set(owner);
                }
                _ => {}
            }
            states.push(f.snapshot(&m, owner));
        }
        for (sum, count) in total.iter_mut().zip(f.coverage.get()) { *sum += count; }
        println!("<<{}>>{}", states.join(",\n"), if seed == 32 { "" } else { "," });
    }
    println!(">>\n====");
    assert!(total.iter().all(|n| *n > 0), "missing injected failure class: {total:?}");
    eprintln!("failure coverage [get, set unchanged, set moved, keep write, keep forget]: {total:?}");
}
