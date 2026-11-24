//! Layer 3 prefix matching: a request is a token list, matched against a
//! tree of sealed 64-position blocks; it starts at the deepest start point
//! the tree holds for its tokens (docs/DESIGN.md, Blocks; docs/SHARED-PREFIX.md,
//! "Find and retain a block").
//!
//! A block enters the tree only sealed and complete (64 positions), as a
//! child of the block before it, and is never changed afterwards. Matching
//! validates every token of every block it walks: a digest narrows the
//! sibling search, the exact comparison decides, so a digest collision
//! takes the miss path and can never return a block from another history.
//! Equal local tokens under different parents are different blocks.
//!
//! A start point is a tree position kept by the caller (a prompt end, every
//! 16th block boundary a prefill crosses, a lane head): the tree counts how
//! many holders keep each one, and a match returns the deepest start point
//! with a holder on the matched path, with the block handles from the root
//! to it. Position 0 is always a start point and needs nothing.
//!
//! A `BlockEntry` names a pool slot at a generation, and a slot is reused:
//! the entry alone goes stale. `LiveTree` is the matcher over Johnny's
//! slot records — it takes a ref on every block it inserts, compares the
//! generation on every match (a mismatch is a miss: the stale block is
//! evicted and the seek repeated), and drops the refs of everything it
//! removes. The plain `PrefixTree` below it knows nothing of pools.
//!
//! The convolution state and the window blocks behind the handles are the
//! pool's business, not this file's: the matcher is behind the
//! `PrefixMatch` trait so the scheduler never names the concrete tree.
//! Nothing here hashes for the pool; it only matches tokens the caller
//! supplies. The sealed flag on a pool's `SlotRecord` and the sealed flag
//! on a tree block are the same fact seen from two sides; the caller
//! seals in both places, this file cannot seal for the pool.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::blocks::{BlockEntry, BlockError, BlockKind, Slot};

/// Positions in one sealed block: Johnny's constant, re-exported so this
/// file and `crate::blocks` can never disagree about a block's size.
pub use crate::blocks::BLOCK;

/// One sealed block in one tree. The id carries its tree's identity: an
/// id from another tree names nothing here, however its index looks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SealedId {
    tree: u64,
    index: u32,
}

impl SealedId {
    /// This block's index inside its own tree (diagnostics, tests).
    pub fn index(&self) -> u32 { self.index }
}

/// Why a block was not added. Nothing is added on any of these.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum InsertError {
    /// A sealed block is exactly `BLOCK` positions, never a partial tail.
    NotFullBlock(usize),
    /// `pos` must be the end of the parent chain (0 for a root block).
    WrongPosition { expected: usize, got: usize },
    /// The parent id names no block in this tree.
    UnknownParent,
    /// The parent id belongs to a different tree.
    ForeignId,
}

/// The deepest start point a request matches.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StartMatch<B> {
    /// Positions the start point covers: a multiple of `BLOCK`, or 0.
    pub at: usize,
    /// The tree block that ends at `at`; `None` only for position 0.
    pub start: Option<SealedId>,
    /// The tree blocks from the root to `at`, in order.
    pub ids: Vec<SealedId>,
    /// The caller's handles for those blocks, in the same order.
    pub blocks: Vec<B>,
}

/// The tree whose handles are Johnny's table entries: what a caller gets
/// back from a match and installs, kind by kind. The matcher never
/// constructs an entry; it carries the caller's.
pub type EntryTree = PrefixTree<BlockEntry>;

/// What the scheduler asks of a prefix matcher. The concrete tree in this
/// file is one implementation; the trait is the seam Rowan asked for while
/// the block interface (PR 49) was revised.
pub trait PrefixMatch<B> {
    /// The deepest retained start point whose blocks are an exact prefix
    /// of `tokens`. Never past the last completely matched block, and
    /// never a position that is not a retained start point.
    fn deepest_start(&self, tokens: &[u32]) -> StartMatch<B>;
}

struct Node<B> {
    parent: Option<SealedId>,
    pos: usize,
    digest: u64,
    tokens: Box<[u32]>,
    handle: B,
    starts: u32,
    children: Vec<SealedId>,
}

/// FNV-1a over the parent, the position and the ordered token ids. A
/// lookup aid only: the exact token comparison always decides the match.
fn digest(parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |x: u64| h = (h ^ x).wrapping_mul(0x100000001b3);
    mix(parent.map(|p| p.index as u64 + 1).unwrap_or(0));
    mix(pos as u64);
    for &t in tokens { mix(t as u64); }
    h
}

static NEXT_TREE: AtomicU64 = AtomicU64::new(1);

/// The prefix tree of sealed blocks.
pub struct PrefixTree<B> {
    tree: u64,
    nodes: Vec<Option<Node<B>>>,
    roots: Vec<SealedId>,
    by_digest: HashMap<u64, Vec<SealedId>>,
}

impl<B: Copy> PrefixTree<B> {
    pub fn new() -> PrefixTree<B> {
        PrefixTree { tree: NEXT_TREE.fetch_add(1, Ordering::Relaxed), nodes: Vec::new(), roots: Vec::new(), by_digest: HashMap::new() }
    }

    fn node(&self, id: SealedId) -> Option<&Node<B>> {
        if id.tree != self.tree { return None; }
        self.nodes.get(id.index as usize).and_then(|n| n.as_ref())
    }

    fn node_mut(&mut self, id: SealedId) -> Option<&mut Node<B>> {
        if id.tree != self.tree { return None; }
        self.nodes.get_mut(id.index as usize).and_then(|n| n.as_mut())
    }

    /// Add one sealed block. `parent` is the block before it (`None` for
    /// the first block of a prefix) and `pos` its absolute start position.
    /// Adding the same block again (a racing publisher) returns the
    /// canonical block already in the tree, with its original handle:
    /// deduplication happens only after the exact token comparison.
    pub fn insert(&mut self, parent: Option<SealedId>, pos: usize, tokens: &[u32], handle: B) -> Result<SealedId, InsertError> {
        if tokens.len() != BLOCK { return Err(InsertError::NotFullBlock(tokens.len())); }
        let expected = match parent {
            None => 0,
            Some(p) => {
                if p.tree != self.tree { return Err(InsertError::ForeignId); }
                self.node(p).ok_or(InsertError::UnknownParent)?.pos + BLOCK
            }
        };
        if pos != expected { return Err(InsertError::WrongPosition { expected, got: pos }); }
        let d = digest(parent, pos, tokens);
        if let Some(bucket) = self.by_digest.get(&d) {
            for &id in bucket {
                let n = self.node(id).expect("digest buckets hold live blocks");
                if n.parent == parent && n.pos == pos && n.tokens.as_ref() == tokens { return Ok(id); }
            }
        }
        let id = SealedId { tree: self.tree, index: self.nodes.len() as u32 };
        self.nodes.push(Some(Node { parent, pos, digest: d, tokens: tokens.into(), handle, starts: 0, children: Vec::new() }));
        match parent {
            None => self.roots.push(id),
            Some(p) => self.node_mut(p).expect("the parent was just checked").children.push(id),
        }
        self.by_digest.entry(d).or_default().push(id);
        Ok(id)
    }

    /// The block already in the tree for exactly this parent, position
    /// and tokens, if there is one.
    fn find(&self, parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> Option<SealedId> {
        self.child(parent, pos, tokens)
    }

    /// Keep the end of this block as a start point for one more holder.
    /// Start points are counted: two holders are normal (a lane head and
    /// a prompt end), and the point survives until the last release.
    /// Returns false when the id names no block in this tree.
    pub fn retain_start(&mut self, id: SealedId) -> bool {
        match self.node_mut(id) {
            Some(n) => { n.starts = n.starts.saturating_add(1); true }
            None => false,
        }
    }

    /// Drop one holder of a start point. Returns false when the id names
    /// no block here or the point had no holder left.
    pub fn release_start(&mut self, id: SealedId) -> bool {
        match self.node_mut(id) {
            Some(n) if n.starts > 0 => { n.starts -= 1; true }
            _ => false,
        }
    }

    /// How many holders keep this block's end as a start point.
    pub fn start_holders(&self, id: SealedId) -> u32 {
        self.node(id).map(|n| n.starts).unwrap_or(0)
    }

    /// Remove this block and everything below it; the tree shrinks, and
    /// the removed handles come back root-first so the caller can drop
    /// whatever the blocks held. Ids are never reused. Returns an empty
    /// vec when the id names no block in this tree.
    pub fn remove(&mut self, id: SealedId) -> Vec<B> {
        let Some(node) = self.node(id) else { return Vec::new() };
        let parent = node.parent;
        // collect the subtree before touching anything
        let mut gone = vec![id];
        let mut i = 0;
        while i < gone.len() {
            let kids = self.node(gone[i]).map(|n| n.children.clone()).unwrap_or_default();
            gone.extend(kids);
            i += 1;
        }
        match parent {
            None => self.roots.retain(|&r| r != id),
            Some(p) => if let Some(n) = self.node_mut(p) { n.children.retain(|&c| c != id); },
        }
        let mut handles = Vec::new();
        for g in gone {
            let n = self.nodes[g.index as usize].take().expect("collected blocks are live");
            if let Some(bucket) = self.by_digest.get_mut(&n.digest) {
                bucket.retain(|&b| b != g);
                if bucket.is_empty() { self.by_digest.remove(&n.digest); }
            }
            handles.push(n.handle);
        }
        handles
    }

    /// The child of `parent` (or a root) whose tokens are exactly this
    /// block. The digest bucket narrows the search; tokens decide.
    fn child(&self, parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> Option<SealedId> {
        let bucket = self.by_digest.get(&digest(parent, pos, tokens))?;
        for &id in bucket {
            let n = self.node(id)?;
            if n.parent == parent && n.pos == pos && n.tokens.as_ref() == tokens { return Some(id); }
        }
        None
    }
}

impl<B: Copy> Default for PrefixTree<B> {
    fn default() -> Self { Self::new() }
}

impl<B: Copy> PrefixMatch<B> for PrefixTree<B> {
    fn deepest_start(&self, tokens: &[u32]) -> StartMatch<B> {
        let mut best = StartMatch { at: 0, start: None, ids: Vec::new(), blocks: Vec::new() };
        let mut parent = None;
        let mut ids: Vec<SealedId> = Vec::new();
        let mut chain: Vec<B> = Vec::new();
        let mut at = 0usize;
        while tokens.len() - at >= BLOCK {
            let Some(id) = self.child(parent, at, &tokens[at..at + BLOCK]) else { break };
            let n = self.node(id).expect("a child id names a live block");
            ids.push(id);
            chain.push(n.handle);
            at += BLOCK;
            parent = Some(id);
            if n.starts > 0 {
                best = StartMatch { at, start: Some(id), ids: ids.clone(), blocks: chain.clone() };
            }
        }
        best
    }
}

/// The pool's slot records, as the live matcher needs them. Johnny's pool
/// worker owns the records; this seam is only the three operations the
/// tree performs on them, so the tree never names the pool type.
pub trait SlotRecords {
    /// Take one more ref on a slot (the record's own `add_ref`).
    fn add_ref(&mut self, slot: Slot, kind: BlockKind) -> Result<(), BlockError>;
    /// Drop one ref (the record's own `drop_ref`).
    fn drop_ref(&mut self, slot: Slot, kind: BlockKind) -> Result<(), BlockError>;
    /// The generation the slot is at now, or `None` when it is not live.
    fn generation(&self, slot: Slot, kind: BlockKind) -> Option<u32>;
}

/// Why a live insert or removal failed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LiveError {
    /// The tree refused the block; the pool was left as it was.
    Insert(InsertError),
    /// A slot record refused (overflow, a drop at zero, out of range).
    Pool(BlockError),
}

impl From<InsertError> for LiveError {
    fn from(e: InsertError) -> LiveError { LiveError::Insert(e) }
}

impl From<BlockError> for LiveError {
    fn from(e: BlockError) -> LiveError { LiveError::Pool(e) }
}

/// The prefix matcher over live pool slots. Every block it holds has one
/// ref on its slot record; every match is checked against the records'
/// current generations, because a slot is reused and its old entry is
/// stale. A stale block on a matched path is a miss from that block on:
/// it is evicted (with everything below it) and the seek repeated
/// against the shrunk tree.
pub struct LiveTree<R> {
    tree: EntryTree,
    records: R,
}

impl<R: SlotRecords> LiveTree<R> {
    pub fn new(records: R) -> LiveTree<R> {
        LiveTree { tree: PrefixTree::new(), records }
    }

    /// Add one sealed block, taking its slot's ref. A block already in
    /// the tree (a racing publisher) takes no second ref.
    pub fn insert(&mut self, parent: Option<SealedId>, pos: usize, tokens: &[u32], entry: BlockEntry) -> Result<SealedId, LiveError> {
        if tokens.len() == BLOCK {
            if let Some(id) = self.tree.find(parent, pos, tokens) { return Ok(id); }
        }
        self.records.add_ref(entry.slot(), entry.kind())?;
        match self.tree.insert(parent, pos, tokens, entry) {
            Ok(id) => Ok(id),
            Err(e) => {
                let _ = self.records.drop_ref(entry.slot(), entry.kind());
                Err(e.into())
            }
        }
    }

    /// Keep or drop one holder of a start point (see `PrefixTree`).
    pub fn retain_start(&mut self, id: SealedId) -> bool { self.tree.retain_start(id) }
    pub fn release_start(&mut self, id: SealedId) -> bool { self.tree.release_start(id) }

    /// Remove a block and everything below it, dropping their refs.
    pub fn remove(&mut self, id: SealedId) -> Result<Vec<BlockEntry>, LiveError> {
        let handles = self.tree.remove(id);
        let mut first_err = None;
        for h in &handles {
            if let Err(e) = self.records.drop_ref(h.slot(), h.kind()) {
                if first_err.is_none() { first_err = Some(e); }
            }
        }
        match first_err {
            None => Ok(handles),
            Some(e) => Err(e.into()),
        }
    }

    /// The deepest start point whose blocks are all still the slots and
    /// generations the tree sealed them at.
    pub fn deepest_start(&mut self, tokens: &[u32]) -> StartMatch<BlockEntry> {
        loop {
            let m = self.tree.deepest_start(tokens);
            let stale = m.blocks.iter().zip(&m.ids).find(|(e, _)| {
                self.records.generation(e.slot(), e.kind()) != Some(e.generation())
            }).map(|(_, &id)| id);
            match stale {
                None => return m,
                Some(id) => { self.tree.remove(id); }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::SlotRecord;

    fn block(seed: u32) -> Vec<u32> { (0..BLOCK as u32).map(|i| seed * 1000 + i).collect() }

    fn tree3() -> (PrefixTree<u64>, Vec<SealedId>) {
        let mut t = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), 10).unwrap();
        let b = t.insert(Some(a), BLOCK, &block(2), 20).unwrap();
        let c = t.insert(Some(b), 2 * BLOCK, &block(3), 30).unwrap();
        (t, vec![a, b, c])
    }

    #[test]
    fn an_empty_tree_starts_at_zero() {
        let t: PrefixTree<u64> = PrefixTree::new();
        let m = t.deepest_start(&block(1));
        assert_eq!(m, StartMatch { at: 0, start: None, ids: vec![], blocks: vec![] });
    }

    #[test]
    fn the_deepest_retained_start_point_wins() {
        let (mut t, ids) = tree3();
        t.retain_start(ids[0]);
        t.retain_start(ids[2]);
        let toks = [block(1), block(2), block(3), block(9)].concat();
        let m = t.deepest_start(&toks);
        assert_eq!(m.at, 3 * BLOCK);
        assert_eq!(m.start, Some(ids[2]));
        assert_eq!(m.ids, ids);
        assert_eq!(m.blocks, vec![10, 20, 30], "handles from the root to the start point, in order");
    }

    #[test]
    fn start_points_are_counted_not_boolean() {
        // Rowan's failing case: a lane head and a prompt end both hold
        // the same point; one release must not unmatch it.
        let (mut t, ids) = tree3();
        t.retain_start(ids[2]);
        t.retain_start(ids[2]);
        assert_eq!(t.start_holders(ids[2]), 2);
        let toks = [block(1), block(2), block(3)].concat();
        assert!(t.release_start(ids[2]), "the lane head lets go");
        assert_eq!(t.deepest_start(&toks).at, 3 * BLOCK, "the prompt end still holds it");
        assert!(t.release_start(ids[2]));
        assert_eq!(t.deepest_start(&toks).at, 0, "the last release unmatches it");
        assert!(!t.release_start(ids[2]), "no holder left to release");
    }

    #[test]
    fn a_block_that_differs_anywhere_stops_the_match() {
        let (mut t, ids) = tree3();
        t.retain_start(ids[2]);
        let mut second = block(2);
        second[63] ^= 1;
        let toks = [block(1), second, block(3)].concat();
        assert_eq!(t.deepest_start(&toks).at, 0, "the second block differs, so even the third block's start point is unreachable");
        let (mut t2, ids2) = tree3();
        t2.retain_start(ids2[1]);
        assert_eq!(t2.deepest_start(&[block(1), block(7)].concat()).at, 0);
    }

    #[test]
    fn equal_local_tokens_under_different_parents_are_different_blocks() {
        let mut t: PrefixTree<u64> = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), 10).unwrap();
        let x = t.insert(None, 0, &block(5), 50).unwrap();
        let a2 = t.insert(Some(a), BLOCK, &block(2), 20).unwrap();
        let x2 = t.insert(Some(x), BLOCK, &block(2), 60).unwrap();
        assert_ne!(a2, x2);
        t.retain_start(a2);
        t.retain_start(x2);
        assert_eq!(t.deepest_start(&[block(5), block(2)].concat()).blocks, vec![50, 60],
            "the match follows the parent chain, never the local tokens alone");
        assert_eq!(t.deepest_start(&[block(1), block(2)].concat()).blocks, vec![10, 20]);
    }

    #[test]
    fn racing_publishers_dedupe_only_on_exact_tokens() {
        let mut t: PrefixTree<u64> = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), 10).unwrap();
        assert_eq!(t.insert(None, 0, &block(1), 99), Ok(a), "the same block again is the canonical one, with its first handle");
        assert_ne!(t.insert(None, 0, &block(2), 20).unwrap(), a, "different tokens are a different root");
    }

    #[test]
    fn unsealed_misplaced_and_foreign_blocks_never_enter() {
        let mut t: PrefixTree<u64> = PrefixTree::new();
        let mut other: PrefixTree<u64> = PrefixTree::new();
        let foreign = other.insert(None, 0, &block(1), 1).unwrap();
        assert_eq!(t.insert(None, 0, &block(1)[..63], 10), Err(InsertError::NotFullBlock(63)), "a partial tail is unsealed");
        assert_eq!(t.insert(None, BLOCK, &block(1), 10), Err(InsertError::WrongPosition { expected: 0, got: BLOCK }));
        assert_eq!(t.insert(Some(foreign), BLOCK, &block(1), 10), Err(InsertError::ForeignId),
            "an id from another tree names nothing here, even at an existing index");
        assert!(!t.retain_start(foreign));
        assert!(t.remove(foreign).is_empty());
        let a = t.insert(None, 0, &block(1), 10).unwrap();
        assert_eq!(t.insert(Some(a), 0, &block(2), 20), Err(InsertError::WrongPosition { expected: BLOCK, got: 0 }));
        // a removed block is unknown to its own tree, not foreign
        t.remove(a);
        assert_eq!(t.insert(Some(a), BLOCK, &block(2), 20), Err(InsertError::UnknownParent));
    }

    #[test]
    fn remove_shrinks_the_tree_and_returns_handles() {
        let (mut t, ids) = tree3();
        t.retain_start(ids[2]);
        let toks = [block(1), block(2), block(3)].concat();
        assert_eq!(t.remove(ids[1]), vec![20, 30], "the block and everything below it, root-first");
        assert_eq!(t.deepest_start(&toks).at, 0);
        assert!(t.remove(ids[1]).is_empty(), "already gone");
        // the same tokens publish again as new blocks; no stale digest entry answers
        let b = t.insert(Some(ids[0]), BLOCK, &block(2), 21).unwrap();
        assert_ne!(b, ids[1], "ids are never reused");
        t.retain_start(b);
        assert_eq!(t.deepest_start(&toks).blocks, vec![10, 21]);
    }

    #[test]
    fn a_partial_final_block_and_sixteen_block_starts() {
        let mut t: PrefixTree<u64> = PrefixTree::new();
        let mut parent = None;
        let mut ids = Vec::new();
        for i in 0..17u32 {
            let id = t.insert(parent, i as usize * BLOCK, &block(i + 1), i as u64).unwrap();
            ids.push(id);
            parent = Some(id);
        }
        t.retain_start(ids[15]); // a 16th-block-boundary start point, position 1024
        let mut toks: Vec<u32> = (1..=17u32).flat_map(block).collect();
        assert_eq!(t.deepest_start(&toks).at, 16 * BLOCK);
        toks.truncate(16 * BLOCK + 63);
        let m = t.deepest_start(&toks);
        assert_eq!(m.at, 16 * BLOCK, "63 tokens into block 17 is not a block");
        assert_eq!(m.blocks.len(), 16);
    }

    /// Johnny's slot records behind the test's own map: the fake is only
    /// storage; every record in it is his `SlotRecord`.
    #[derive(Default)]
    struct FakeRecords { map: HashMap<(u32, bool), SlotRecord> }

    impl FakeRecords {
        fn key(slot: Slot, kind: BlockKind) -> (u32, bool) { (slot.get(), kind == BlockKind::Full) }
        fn record(&mut self, slot: Slot, kind: BlockKind) -> &mut SlotRecord {
            self.map.entry(Self::key(slot, kind)).or_insert_with(|| SlotRecord::fresh(kind))
        }
        fn refs(&mut self, slot: Slot, kind: BlockKind) -> u32 { self.record(slot, kind).refs() }
        fn reuse(&mut self, slot: Slot, kind: BlockKind) { self.record(slot, kind).bump_generation().unwrap(); }
    }

    impl SlotRecords for FakeRecords {
        fn add_ref(&mut self, slot: Slot, kind: BlockKind) -> Result<(), BlockError> { self.record(slot, kind).add_ref() }
        fn drop_ref(&mut self, slot: Slot, kind: BlockKind) -> Result<(), BlockError> { self.record(slot, kind).drop_ref() }
        fn generation(&self, slot: Slot, kind: BlockKind) -> Option<u32> {
            self.map.get(&Self::key(slot, kind)).map(|r| r.generation())
        }
    }

    fn entry(slot: u32, generation: u32, kind: BlockKind) -> BlockEntry {
        BlockEntry::new(Slot::new(slot).unwrap(), generation, 4096, kind)
    }

    #[test]
    fn a_reused_slots_stale_entry_is_a_miss_never_a_match() {
        // Rowan's failing case: insert A at (slot 3, gen 0); the pool
        // reuses slot 3 at gen 1 for other tokens; A's tokens must miss.
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), entry(3, 0, BlockKind::Full)).unwrap();
        let b = live.insert(Some(a), BLOCK, &block(2), entry(4, 0, BlockKind::Full)).unwrap();
        live.retain_start(b);
        let toks = [block(1), block(2)].concat();
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "insert took a ref");
        assert_eq!(live.deepest_start(&toks).at, 2 * BLOCK);
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // slot 3 is generation 1 now
        let m = live.deepest_start(&toks);
        assert_eq!(m.at, 0, "the gen-0 entry for a gen-1 slot is stale: a miss, not a match");
        assert_eq!(live.deepest_start(&toks).at, 0, "and it was evicted, not rechecked forever");
        // a fresh block at the slot's new generation matches again
        let a2 = live.insert(None, 0, &block(1), entry(3, 1, BlockKind::Full)).unwrap();
        live.retain_start(a2);
        assert_eq!(live.deepest_start(&block(1)).blocks, vec![entry(3, 1, BlockKind::Full)]);
    }

    #[test]
    fn live_inserts_and_removes_balance_their_refs() {
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), entry(3, 0, BlockKind::Full)).unwrap();
        assert_eq!(live.insert(None, 0, &block(1), entry(3, 0, BlockKind::Full)), Ok(a), "a racing publisher dedupes");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "and takes no second ref");
        // a refused insert rolls its ref back
        assert!(live.insert(Some(a), 0, &block(2), entry(4, 0, BlockKind::Full)).is_err());
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Full), 0);
        let gone = live.remove(a).unwrap();
        assert_eq!(gone, vec![entry(3, 0, BlockKind::Full)]);
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "removal dropped the ref");
    }

    #[test]
    fn entry_handles_come_back_as_johnnys_entries() {
        let full = BlockEntry::new(Slot::new(3).unwrap(), 1, 4096, BlockKind::Full);
        let win = BlockEntry::new(Slot::new(5).unwrap(), 2, 8192, BlockKind::Window);
        let mut t: EntryTree = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), full).unwrap();
        let b = t.insert(Some(a), BLOCK, &block(2), win).unwrap();
        t.retain_start(b);
        let m = t.deepest_start(&[block(1), block(2), block(3)].concat());
        assert_eq!(m.at, 2 * BLOCK);
        assert_eq!(m.blocks, vec![full, win]);
        assert_eq!(m.blocks[0].kind(), BlockKind::Full);
        assert_eq!(m.blocks[0].offset(), 4096);
        assert_eq!(m.blocks[0].slot().get(), 3);
        assert_eq!(m.blocks[1].generation(), 2);
    }
}
