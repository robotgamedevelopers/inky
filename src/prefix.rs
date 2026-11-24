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
//! slot records — it takes a ref on every block it inserts, refuses an
//! entry whose generation its slot has moved past, evicts every stale
//! block on a seek before matching, and drops the refs of everything it
//! removes. The plain `PrefixTree` below it knows nothing of pools.
//!
//! Modelled by `specs/PrefixTree.tla`; this code is bound to that model
//! by the trace check in `specs/run_tlc.zsh` (`PrefixTreeTrace`): a
//! seeded walk over `LiveTree` emits its states, and TLC checks each
//! emitted state is a behaviour of the spec.
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

    /// Every live block with its handle, root-first in insertion order
    /// (a parent always precedes its children).
    pub fn entries(&self) -> Vec<(SealedId, B)> {
        self.nodes.iter().enumerate()
            .filter_map(|(i, n)| n.as_ref().map(|n| (SealedId { tree: self.tree, index: i as u32 }, n.handle)))
            .collect()
    }

    /// Keep the end of this block as a start point for one more holder.
    /// Start points are counted: two holders are normal (a lane head and
    /// a prompt end), and the point survives until the last release.
    /// `Ok(false)` when the id names no block in this tree; the count is
    /// checked, like Johnny's slot-record mutators, never saturated.
    pub fn retain_start(&mut self, id: SealedId) -> Result<bool, BlockError> {
        match self.node_mut(id) {
            Some(n) => {
                n.starts = n.starts.checked_add(1).ok_or(BlockError::Overflow)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// The caller's handle on a block in this tree, if it is here.
    pub fn handle(&self, id: SealedId) -> Option<B> {
        self.node(id).map(|n| n.handle)
    }

    /// Drop one holder of a start point. Returns false when the id names
    /// no block here or the point had no holder left; an evicted block
    /// and a drained point are the same `false`, so a caller that must
    /// tell them apart checks `handle` or `start_holders` first.
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
    /// The entry names a generation its slot is no longer at (or a slot
    /// that is not live): the entry is stale on arrival, so nothing was
    /// inserted and no ref was taken.
    StaleEntry { slot: Slot, claimed: u32, current: Option<u32> },
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
/// stale. An insert naming a generation the slot has moved past is
/// refused on arrival, and a seek evicts every stale block (each with
/// everything below it) before matching, as the model's `Seek` does.
pub struct LiveTree<R> {
    tree: EntryTree,
    records: R,
}

impl<R: SlotRecords> LiveTree<R> {
    pub fn new(records: R) -> LiveTree<R> {
        LiveTree { tree: PrefixTree::new(), records }
    }

    /// Add one sealed block, taking its slot's ref. The entry must name
    /// its slot's current generation: a stale entry (or one naming a
    /// slot that is not live) is refused before any ref is taken, even
    /// when its tokens are already in the tree. A block already in the
    /// tree (a racing publisher) takes no second ref — unless the twin
    /// in the tree is stale: its slot has moved on, so the twin is
    /// evicted (its ref dropped) and the fresh block takes its place.
    /// The returned id's handle may be another publisher's entry;
    /// `handle` shows the entry the tree actually holds. A refused
    /// insert rolls its ref back; if the rollback itself refuses, that
    /// pool error is the one returned.
    pub fn insert(&mut self, parent: Option<SealedId>, pos: usize, tokens: &[u32], entry: BlockEntry) -> Result<SealedId, LiveError> {
        match self.records.generation(entry.slot(), entry.kind()) {
            Some(g) if g == entry.generation() => {}
            current => return Err(LiveError::StaleEntry { slot: entry.slot(), claimed: entry.generation(), current }),
        }
        if tokens.len() == BLOCK {
            if let Some(id) = self.tree.find(parent, pos, tokens) {
                let twin = self.tree.handle(id).expect("a found block has a handle");
                if self.records.generation(twin.slot(), twin.kind()) == Some(twin.generation()) {
                    return Ok(id);
                }
                self.remove(id)?; // the stale twin: evict it, then insert fresh below
            }
        }
        self.records.add_ref(entry.slot(), entry.kind())?;
        match self.tree.insert(parent, pos, tokens, entry) {
            Ok(id) => Ok(id),
            Err(e) => match self.records.drop_ref(entry.slot(), entry.kind()) {
                Ok(()) => Err(e.into()),
                Err(pool) => Err(pool.into()),
            },
        }
    }

    /// Keep or drop one holder of a start point (see `PrefixTree`).
    pub fn retain_start(&mut self, id: SealedId) -> Result<bool, BlockError> { self.tree.retain_start(id) }
    pub fn release_start(&mut self, id: SealedId) -> bool { self.tree.release_start(id) }

    /// The entry the tree holds on a block, if it is here. After a
    /// racing publisher's `Ok(id)`, this is how the caller tells whose
    /// entry the tree kept.
    pub fn handle(&self, id: SealedId) -> Option<BlockEntry> { self.tree.handle(id) }

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
    /// generations the tree sealed them at. A seek first evicts every
    /// stale block in the tree — not only those on the matched path, as
    /// the model's `Seek` does — each through `remove`, so refs are
    /// dropped, never leaked; a pool error from an eviction is
    /// returned, not swallowed. The eviction scan is linear in the
    /// tree's blocks, which bounds what a seek may cost.
    pub fn deepest_start(&mut self, tokens: &[u32]) -> Result<StartMatch<BlockEntry>, LiveError> {
        let stale: Vec<SealedId> = self.tree.entries().into_iter()
            .filter(|(_, e)| self.records.generation(e.slot(), e.kind()) != Some(e.generation()))
            .map(|(id, _)| id)
            .collect();
        for id in stale {
            // an ancestor's eviction already took this block: removing
            // it again removes nothing and drops no second ref
            self.remove(id)?;
        }
        Ok(self.tree.deepest_start(tokens))
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
        t.retain_start(ids[0]).unwrap();
        t.retain_start(ids[2]).unwrap();
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
        t.retain_start(ids[2]).unwrap();
        t.retain_start(ids[2]).unwrap();
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
        t.retain_start(ids[2]).unwrap();
        let mut second = block(2);
        second[63] ^= 1;
        let toks = [block(1), second, block(3)].concat();
        assert_eq!(t.deepest_start(&toks).at, 0, "the second block differs, so even the third block's start point is unreachable");
        let (mut t2, ids2) = tree3();
        t2.retain_start(ids2[1]).unwrap();
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
        t.retain_start(a2).unwrap();
        t.retain_start(x2).unwrap();
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
        assert_eq!(t.retain_start(foreign), Ok(false), "a foreign id is no block here");
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
        t.retain_start(ids[2]).unwrap();
        let toks = [block(1), block(2), block(3)].concat();
        assert_eq!(t.remove(ids[1]), vec![20, 30], "the block and everything below it, root-first");
        assert_eq!(t.deepest_start(&toks).at, 0);
        assert!(t.remove(ids[1]).is_empty(), "already gone");
        // the same tokens publish again as new blocks; no stale digest entry answers
        let b = t.insert(Some(ids[0]), BLOCK, &block(2), 21).unwrap();
        assert_ne!(b, ids[1], "ids are never reused");
        t.retain_start(b).unwrap();
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
        t.retain_start(ids[15]).unwrap(); // a 16th-block-boundary start point, position 1024
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
            // a record starts at generation 1 when the slot is first
            // taken: generation 0 is "never taken" (blocks.rs)
            self.map.entry(Self::key(slot, kind)).or_insert_with(|| {
                let mut r = SlotRecord::fresh(kind);
                r.bump_generation().unwrap();
                r
            })
        }
        fn refs(&mut self, slot: Slot, kind: BlockKind) -> u32 { self.record(slot, kind).refs() }
        fn pool_refs(&self, slot: Slot, kind: BlockKind) -> u32 {
            self.map.get(&Self::key(slot, kind)).map(|r| r.refs()).unwrap_or(0)
        }
        fn reuse(&mut self, slot: Slot, kind: BlockKind) { self.record(slot, kind).bump_generation().unwrap(); }
    }

    impl SlotRecords for FakeRecords {
        fn add_ref(&mut self, slot: Slot, kind: BlockKind) -> Result<(), BlockError> { self.record(slot, kind).add_ref() }
        fn drop_ref(&mut self, slot: Slot, kind: BlockKind) -> Result<(), BlockError> { self.record(slot, kind).drop_ref() }
        fn generation(&self, slot: Slot, kind: BlockKind) -> Option<u32> {
            // an untouched slot reads as generation 1, the generation
            // `record` takes it at (fresh 0 bumped once on first use)
            Some(self.map.get(&Self::key(slot, kind)).map(|r| r.generation()).unwrap_or(1))
        }
    }

    fn entry(slot: u32, generation: u32, kind: BlockKind) -> BlockEntry {
        BlockEntry::new(Slot::new(slot).unwrap(), generation, 4096, kind)
    }

    #[test]
    fn a_reused_slots_stale_entry_is_a_miss_never_a_match() {
        // Rowan's failing case: insert A at (slot 3, gen 1); the pool
        // reuses slot 3 at gen 2 for other tokens; A's tokens must miss,
        // and the eviction must drop both blocks' refs, not leak them.
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), entry(3, 1, BlockKind::Full)).unwrap();
        let b = live.insert(Some(a), BLOCK, &block(2), entry(4, 1, BlockKind::Full)).unwrap();
        live.retain_start(b).unwrap();
        let toks = [block(1), block(2)].concat();
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "insert took a ref");
        assert_eq!(live.deepest_start(&toks).unwrap().at, 2 * BLOCK);
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // slot 3 is generation 2 now
        let m = live.deepest_start(&toks).unwrap();
        assert_eq!(m.at, 0, "the gen-1 entry for a gen-2 slot is stale: a miss, not a match");
        assert_eq!(live.deepest_start(&toks).unwrap().at, 0, "and it was evicted, not rechecked forever");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "the evicted root's ref was dropped");
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Full), 0, "the evicted child's ref was dropped too (Rowan's leak)");
        // a fresh block at the slot's new generation matches again
        let a2 = live.insert(None, 0, &block(1), entry(3, 2, BlockKind::Full)).unwrap();
        live.retain_start(a2).unwrap();
        assert_eq!(live.deepest_start(&block(1)).unwrap().blocks, vec![entry(3, 2, BlockKind::Full)]);
    }

    #[test]
    fn a_fresh_block_is_never_lost_to_its_stale_twin() {
        // the stale twin is still in the tree when the fresh block is
        // sealed at the slot's new generation: the dedupe must evict
        // the twin (dropping its ref) and keep the fresh block
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), entry(3, 1, BlockKind::Full)).unwrap();
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // gen 2; the gen-1 twin is stale
        let a2 = live.insert(None, 0, &block(1), entry(3, 2, BlockKind::Full)).unwrap();
        assert_ne!(a2, a, "the fresh block is a new block, not the stale twin");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "twin's ref dropped, fresh ref held");
        live.retain_start(a2).unwrap();
        assert_eq!(live.deepest_start(&block(1)).unwrap().blocks, vec![entry(3, 2, BlockKind::Full)],
            "and the fresh block survives the next seek");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1);
    }

    #[test]
    fn a_stale_entry_is_refused_at_insert() {
        // the cold reader's probe: slot 3 is at generation 2, and a
        // generation-1 entry must not be accepted — for new tokens or
        // for tokens already in the tree — and must take no ref
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), entry(3, 1, BlockKind::Full)).unwrap();
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // slot 3 is generation 2 now
        let stale = LiveError::StaleEntry { slot: Slot::new(3).unwrap(), claimed: 1, current: Some(2) };
        assert_eq!(live.insert(None, 0, &block(9), entry(3, 1, BlockKind::Full)), Err(stale.clone()),
            "a gen-1 entry for a gen-2 slot is refused, not inserted");
        assert_eq!(live.insert(None, 0, &block(1), entry(3, 1, BlockKind::Full)), Err(stale),
            "the same refusal when the tokens are already in the tree");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "no ref was taken by either refusal");
        assert_eq!(live.handle(a), Some(entry(3, 1, BlockKind::Full)), "the tree is unchanged");
        // a fresh entry at the current generation still lands
        assert!(live.insert(None, 0, &block(9), entry(3, 2, BlockKind::Full)).is_ok());
    }

    #[test]
    fn a_seek_evicts_stale_blocks_off_the_matched_path() {
        // the model's Seek evicts every stale block; the code now does
        // the same, so a stale block the request never touches is gone
        // (with its ref) after any seek, not only after a matching one
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), entry(3, 1, BlockKind::Full)).unwrap();
        let off = live.insert(None, 0, &block(7), entry(4, 1, BlockKind::Full)).unwrap();
        live.retain_start(a).unwrap();
        live.retain_start(off).unwrap();
        live.records.reuse(Slot::new(4).unwrap(), BlockKind::Full); // slot 4 at gen 2: `off` is stale
        let m = live.deepest_start(&block(1)).unwrap();
        assert_eq!(m.at, BLOCK, "the live path still matches");
        assert_eq!(live.handle(off), None, "the stale off-path block was evicted by the seek");
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Full), 0, "and its ref was dropped");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "the live block keeps its ref");
    }

    #[test]
    fn live_inserts_and_removes_balance_their_refs() {
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), entry(3, 1, BlockKind::Full)).unwrap();
        assert_eq!(live.insert(None, 0, &block(1), entry(3, 1, BlockKind::Full)), Ok(a), "a racing publisher dedupes");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "and takes no second ref");
        // a refused insert rolls its ref back
        assert!(live.insert(Some(a), 0, &block(2), entry(4, 1, BlockKind::Full)).is_err());
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Full), 0);
        let gone = live.remove(a).unwrap();
        assert_eq!(gone, vec![entry(3, 1, BlockKind::Full)]);
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "removal dropped the ref");
    }

    /// One abstract block of the model (two tokens over {1, 2}) as a
    /// code block: the first token fills the first half, the second
    /// the second half, so abstractly distinct blocks stay distinct.
    fn enc(block: [u32; 2]) -> Vec<u32> {
        let mut v = vec![block[0]; BLOCK / 2];
        v.extend([block[1]; BLOCK / 2]);
        v
    }

    fn tla_seq(s: &[u32]) -> String {
        let inner: Vec<String> = s.iter().map(|t| t.to_string()).collect();
        format!("<<{}>>", inner.join(", "))
    }

    /// The model's block-boundary sequences over {1, 2}, shortest first.
    fn tla_universe() -> Vec<Vec<u32>> {
        let mut u = vec![Vec::new()];
        for a in 1..=2 { for b in 1..=2 { u.push(vec![a, b]); } }
        for a in 1..=2 { for b in 1..=2 { for c in 1..=2 { for d in 1..=2 { u.push(vec![a, b, c, d]); } } } }
        u
    }

    fn tla_fn(values: &HashMap<Vec<u32>, i64>) -> String {
        // a CASE function over the whole domain: explicit-point
        // function literals ([<<>> |-> 0, ...]) do not parse
        let cases: Vec<String> = tla_universe().iter()
            .filter(|s| values.get(*s).copied().unwrap_or(0) != 0)
            .map(|s| format!("s = {} -> {}", tla_seq(s), values[s]))
            .collect();
        if cases.is_empty() { return "[s \\in BlockSeqs |-> 0]".to_string(); }
        format!("[s \\in BlockSeqs |-> CASE {} [] OTHER -> 0]", cases.join(" [] "))
    }

    /// The trace dump behind the `PrefixTreeTrace` TLC check: a seeded
    /// walk over `LiveTree` (slots 1-3, abstract generations one below
    /// the records' so the model starts at 0), every operation's
    /// resulting state emitted as one row of `PrefixTreeTraceData.tla`.
    /// TLC then checks the emitted states are behaviours of
    /// `specs/PrefixTree.tla`. The walk opens with a steered pass over
    /// the model's request — held starts on the path, a stale root, a
    /// stale-twin reinsert — then wanders, seeks included, so a broken
    /// seek writes a row the model cannot step to. Ignored by default;
    /// `specs/run_tlc.zsh` runs it before the trace check.
    #[test]
    #[ignore = "trace dump; run by specs/run_tlc.zsh"]
    fn prefix_trace_dump() {
        let out = std::env::var("INKY_PREFIX_TRACE_OUT")
            .unwrap_or_else(|_| "specs/PrefixTreeTraceData.tla".to_string());
        let mut live = LiveTree::new(FakeRecords::default());
        let mut seq_of: HashMap<SealedId, Vec<u32>> = HashMap::new();
        let mut rows: Vec<String> = Vec::new();
        let mut random: u64 = 0x9e3779b97f4a7c15;
        let mut pick = move |n: usize| -> usize {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            (random >> 32) as usize % n
        };

        let snapshot = |live: &LiveTree<FakeRecords>, seq_of: &HashMap<SealedId, Vec<u32>>,
                        op: &str, s: &[u32], slot: u32, result: &[u32], sought: bool| -> String {
            let mut sealed: Vec<String> = vec![tla_seq(&[])];
            let (mut starts, mut slot_of_m, mut gen_of, mut refs) =
                (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
            for (id, e) in live.tree.entries() {
                let seq = seq_of[&id].clone();
                sealed.push(tla_seq(&seq));
                starts.insert(seq.clone(), live.tree.start_holders(id) as i64);
                slot_of_m.insert(seq.clone(), e.slot().get() as i64);
                gen_of.insert(seq.clone(), e.generation() as i64 - 1);
                refs.insert(seq, 1);
            }
            // the pool's own counts, read from the records: PoolBooks
            // in PrefixTreeTrace binds them to the sealed blocks
            let pool_refs = format!("[s \\in Slots |-> IF s = 1 THEN {} ELSE IF s = 2 THEN {} ELSE {}]",
                live.records.pool_refs(Slot::new(1).unwrap(), BlockKind::Full),
                live.records.pool_refs(Slot::new(2).unwrap(), BlockKind::Full),
                live.records.pool_refs(Slot::new(3).unwrap(), BlockKind::Full));
            let g = |sl: u32| live.records.generation(Slot::new(sl).unwrap(), BlockKind::Full).unwrap() - 1;
            let pool = format!("[s \\in Slots |-> IF s = 1 THEN {} ELSE IF s = 2 THEN {} ELSE {}]", g(1), g(2), g(3));
            format!(
                "[op |-> \"{op}\", s |-> {}, slot |-> {slot}, sealed |-> {{{}}}, starts |-> {}, \
                 slotOf |-> {}, genOf |-> {}, poolGen |-> {}, refs |-> {}, poolRefs |-> {}, result |-> {}, sought |-> {}]",
                tla_seq(s), sealed.join(", "), tla_fn(&starts), tla_fn(&slot_of_m),
                tla_fn(&gen_of), pool, tla_fn(&refs), pool_refs, tla_seq(result),
                if sought { "TRUE" } else { "FALSE" })
        };
        let req_abstract: Vec<u32> = vec![1, 2, 1, 2];
        let req: Vec<u32> = [enc([1, 2]), enc([1, 2])].concat();

        rows.push(snapshot(&live, &seq_of, "init", &[], 0, &[], false));

        // --- the steered pass over the model's request ---
        let root = live.insert(None, 0, &enc([1, 2]), entry(1, 1, BlockKind::Full)).unwrap();
        seq_of.insert(root, vec![1, 2]);
        rows.push(snapshot(&live, &seq_of, "seal", &[1, 2], 1, &[], false));
        let child = live.insert(Some(root), BLOCK, &enc([1, 2]), entry(2, 1, BlockKind::Full)).unwrap();
        seq_of.insert(child, req_abstract.clone());
        rows.push(snapshot(&live, &seq_of, "seal", &req_abstract, 2, &[], false));
        live.retain_start(root).unwrap();
        rows.push(snapshot(&live, &seq_of, "retain", &[1, 2], 0, &[], false));
        live.retain_start(child).unwrap();
        rows.push(snapshot(&live, &seq_of, "retain", &req_abstract, 0, &[], false));
        let m = live.deepest_start(&req).unwrap();
        rows.push(snapshot(&live, &seq_of, "seek", &[], 0, &req_abstract[..m.at / BLOCK * 2], true));
        assert!(live.release_start(child));
        rows.push(snapshot(&live, &seq_of, "release", &req_abstract, 0, &[], false));
        let m = live.deepest_start(&req).unwrap();
        rows.push(snapshot(&live, &seq_of, "seek", &[], 0, &req_abstract[..m.at / BLOCK * 2], true));
        live.records.reuse(Slot::new(1).unwrap(), BlockKind::Full); // the root's slot moves on: the root is stale
        rows.push(snapshot(&live, &seq_of, "reuse", &[], 1, &[], false));
        let m = live.deepest_start(&req).unwrap();
        rows.push(snapshot(&live, &seq_of, "seek", &[], 0, &req_abstract[..m.at / BLOCK * 2], true));
        // the stale twin, reinserted fresh on another slot
        let root2 = live.insert(None, 0, &enc([1, 2]), entry(3, 1, BlockKind::Full)).unwrap();
        seq_of.insert(root2, vec![1, 2]);
        rows.push(snapshot(&live, &seq_of, "seal", &[1, 2], 3, &[], false));
        live.retain_start(root2).unwrap();
        rows.push(snapshot(&live, &seq_of, "retain", &[1, 2], 0, &[], false));
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // root2 is stale now, still sealed
        rows.push(snapshot(&live, &seq_of, "reuse", &[], 3, &[], false));
        let root3 = live.insert(None, 0, &enc([1, 2]), entry(2, 1, BlockKind::Full)).unwrap();
        assert_ne!(root3, root2, "the fresh block replaces the stale twin");
        seq_of.retain(|id, _| *id != root2);
        seq_of.insert(root3, vec![1, 2]);
        rows.push(snapshot(&live, &seq_of, "seal-twin", &[1, 2], 2, &[], false));
        live.retain_start(root3).unwrap();
        rows.push(snapshot(&live, &seq_of, "retain", &[1, 2], 0, &[], false));
        let m = live.deepest_start(&req).unwrap();
        rows.push(snapshot(&live, &seq_of, "seek", &[], 0, &req_abstract[..m.at / BLOCK * 2], true));

        // --- the wander: every operation, seeks included ---
        let mut mutating = 0;
        let mut attempts = 0;
        while mutating < 40 && attempts < 500 {
            attempts += 1;
            let live_ids: Vec<SealedId> = live.tree.entries().into_iter().map(|(id, _)| id).collect();
            seq_of.retain(|id, _| live_ids.contains(id));
            match pick(8) {
                0 => { // seal a fresh block under a live parent (or the root)
                    let parents: Vec<Option<SealedId>> = std::iter::once(None)
                        .chain(live_ids.iter().filter(|id| seq_of[id].len() == 2).map(|id| Some(*id)))
                        .collect();
                    let parent = parents[pick(parents.len())];
                    let base: Vec<u32> = parent.map(|p| seq_of[&p].clone()).unwrap_or_default();
                    let block = [1 + pick(2) as u32, 1 + pick(2) as u32];
                    let mut seq = base.clone();
                    seq.extend(block);
                    if seq_of.values().any(|s| *s == seq) { continue; }
                    let slot = 1 + pick(3) as u32;
                    let gen = live.records.generation(Slot::new(slot).unwrap(), BlockKind::Full).unwrap();
                    let id = live.insert(parent, base.len() / 2 * BLOCK, &enc(block),
                        entry(slot, gen, BlockKind::Full)).expect("a fresh seal at the current generation");
                    seq_of.insert(id, seq.clone());
                    rows.push(snapshot(&live, &seq_of, "seal", &seq, slot, &[], false));
                }
                1 => { // a stale insert, refused: the state must not move
                    let slot = 1 + pick(3) as u32;
                    let gen = live.records.generation(Slot::new(slot).unwrap(), BlockKind::Full).unwrap();
                    if gen < 2 { continue; }
                    let block = [1 + pick(2) as u32, 1 + pick(2) as u32];
                    let got = live.insert(None, 0, &enc(block), entry(slot, gen - 1, BlockKind::Full));
                    assert_eq!(got, Err(LiveError::StaleEntry {
                        slot: Slot::new(slot).unwrap(), claimed: gen - 1, current: Some(gen) }),
                        "the walk's stale insert is refused");
                    rows.push(snapshot(&live, &seq_of, "seal-stale", &block.to_vec(), slot, &[], false));
                }
                2 => { // retain a start point (at most two holders each)
                    let held: Vec<SealedId> = live_ids.iter().copied()
                        .filter(|id| live.tree.start_holders(*id) < 2).collect();
                    if held.is_empty() { continue; }
                    let id = held[pick(held.len())];
                    live.retain_start(id).unwrap();
                    let seq = seq_of[&id].clone();
                    rows.push(snapshot(&live, &seq_of, "retain", &seq, 0, &[], false));
                }
                3 => { // release a held start point
                    let held: Vec<SealedId> = live_ids.iter().copied()
                        .filter(|id| live.tree.start_holders(*id) > 0).collect();
                    if held.is_empty() { continue; }
                    let id = held[pick(held.len())];
                    assert!(live.release_start(id));
                    let seq = seq_of[&id].clone();
                    rows.push(snapshot(&live, &seq_of, "release", &seq, 0, &[], false));
                }
                4 => { // reuse a slot (abstract generation at most 2)
                    let slot = 1 + pick(3) as u32;
                    let gen = live.records.generation(Slot::new(slot).unwrap(), BlockKind::Full).unwrap();
                    if gen >= 3 { continue; }
                    live.records.reuse(Slot::new(slot).unwrap(), BlockKind::Full);
                    rows.push(snapshot(&live, &seq_of, "reuse", &[], slot, &[], false));
                }
                5 => { // insert a block that is already in the tree
                    if live_ids.is_empty() { continue; }
                    let id = live_ids[pick(live_ids.len())];
                    let seq = seq_of[&id].clone();
                    let twin = live.handle(id).unwrap();
                    let (parent, pos) = if seq.len() == 2 { (None, 0) } else {
                        let pid = *seq_of.iter().find(|(_, s)| **s == seq[..2]).unwrap().0;
                        (Some(pid), BLOCK)
                    };
                    let block = [seq[seq.len() - 2], seq[seq.len() - 1]];
                    if live.records.generation(twin.slot(), twin.kind()) == Some(twin.generation()) {
                        // the twin is live: a dedup, the tree keeps its entry
                        let got = live.insert(parent, pos, &enc(block),
                            entry(twin.slot().get(), twin.generation(), BlockKind::Full)).unwrap();
                        assert_eq!(got, id, "the dedup returns the block already in the tree");
                        rows.push(snapshot(&live, &seq_of, "seal-dup", &seq, twin.slot().get(), &[], false));
                    } else {
                        // the twin is stale: evicted, the fresh entry sealed in its place
                        let slot = 1 + pick(3) as u32;
                        let gen = live.records.generation(Slot::new(slot).unwrap(), BlockKind::Full).unwrap();
                        let fresh = live.insert(parent, pos, &enc(block),
                            entry(slot, gen, BlockKind::Full)).unwrap();
                        assert_ne!(fresh, id, "the fresh block replaces the stale twin");
                        seq_of.retain(|k, _| *k != id);
                        seq_of.insert(fresh, seq.clone());
                        rows.push(snapshot(&live, &seq_of, "seal-twin", &seq, slot, &[], false));
                    }
                }
                6 => { // seek the model's request, wherever the walk is
                    let m = live.deepest_start(&req).unwrap();
                    rows.push(snapshot(&live, &seq_of, "seek", &[], 0, &req_abstract[..m.at / BLOCK * 2], true));
                }
                _ => { // remove a block and everything below it
                    if live_ids.is_empty() { continue; }
                    let id = live_ids[pick(live_ids.len())];
                    let seq = seq_of[&id].clone();
                    live.remove(id).unwrap();
                    rows.push(snapshot(&live, &seq_of, "remove", &seq, 0, &[], false));
                }
            }
            mutating += 1;
        }
        assert_eq!(mutating, 40, "the walk took its wandering steps");
        let m = live.deepest_start(&req).unwrap();
        rows.push(snapshot(&live, &seq_of, "seek", &[], 0, &req_abstract[..m.at / BLOCK * 2], true));

        let body = format!(
            "------------------------- MODULE PrefixTreeTraceData -------------------------\n\
             \\* Generated by prefix::tests::prefix_trace_dump (specs/run_tlc.zsh runs it).\n\
             \\* Do not edit; the generator is the code under test.\n\
             EXTENDS PrefixTree\n\nTrace == <<\n{}\n>>\n\n=============================================================================\n",
            rows.join(",\n"));
        std::fs::write(&out, body).unwrap();
    }


    #[test]
    fn entry_handles_come_back_as_johnnys_entries() {
        let full = BlockEntry::new(Slot::new(3).unwrap(), 1, 4096, BlockKind::Full);
        let win = BlockEntry::new(Slot::new(5).unwrap(), 2, 8192, BlockKind::Window);
        let mut t: EntryTree = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), full).unwrap();
        let b = t.insert(Some(a), BLOCK, &block(2), win).unwrap();
        t.retain_start(b).unwrap();
        let m = t.deepest_start(&[block(1), block(2), block(3)].concat());
        assert_eq!(m.at, 2 * BLOCK);
        assert_eq!(m.blocks, vec![full, win]);
        assert_eq!(m.blocks[0].kind(), BlockKind::Full);
        assert_eq!(m.blocks[0].offset(), 4096);
        assert_eq!(m.blocks[0].slot().get(), 3);
        assert_eq!(m.blocks[1].generation(), 2);
    }
}
