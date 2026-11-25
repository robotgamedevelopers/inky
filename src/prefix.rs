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
//! the entry alone goes stale. Three tree layers live in this module:
//! - `PrefixTree`: the pure token-digest and exact-token tree with reference-counted
//!   start points, independent of GPU pools.
//! - `EntryTree`: pairs each tree block with its `(Window, Full)` `BlockEntry` handles;
//!   `ServeBlockCache` (`src/serve/block_cache.rs`) and `Layer3Pool` (`src/poolgate.rs`)
//!   use `EntryTree` to match, retain, evict, and spill cross-request checkpoints.
//! - `LiveTree`: wraps `PrefixTree` directly over a single `BlockPool`'s slot records —
//!   cached blocks sit at zero pool refs until a lane pins them, stale generations are
//!   evicted on seek before matching, and pinned refs drop generation-checked.
//!
//! Modelled by `specs/PrefixTree.tla`; `LiveTree` is bound to that model
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

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::blocks::{BlockEntry, BlockError};
#[cfg(test)]
use crate::blocks::{BlockKind, Slot};

/// Positions in one sealed block: Johnny's constant, re-exported so this
/// file and `crate::blocks` can never disagree about a block's size.
pub use crate::blocks::BLOCK;

/// One sealed block in one tree. The id carries its tree's identity: an
/// id from another tree names nothing here, however its index looks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SealedId {
    tree: u64,
    index: u64,
}

impl SealedId {
    #[cfg(standalone_blocks)]
    pub(crate) fn index(self) -> u64 {
        self.index
    }
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
    /// All distinct block IDs in the tree have been issued.
    IdExhausted,
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
/// The two pool entries one sealed block carries at a start point:
/// its `Full` entry and, where the block has one, its `Window` entry
/// (BlockLayout.tla's two kinds for one block). The tree matches on
/// tokens; both entries ride on the node so a start point can be
/// installed without a second lookup.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockHandles {
    full: BlockEntry,
    window: Option<BlockEntry>,
}

impl BlockHandles {
    pub fn new(full: BlockEntry, window: Option<BlockEntry>) -> BlockHandles { BlockHandles { full, window } }
    pub fn full(self) -> BlockEntry { self.full }
    pub fn window(self) -> Option<BlockEntry> { self.window }
    #[cfg(test)]
    fn entries(self) -> impl Iterator<Item = BlockEntry> {
        std::iter::once(self.full).chain(self.window)
    }
}

pub type EntryTree = PrefixTree<BlockHandles>;

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
    namespace: u64,
    parent: Option<SealedId>,
    pos: usize,
    digest: u64,
    tokens: Box<[u32]>,
    handle: B,
    starts: u32,
    pins: u32,
    children: Vec<SealedId>,
}

/// FNV-1a over the namespace, parent, position and ordered token ids. A
/// lookup aid only: the exact namespace, parent, position and token comparison
/// always decides the match.
fn digest_in_namespace(namespace: u64, parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |x: u64| h = (h ^ x).wrapping_mul(0x100000001b3);
    mix(namespace);
    mix(parent.map(|p| p.index + 1).unwrap_or(0));
    mix(pos as u64);
    for &t in tokens { mix(t as u64); }
    h
}

/// The prefix tree of sealed blocks.
static NEXT_TREE: AtomicU64 = AtomicU64::new(1);

pub struct PrefixTree<B> {
    tree: u64,
    next_index: u64,
    // Keep live blocks in insertion order; IDs of evicted blocks remain invalid.
    nodes: BTreeMap<u64, Node<B>>,
    roots: Vec<SealedId>,
    by_digest: HashMap<u64, Vec<SealedId>>,
}

impl<B: Copy> PrefixTree<B> {
    pub fn new() -> PrefixTree<B> {
        PrefixTree {
            tree: NEXT_TREE.fetch_add(1, Ordering::Relaxed),
            next_index: 0,
            nodes: BTreeMap::new(),
            roots: Vec::new(),
            by_digest: HashMap::new(),
        }
    }

    fn node(&self, id: SealedId) -> Option<&Node<B>> {
        if id.tree != self.tree { return None; }
        self.nodes.get(&id.index)
    }

    fn node_mut(&mut self, id: SealedId) -> Option<&mut Node<B>> {
        if id.tree != self.tree { return None; }
        self.nodes.get_mut(&id.index)
    }

    /// Add one sealed block in the default namespace (`0`).
    pub fn insert(&mut self, parent: Option<SealedId>, pos: usize, tokens: &[u32], handle: B) -> Result<SealedId, InsertError> {
        self.insert_in_namespace(0, parent, pos, tokens, handle)
    }

    /// Add one sealed block scoped to `namespace`. `parent` is the block before
    /// it (`None` for the first block of a prefix) and must belong to the same
    /// `namespace`. Deduplication happens only within `namespace` after exact
    /// token comparison.
    pub fn insert_in_namespace(
        &mut self,
        namespace: u64,
        parent: Option<SealedId>,
        pos: usize,
        tokens: &[u32],
        handle: B,
    ) -> Result<SealedId, InsertError> {
        if tokens.len() != BLOCK { return Err(InsertError::NotFullBlock(tokens.len())); }
        let expected = match parent {
            None => 0,
            Some(p) => {
                if p.tree != self.tree { return Err(InsertError::ForeignId); }
                let pn = self.node(p).ok_or(InsertError::UnknownParent)?;
                if pn.namespace != namespace { return Err(InsertError::UnknownParent); }
                pn.pos + BLOCK
            }
        };
        if pos != expected { return Err(InsertError::WrongPosition { expected, got: pos }); }
        let d = digest_in_namespace(namespace, parent, pos, tokens);
        if let Some(bucket) = self.by_digest.get(&d) {
            for &id in bucket {
                let n = self.node(id).expect("digest buckets hold live blocks");
                if n.namespace == namespace && n.parent == parent && n.pos == pos && n.tokens.as_ref() == tokens {
                    return Ok(id);
                }
            }
        }
        let index = self.next_index;
        self.next_index = index.checked_add(1).ok_or(InsertError::IdExhausted)?;
        let id = SealedId { tree: self.tree, index };
        self.nodes.insert(index, Node {
            namespace, parent, pos, digest: d, tokens: tokens.into(), handle,
            starts: 0, pins: 0, children: Vec::new(),
        });
        match parent {
            None => self.roots.push(id),
            Some(p) => self.node_mut(p).expect("the parent was just checked").children.push(id),
        }
        self.by_digest.entry(d).or_default().push(id);
        Ok(id)
    }

    /// The block already in the tree for exactly this parent, position
    /// and tokens in the default namespace, if there is one.
    #[cfg(test)]
    fn find(&self, parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> Option<SealedId> {
        self.child(parent, pos, tokens)
    }

    /// Every live block with its handle, root-first in insertion order
    /// (a parent always precedes its children).
    #[cfg(any(test, standalone_blocks))]
    pub fn entries(&self) -> Vec<(SealedId, B)> {
        self.nodes.iter()
            .map(|(&index, n)| (SealedId { tree: self.tree, index }, n.handle))
            .collect()
    }

    #[cfg(standalone_blocks)]
    pub(crate) fn trace_block(&self, id: SealedId) -> Option<(Option<SealedId>, &[u32])> {
        self.node(id).map(|node| (node.parent, node.tokens.as_ref()))
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

    /// Replace a live block's stored handle in place (e.g. refreshing a stale window entry
    /// without removing descendants or breaking active start points).
    pub fn set_handle(&mut self, id: SealedId, handle: B) -> bool {
        match self.node_mut(id) {
            Some(n) => { n.handle = handle; true }
            None => false,
        }
    }

    /// Remove `id` and any ancestors that have no children, no start holders, and no active pins.
    pub fn prune_unreferenced(&mut self, mut id: SealedId) -> usize {
        let mut removed = 0usize;
        while let Some(n) = self.node(id) {
            if !n.children.is_empty() || n.starts > 0 || n.pins > 0 {
                break;
            }
            let parent = n.parent;
            self.remove(id);
            removed += 1;
            match parent {
                Some(p) => id = p,
                None => break,
            }
        }
        removed
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

    /// How many pins hold this block. Pins are counted here; the pool
    /// ref behind them is the live tree's business (one ref per block,
    /// taken at the first pin).
    #[cfg(any(test, standalone_blocks))]
    pub fn pin_count(&self, id: SealedId) -> u32 {
        self.node(id).map(|n| n.pins).unwrap_or(0)
    }

    /// Add one pin. `Ok(false)` when the id names no block here.
    pub fn pin(&mut self, id: SealedId) -> Result<bool, BlockError> {
        match self.node_mut(id) {
            Some(n) => { n.pins = n.pins.checked_add(1).ok_or(BlockError::Overflow)?; Ok(true) }
            None => Ok(false),
        }
    }

    /// Drop one pin. Returns false when the id names no block here or
    /// the block had no pin left.
    pub fn unpin(&mut self, id: SealedId) -> bool {
        match self.node_mut(id) {
            Some(n) if n.pins > 0 => { n.pins -= 1; true }
            _ => false,
        }
    }

    /// Remove this block and everything below it; the tree shrinks, and
    /// the removed handles come back root-first so the caller can drop
    /// whatever the blocks held. Ids are never reused. Returns an empty
    /// vec when the id names no block in this tree.
    pub fn remove(&mut self, id: SealedId) -> Vec<B> {
        self.remove_with_pins(id).into_iter().map(|(h, _)| h).collect()
    }

    /// `remove`, with each removed block's pin count beside its handle,
    /// root-first: the caller drops one pool ref per pinned block.
    pub fn remove_with_pins(&mut self, id: SealedId) -> Vec<(B, u32)> {
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
            let n = self.nodes.remove(&g.index).expect("collected blocks are live");
            if let Some(bucket) = self.by_digest.get_mut(&n.digest) {
                bucket.retain(|&b| b != g);
                if bucket.is_empty() { self.by_digest.remove(&n.digest); }
            }
            handles.push((n.handle, n.pins));
        }
        handles
    }

    /// The child of `parent` (or a root) whose tokens are exactly this
    /// block in the default namespace (`0`).
    #[cfg(test)]
    fn child(&self, parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> Option<SealedId> {
        self.child_in_namespace(0, parent, pos, tokens)
    }

    /// The child of `parent` (or a root) within `namespace` whose tokens are
    /// exactly this block. The digest bucket narrows the search; exact fields decide.
    fn child_in_namespace(&self, namespace: u64, parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> Option<SealedId> {
        let bucket = self.by_digest.get(&digest_in_namespace(namespace, parent, pos, tokens))?;
        for &id in bucket {
            let n = self.node(id)?;
            if n.namespace == namespace && n.parent == parent && n.pos == pos && n.tokens.as_ref() == tokens { // MUTANT_ANCHOR_NS_MATCH
                return Some(id);
            }
        }
        None
    }

    /// The deepest retained start point within `namespace` whose blocks are an
    /// exact prefix of `tokens`.
    pub fn deepest_start_in_namespace(&self, namespace: u64, tokens: &[u32]) -> StartMatch<B> {
        let mut parent = None;
        let mut ids: Vec<SealedId> = Vec::new();
        let mut chain: Vec<B> = Vec::new();
        let mut best_len = 0usize;
        let mut best_start = None;
        let mut at = 0usize;
        while tokens.len() - at >= BLOCK {
            let Some(id) = self.child_in_namespace(namespace, parent, at, &tokens[at..at + BLOCK]) else { break };
            let n = self.node(id).expect("a child id names a live block");
            ids.push(id);
            chain.push(n.handle);
            at += BLOCK;
            parent = Some(id);
            if n.starts > 0 { // MUTANT_ANCHOR_DEEPEST_NS
                best_len = ids.len();
                best_start = Some(id);
            }
        }
        if let Some(start) = best_start {
            ids.truncate(best_len);
            chain.truncate(best_len);
            StartMatch { at: best_len * BLOCK, start: Some(start), ids, blocks: chain }
        } else {
            StartMatch { at: 0, start: None, ids: Vec::new(), blocks: Vec::new() }
        }
    }
}

impl<B: Copy> Default for PrefixTree<B> {
    fn default() -> Self { Self::new() }
}

impl<B: Copy> PrefixMatch<B> for PrefixTree<B> {
    fn deepest_start(&self, tokens: &[u32]) -> StartMatch<B> {
        self.deepest_start_in_namespace(0, tokens)
    }
}

/// The pool's slot records, as the live matcher needs them. Johnny's pool
/// worker owns the records; this seam is only the three operations the
/// tree performs on them, so the tree never names the pool type.
#[cfg(test)]
pub trait SlotRecords {
    /// Take one more ref on a slot, for an entry at `generation`. The
    /// record must refuse when it is no longer at that generation:
    /// a generation-blind ref would pin (or, on drop, corrupt) the
    /// slot's new owner after a reuse.
    fn add_ref(&mut self, slot: Slot, kind: BlockKind, generation: u32) -> Result<(), BlockError>;
    /// Drop one ref, under the same generation check as `add_ref`.
    fn drop_ref(&mut self, slot: Slot, kind: BlockKind, generation: u32) -> Result<(), BlockError>;
    /// The generation the slot is at now, or `None` when it is not live.
    fn generation(&self, slot: Slot, kind: BlockKind) -> Option<u32>;
}

/// Why a live insert or removal failed.
#[cfg(test)]
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
    /// A handles pair whose entries are not one `Full` and at most one
    /// `Window` entry.
    WrongKind { expected: BlockKind, got: BlockKind },
}

#[cfg(test)]
impl From<InsertError> for LiveError {
    fn from(e: InsertError) -> LiveError { LiveError::Insert(e) }
}

#[cfg(test)]
impl From<BlockError> for LiveError {
    fn from(e: BlockError) -> LiveError { LiveError::Pool(e) }
}

/// The prefix matcher over live pool slots. A cached block sits at
/// zero pool refs: it is evictable, and lookup validates it by
/// generation instead of by holding it. A lane pins a block it is
/// actively stepping from; the first pin takes one generation-checked
/// ref on each of the block's entries, and the last unpin drops them
/// the same checked way, so a slot reused in between never has its
/// new owner's refcount corrupted. A seek evicts every stale block
/// (each with everything below it) before matching, as the model's
/// `Seek` does, and never returns the whole prompt: the final token
/// must remain for prefill, so an exact full match steps back one
/// block.
#[cfg(test)]
pub struct LiveTree<R> {
    tree: EntryTree,
    records: R,
}

#[cfg(test)]
impl<R: SlotRecords> LiveTree<R> {
    pub fn new(records: R) -> LiveTree<R> {
        LiveTree { tree: PrefixTree::new(), records }
    }

    fn check_current(&self, e: BlockEntry) -> Result<(), LiveError> {
        match self.records.generation(e.slot(), e.kind()) {
            Some(g) if g == e.generation() => Ok(()),
            current => Err(LiveError::StaleEntry { slot: e.slot(), claimed: e.generation(), current }),
        }
    }

    fn handles_current(&self, h: BlockHandles) -> Result<(), LiveError> {
        for e in h.entries() { self.check_current(e)?; }
        Ok(())
    }

    /// Drop the ref behind one entry, only while the record is still
    /// at the entry's generation. A slot reused since the entry was
    /// sealed has a new owner (and, on Emma's poolgate, its refcount
    /// was reset at the reuse): dropping then would corrupt it, so a
    /// generation mismatch skips the drop, never forces it.
    fn drop_ref_if_current(&mut self, e: BlockEntry) -> Result<(), LiveError> {
        if self.records.generation(e.slot(), e.kind()) == Some(e.generation()) {
            self.records.drop_ref(e.slot(), e.kind(), e.generation())?;
        }
        Ok(())
    }

    fn drop_pinned(&mut self, h: BlockHandles) -> Result<(), LiveError> {
        for e in h.entries() { self.drop_ref_if_current(e)?; }
        Ok(())
    }

    /// Add one sealed block. The block is cached at zero pool refs:
    /// no ref is taken here. Both entries must name their slots'
    /// current generations (a stale entry, or one naming a slot that
    /// is not live, is refused before anything is inserted, even when
    /// its tokens are already in the tree), and the pair must be one
    /// `Full` entry plus at most one `Window` entry. A block already
    /// in the tree (a racing publisher) returns the canonical block —
    /// unless the twin in the tree is stale: its slot has moved on, so
    /// the twin is evicted (its pinned refs dropped, checked) and the
    /// fresh block takes its place, also unpinned. The returned id's
    /// handles may be another publisher's entries; `handle` shows the
    /// pair the tree actually holds.
    pub fn insert(&mut self, parent: Option<SealedId>, pos: usize, tokens: &[u32], handles: BlockHandles) -> Result<SealedId, LiveError> {
        if handles.full.kind() != BlockKind::Full {
            return Err(LiveError::WrongKind { expected: BlockKind::Full, got: handles.full.kind() });
        }
        if let Some(w) = handles.window {
            if w.kind() != BlockKind::Window {
                return Err(LiveError::WrongKind { expected: BlockKind::Window, got: w.kind() });
            }
        }
        self.handles_current(handles)?;
        if tokens.len() == BLOCK {
            if let Some(id) = self.tree.find(parent, pos, tokens) {
                let twin = self.tree.handle(id).expect("a found block has handles");
                if self.handles_current(twin).is_ok() { return Ok(id); }
                self.remove(id)?; // the stale twin: evict it, then insert fresh below
            }
        }
        Ok(self.tree.insert(parent, pos, tokens, handles)?)
    }

    /// Pin a block for an active lane or step. The first pin takes one
    /// generation-checked ref on each of the block's entries; further
    /// pins only count. A stale block cannot be pinned. `Ok(false)`
    /// when the id names no block here.
    pub fn pin(&mut self, id: SealedId) -> Result<bool, LiveError> {
        let Some(handles) = self.tree.handle(id) else { return Ok(false) };
        self.handles_current(handles)?;
        if self.tree.pin_count(id) == 0 {
            let mut taken: Vec<BlockEntry> = Vec::new();
            for e in handles.entries() {
                match self.records.add_ref(e.slot(), e.kind(), e.generation()) {
                    Ok(()) => taken.push(e),
                    Err(err) => {
                        for t in taken { let _ = self.drop_ref_if_current(t); }
                        return Err(err.into());
                    }
                }
            }
        }
        self.tree.pin(id)?;
        Ok(true)
    }

    /// Drop one pin. At the last pin the entries' refs are dropped,
    /// generation-checked: a slot reused while pinned keeps its new
    /// owner's refcount untouched. `Ok(false)` when the id names no
    /// block here or the block had no pin left.
    pub fn unpin(&mut self, id: SealedId) -> Result<bool, LiveError> {
        let Some(handles) = self.tree.handle(id) else { return Ok(false) };
        if self.tree.pin_count(id) == 0 { return Ok(false); }
        if self.tree.pin_count(id) == 1 { self.drop_pinned(handles)?; }
        Ok(self.tree.unpin(id))
    }

    /// How many pins hold this block.
    pub fn pin_count(&self, id: SealedId) -> u32 { self.tree.pin_count(id) }

    /// Keep or drop one holder of a start point (see `PrefixTree`).
    pub fn retain_start(&mut self, id: SealedId) -> Result<bool, BlockError> { self.tree.retain_start(id) }
    pub fn release_start(&mut self, id: SealedId) -> bool { self.tree.release_start(id) }

    /// The handles the tree holds on a block, if it is here. After a
    /// racing publisher's `Ok(id)`, this is how the caller tells whose
    /// entries the tree kept.
    pub fn handle(&self, id: SealedId) -> Option<BlockHandles> { self.tree.handle(id) }

    /// Remove a block and everything below it. Pinned blocks among
    /// them drop their refs generation-checked (see
    /// `drop_ref_if_current`); idle blocks held no ref to drop.
    pub fn remove(&mut self, id: SealedId) -> Result<Vec<BlockHandles>, LiveError> {
        let removed = self.tree.remove_with_pins(id);
        let mut first_err = None;
        for (h, pins) in &removed {
            if *pins > 0 {
                if let Err(e) = self.drop_pinned(*h) {
                    if first_err.is_none() { first_err = Some(e); }
                }
            }
        }
        let handles: Vec<BlockHandles> = removed.into_iter().map(|(h, _)| h).collect();
        match first_err {
            None => Ok(handles),
            Some(e) => Err(e),
        }
    }

    /// The deepest start point whose blocks are all still the slots
    /// and generations the tree sealed them at. A seek first evicts
    /// every stale block in the tree — not only those on the matched
    /// path, as the model's `Seek` does — so a later lookup never
    /// meets it; a pool error from an eviction is returned, not
    /// swallowed. The result never covers the whole prompt when the
    /// prompt is non-empty: an exact full match steps back one block,
    /// leaving at least the final token to prefill.
    pub fn deepest_start(&mut self, tokens: &[u32]) -> Result<StartMatch<BlockHandles>, LiveError> {
        let stale: Vec<SealedId> = self.tree.entries().into_iter()
            .filter(|(_, h)| h.entries().any(|e| self.records.generation(e.slot(), e.kind()) != Some(e.generation())))
            .map(|(id, _)| id)
            .collect();
        for id in stale {
            // an ancestor's eviction already took this block: removing
            // it again removes nothing and drops no second ref
            self.remove(id)?;
        }
        let m = self.tree.deepest_start(tokens);
        if !tokens.is_empty() && m.at == tokens.len() {
            // exact full-prompt match: step back one block so the
            // suffix to prefill is never empty
            return Ok(self.tree.deepest_start(&tokens[..tokens.len() - BLOCK]));
        }
        Ok(m)
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
    fn pruned_prefixes_release_node_storage_and_leave_old_ids_invalid() {
        let mut tree = PrefixTree::new();
        let root = tree.insert(None, 0, &block(1), 11).unwrap();
        tree.retain_start(root).unwrap();
        let old = tree.insert(Some(root), BLOCK, &block(2), 12).unwrap();
        assert_eq!(tree.prune_unreferenced(old), 1);
        for i in 0..4096 {
            let tip = tree.insert(Some(root), BLOCK, &block(2), i).unwrap();
            assert_ne!(old, tip);
            assert!(!tree.pin(old).unwrap());
            assert!(tree.remove(old).is_empty());
            assert_eq!(tree.insert(Some(old), 2 * BLOCK, &block(3), 13), Err(InsertError::UnknownParent));
            assert_eq!(tree.entries(), vec![(root, 11), (tip, i)]);
            assert_eq!(tree.prune_unreferenced(tip), 1);
        }
        assert_eq!(tree.nodes.len(), 1, "evicted nodes must not accumulate");
        assert!(tree.release_start(root));
        assert_eq!(tree.prune_unreferenced(root), 1);
        assert!(tree.nodes.is_empty());
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
    fn exhausted_tree_ids_never_wrap_or_replace_existing_nodes() {
        let mut tree = PrefixTree::new();
        let root = tree.insert(None, 0, &block(1), 11).unwrap();
        tree.next_index = u64::MAX - 1;
        let tip = tree.insert(Some(root), BLOCK, &block(2), 12).unwrap();
        assert_eq!(tree.insert(Some(root), BLOCK, &block(2), 13), Ok(tip));
        assert_eq!(tree.insert(Some(tip), 2 * BLOCK, &block(3), 14), Err(InsertError::IdExhausted));
        assert_eq!(tree.entries(), vec![(root, 11), (tip, 12)]);
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
        fn reuse(&mut self, slot: Slot, kind: BlockKind) {
            let r = self.record(slot, kind);
            while r.refs() > 0 { r.drop_ref().unwrap(); }
            r.reissue().unwrap();
        }
    }

    impl SlotRecords for FakeRecords {
        fn add_ref(&mut self, slot: Slot, kind: BlockKind, generation: u32) -> Result<(), BlockError> {
            if self.generation(slot, kind) != Some(generation) { return Err(BlockError::Generation); }
            self.record(slot, kind).add_ref()
        }
        fn drop_ref(&mut self, slot: Slot, kind: BlockKind, generation: u32) -> Result<(), BlockError> {
            if self.generation(slot, kind) != Some(generation) { return Err(BlockError::Generation); }
            self.record(slot, kind).drop_ref()
        }
        fn generation(&self, slot: Slot, kind: BlockKind) -> Option<u32> {
            // an untouched slot reads as generation 1, the generation
            // `record` takes it at (fresh 0 bumped once on first use)
            Some(self.map.get(&Self::key(slot, kind)).map(|r| r.generation()).unwrap_or(1))
        }
    }

    fn entry(slot: u32, generation: u32, kind: BlockKind) -> BlockEntry {
        BlockEntry::new(Slot::new(slot).unwrap(), generation, 4096, kind, 2)
    }

    fn handles(slot: u32, generation: u32) -> BlockHandles {
        BlockHandles::new(entry(slot, generation, BlockKind::Full), None)
    }

    fn handles_win(slot: u32, generation: u32, wslot: u32, wgen: u32) -> BlockHandles {
        BlockHandles::new(entry(slot, generation, BlockKind::Full), Some(entry(wslot, wgen, BlockKind::Window)))
    }

    #[test]
    fn a_reused_slots_stale_entry_is_a_miss_never_a_match() {
        // Rowan's failing case: insert A at (slot 3, gen 1); the pool
        // reuses slot 3 at gen 2 for other tokens; A's tokens must miss,
        // and pinned refs must settle, not leak.
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), handles(3, 1)).unwrap();
        let b = live.insert(Some(a), BLOCK, &block(2), handles(4, 1)).unwrap();
        live.retain_start(b).unwrap();
        let toks = [block(1), block(2)].concat();
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "a cached block sits at zero refs");
        live.pin(a).unwrap();
        live.pin(b).unwrap();
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "the first pin took the ref");
        live.retain_start(a).unwrap();
        assert_eq!(live.deepest_start(&toks).unwrap().at, BLOCK, "exact full match steps back one block");
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // slot 3 is generation 2 now
        let m = live.deepest_start(&toks).unwrap();
        assert_eq!(m.at, 0, "the gen-1 entry for a gen-2 slot is stale: a miss, not a match");
        assert_eq!(live.deepest_start(&toks).unwrap().at, 0, "and it was evicted, not rechecked forever");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "no ref leaks on the reused slot");
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Full), 0, "the evicted child's pinned ref was dropped too");
        // a fresh block at the slot's new generation matches again
        let a2 = live.insert(None, 0, &block(1), handles(3, 2)).unwrap();
        live.retain_start(a2).unwrap();
        assert_eq!(live.deepest_start(&block(1)).unwrap().at, 0, "one block is the whole prompt: the suffix is never empty");
        assert_eq!(live.deepest_start(&[block(1), block(9)].concat()).unwrap().blocks, vec![handles(3, 2)]);
    }

    #[test]
    fn pin_on_a_child_checks_its_entries_and_seek_checks_its_path() {
        let mut live = LiveTree::new(FakeRecords::default());
        let parent = live.insert(None, 0, &block(1), handles(1, 1)).unwrap();
        let child_handles = handles_win(2, 1, 3, 1);
        let child = live.insert(Some(parent), BLOCK, &block(2), child_handles).unwrap();
        live.retain_start(child).unwrap();
        live.records.reuse(Slot::new(1).unwrap(), BlockKind::Full);

        assert_eq!(live.insert(Some(parent), BLOCK, &block(2), child_handles).unwrap(), child);
        assert!(live.pin(child).unwrap());
        assert_eq!(live.pin_count(parent), 0);
        for e in child_handles.entries() {
            assert_eq!(live.records.refs(e.slot(), e.kind()), 1);
        }

        let matched = live.deepest_start(&[block(1), block(2), block(3)].concat()).unwrap();
        assert_eq!(matched.at, 0);
        assert!(live.handle(child).is_none());
        for e in child_handles.entries() {
            assert_eq!(live.records.refs(e.slot(), e.kind()), 0);
        }
    }

    #[test]
    fn a_fresh_block_is_never_lost_to_its_stale_twin() {
        // the stale twin is still in the tree when the fresh block is
        // sealed at the slot's new generation: the dedupe must evict
        // the twin and keep the fresh block, unpinned like any cache entry
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), handles(3, 1)).unwrap();
        live.pin(a).unwrap();
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // gen 2; the gen-1 twin is stale
        let a2 = live.insert(None, 0, &block(1), handles(3, 2)).unwrap();
        assert_ne!(a2, a, "the fresh block is a new block, not the stale twin");
        assert_eq!(live.pin_count(a2), 0, "the fresh block starts unpinned");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0,
            "the twin's pin died with its generation; no blind drop touched the new owner");
        live.pin(a2).unwrap();
        live.retain_start(a2).unwrap();
        assert_eq!(live.deepest_start(&[block(1), block(9)].concat()).unwrap().blocks, vec![handles(3, 2)],
            "and the fresh block survives the next seek");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1);
    }

    #[test]
    fn a_stale_entry_is_refused_at_insert() {
        // the cold reader's probe: slot 3 is at generation 2, and a
        // generation-1 entry must not be accepted — for new tokens or
        // for tokens already in the tree — and must take no ref
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), handles(3, 1)).unwrap();
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // slot 3 is generation 2 now
        let stale = LiveError::StaleEntry { slot: Slot::new(3).unwrap(), claimed: 1, current: Some(2) };
        assert_eq!(live.insert(None, 0, &block(9), handles(3, 1)), Err(stale.clone()),
            "a gen-1 entry for a gen-2 slot is refused, not inserted");
        assert_eq!(live.insert(None, 0, &block(1), handles(3, 1)), Err(stale),
            "the same refusal when the tokens are already in the tree");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "no ref was taken by either refusal");
        assert_eq!(live.handle(a), Some(handles(3, 1)), "the tree is unchanged");
        // a fresh entry at the current generation still lands
        assert!(live.insert(None, 0, &block(9), handles(3, 2)).is_ok());
    }

    #[test]
    fn a_seek_evicts_stale_blocks_off_the_matched_path() {
        // the model's Seek evicts every stale block; the code does the
        // same, so a stale block the request never touches is gone
        // after any seek, not only after a matching one
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), handles(3, 1)).unwrap();
        let off = live.insert(None, 0, &block(7), handles(4, 1)).unwrap();
        live.retain_start(a).unwrap();
        live.pin(off).unwrap();
        live.records.reuse(Slot::new(4).unwrap(), BlockKind::Full); // slot 4 at gen 2: `off` is stale
        let m = live.deepest_start(&[block(1), block(9)].concat()).unwrap();
        assert_eq!(m.at, BLOCK, "the live path still matches");
        assert_eq!(live.handle(off), None, "the stale off-path block was evicted by the seek");
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Full), 0, "and no ref is left on the reused slot");
    }

    #[test]
    fn pins_take_and_drop_one_ref_per_block() {
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), handles_win(3, 1, 4, 1)).unwrap();
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "idle: no Full ref");
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Window), 0, "idle: no Window ref");
        live.pin(a).unwrap();
        live.pin(a).unwrap();
        assert_eq!(live.pin_count(a), 2);
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "two pins, one Full ref");
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Window), 1, "two pins, one Window ref");
        assert_eq!(live.handle(a).unwrap().window(), Some(entry(4, 1, BlockKind::Window)), "both entries ride on the node");
        live.unpin(a).unwrap();
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1, "a pin remains");
        live.unpin(a).unwrap();
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "the last unpin dropped the Full ref");
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Window), 0, "and the Window ref");
        assert_eq!(live.unpin(a), Ok(false), "no pin left to drop");
        // removing a pinned block drops its refs; removing an idle one drops none
        live.pin(a).unwrap();
        let gone = live.remove(a).unwrap();
        assert_eq!(gone, vec![handles_win(3, 1, 4, 1)]);
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0);
    }

    #[test]
    fn a_reused_slots_new_owner_keeps_its_refcount() {
        // the generation-checked drop: a block pinned at gen 1, its
        // slot reused to gen 2 and re-owned; evicting the stale block
        // must not drop the new owner's ref
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), handles(3, 1)).unwrap();
        live.pin(a).unwrap();
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full); // gen 2; Emma's poolgate resets refs there
        live.records.add_ref(Slot::new(3).unwrap(), BlockKind::Full, 2).unwrap(); // the new owner's ref
        live.remove(a).unwrap();
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 1,
            "the stale block's eviction skipped the drop: the new owner's ref stands");
        // and a generation-blind checked drop at the old generation refuses outright
        assert_eq!(live.records.drop_ref(Slot::new(3).unwrap(), BlockKind::Full, 1), Err(BlockError::Generation));
    }

    #[test]
    fn live_inserts_and_removes_balance_their_refs() {
        let mut live = LiveTree::new(FakeRecords::default());
        let a = live.insert(None, 0, &block(1), handles(3, 1)).unwrap();
        assert_eq!(live.insert(None, 0, &block(1), handles(3, 1)), Ok(a), "a racing publisher dedupes");
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "and still takes no ref");
        // a refused insert takes no ref either
        assert!(live.insert(Some(a), 0, &block(2), handles(4, 1)).is_err());
        assert_eq!(live.records.refs(Slot::new(4).unwrap(), BlockKind::Full), 0);
        live.pin(a).unwrap();
        let gone = live.remove(a).unwrap();
        assert_eq!(gone, vec![handles(3, 1)]);
        assert_eq!(live.records.refs(Slot::new(3).unwrap(), BlockKind::Full), 0, "removal dropped the pinned ref");
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
    /// the model's request — held starts, pins, a stale root, a
    /// stale-twin reinsert — then wanders, seeks included, so a broken
    /// seek writes a row the model cannot step to. Ignored by default;
    #[derive(Default, Clone, Copy)]
    struct WalkCoverage {
        reuse_full: usize,
        reuse_window: usize,
        stale_win_evicted_on_seek: usize,
        stale_full_evicted_on_seek: usize,
        deepest_two_blocks_over_one: usize,
        full_match_stepback: usize,
        stale_win_insert_refused: usize,
        stale_full_insert_refused: usize,
    }

    fn run_prefix_walk(seed: u64, with_steered_preamble: bool) -> (Vec<String>, WalkCoverage) {
        let mut live = LiveTree::new(FakeRecords::default());
        let mut seq_of: HashMap<SealedId, Vec<u32>> = HashMap::new();
        let mut rows: Vec<String> = Vec::new();
        let mut cov = WalkCoverage::default();
        let mut random = seed;
        let mut pick = move |n: usize| -> usize {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (random >> 32) as usize % n
        };

        let req_full_abs: Vec<u32> = vec![1, 2, 1, 2];
        let req_ext_abs: Vec<u32> = vec![1, 2, 1, 2, 1];
        let req_full: Vec<u32> = [enc([1, 2]), enc([1, 2])].concat();
        let req_ext: Vec<u32> = [enc([1, 2]), enc([1, 2]), vec![1]].concat();

        let snapshot = |live: &LiveTree<FakeRecords>,
                        seq_of: &HashMap<SealedId, Vec<u32>>,
                        op: &str,
                        s: &[u32],
                        slot: u32,
                        wslot: u32,
                        result: &[u32],
                        sought: bool,
                        req_abs: &[u32]|
         -> String {
            let mut sealed: Vec<String> = vec![tla_seq(&[])];
            let (mut starts, mut slot_of_m, mut gen_of, mut win_slot, mut win_gen, mut refs) = (
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
                HashMap::new(),
            );
            for (id, h) in live.tree.entries() {
                let seq = seq_of[&id].clone();
                sealed.push(tla_seq(&seq));
                starts.insert(seq.clone(), live.tree.start_holders(id) as i64);
                slot_of_m.insert(seq.clone(), h.full().slot().get() as i64);
                gen_of.insert(seq.clone(), h.full().generation() as i64 - 1);
                if let Some(w) = h.window() {
                    win_slot.insert(seq.clone(), w.slot().get() as i64);
                    win_gen.insert(seq.clone(), w.generation() as i64 - 1);
                }
                if live.pin_count(id) > 0 {
                    refs.insert(seq, 1);
                }
            }
            let pr_full = |sl: u32| live.records.pool_refs(Slot::new(sl).unwrap(), BlockKind::Full);
            let pr_win = |sl: u32| live.records.pool_refs(Slot::new(sl).unwrap(), BlockKind::Window);
            let pool_refs = format!(
                "[s \\in Slots |-> IF s = 1 THEN {} ELSE IF s = 2 THEN {} ELSE {}]",
                pr_full(1),
                pr_full(2),
                pr_full(3)
            );
            let win_pool_refs = format!(
                "[s \\in Slots |-> IF s = 1 THEN {} ELSE IF s = 2 THEN {} ELSE {}]",
                pr_win(1),
                pr_win(2),
                pr_win(3)
            );
            let fg = |sl: u32| live.records.generation(Slot::new(sl).unwrap(), BlockKind::Full).unwrap() - 1;
            let wg = |sl: u32| live.records.generation(Slot::new(sl).unwrap(), BlockKind::Window).unwrap() - 1;
            let pool = format!(
                "[s \\in Slots |-> IF s = 1 THEN {} ELSE IF s = 2 THEN {} ELSE {}]",
                fg(1),
                fg(2),
                fg(3)
            );
            let win_pool = format!(
                "[s \\in Slots |-> IF s = 1 THEN {} ELSE IF s = 2 THEN {} ELSE {}]",
                wg(1),
                wg(2),
                wg(3)
            );
            format!(
                "[op |-> \"{op}\", s |-> {}, slot |-> {slot}, wslot |-> {wslot}, sealed |-> {{{}}}, starts |-> {}, \
                 slotOf |-> {}, genOf |-> {}, winSlotOf |-> {}, winGenOf |-> {}, poolGen |-> {}, winPoolGen |-> {}, \
                 refs |-> {}, poolRefs |-> {}, winPoolRefs |-> {}, result |-> {}, sought |-> {}, req |-> {}, bad |-> FALSE]",
                tla_seq(s),
                sealed.join(", "),
                tla_fn(&starts),
                tla_fn(&slot_of_m),
                tla_fn(&gen_of),
                tla_fn(&win_slot),
                tla_fn(&win_gen),
                pool,
                win_pool,
                tla_fn(&refs),
                pool_refs,
                win_pool_refs,
                tla_seq(result),
                if sought { "TRUE" } else { "FALSE" },
                tla_seq(req_abs),
            )
        };

        let pair = |live: &LiveTree<FakeRecords>, slot: u32, wslot: u32| -> BlockHandles {
            let fg = live.records.generation(Slot::new(slot).unwrap(), BlockKind::Full).unwrap();
            let wg = live.records.generation(Slot::new(wslot).unwrap(), BlockKind::Window).unwrap();
            handles_win(slot, fg, wslot, wg)
        };

        let do_seek = |live: &mut LiveTree<FakeRecords>,
                       seq_of: &mut HashMap<SealedId, Vec<u32>>,
                       rows: &mut Vec<String>,
                       cov: &mut WalkCoverage,
                       use_ext: bool| {
            let had_stale_win_only = live.tree.entries().iter().any(|(_, h)| {
                let f = h.full();
                let f_ok = live.records.generation(f.slot(), BlockKind::Full) == Some(f.generation());
                let w_stale = h.window().is_some_and(|w| {
                    live.records.generation(w.slot(), BlockKind::Window) != Some(w.generation())
                });
                f_ok && w_stale
            });
            let had_stale_full_only = live.tree.entries().iter().any(|(_, h)| {
                let f = h.full();
                let f_stale = live.records.generation(f.slot(), BlockKind::Full) != Some(f.generation());
                let w_ok = h.window().map_or(true, |w| {
                    live.records.generation(w.slot(), BlockKind::Window) == Some(w.generation())
                });
                f_stale && w_ok
            });
            let held_b1 = seq_of
                .iter()
                .any(|(id, sq)| *sq == [1, 2] && live.tree.start_holders(*id) > 0);
            let held_b2 = seq_of
                .iter()
                .any(|(id, sq)| *sq == [1, 2, 1, 2] && live.tree.start_holders(*id) > 0);

            let (tokens, req_abs) = if use_ext {
                (&req_ext[..], &req_ext_abs[..])
            } else {
                (&req_full[..], &req_full_abs[..])
            };
            let m = live.deepest_start(tokens).unwrap();
            let live_ids: Vec<SealedId> = live.tree.entries().into_iter().map(|(id, _)| id).collect();
            seq_of.retain(|id, _| live_ids.contains(id));
            if had_stale_win_only {
                cov.stale_win_evicted_on_seek += 1;
            }
            if had_stale_full_only {
                cov.stale_full_evicted_on_seek += 1;
            }
            if use_ext && held_b1 && held_b2 && m.at == 2 * BLOCK {
                cov.deepest_two_blocks_over_one += 1;
            }
            if !use_ext && held_b2 && m.at < 2 * BLOCK {
                cov.full_match_stepback += 1;
            }
            rows.push(snapshot(
                live,
                seq_of,
                "seek",
                &[],
                0,
                0,
                &req_full_abs[..m.at / BLOCK * 2],
                true,
                req_abs,
            ));
        };

        rows.push(snapshot(&live, &seq_of, "init", &[], 0, 0, &[], false, &req_full_abs));

        if with_steered_preamble {
            let root = live.insert(None, 0, &enc([1, 2]), pair(&live, 1, 3)).unwrap();
            seq_of.insert(root, vec![1, 2]);
            rows.push(snapshot(&live, &seq_of, "seal", &[1, 2], 1, 3, &[], false, &req_full_abs));
            let child = live.insert(Some(root), BLOCK, &enc([1, 2]), pair(&live, 2, 1)).unwrap();
            seq_of.insert(child, req_full_abs.clone());
            rows.push(snapshot(&live, &seq_of, "seal", &req_full_abs, 2, 1, &[], false, &req_full_abs));
            live.retain_start(root).unwrap();
            rows.push(snapshot(&live, &seq_of, "retain", &[1, 2], 0, 0, &[], false, &req_full_abs));
            live.retain_start(child).unwrap();
            rows.push(snapshot(&live, &seq_of, "retain", &req_full_abs, 0, 0, &[], false, &req_full_abs));
            live.pin(root).unwrap();
            rows.push(snapshot(&live, &seq_of, "pin", &[1, 2], 0, 0, &[], false, &req_full_abs));
            live.pin(child).unwrap();
            rows.push(snapshot(&live, &seq_of, "pin", &req_full_abs, 0, 0, &[], false, &req_full_abs));
            // 1. Extended request (2 blocks + 1 token): deepest (2 blocks) beats shallowest (1 block).
            do_seek(&mut live, &mut seq_of, &mut rows, &mut cov, true);
            // 2. Exact 2-block request: steps back from 2 blocks to 1 block.
            do_seek(&mut live, &mut seq_of, &mut rows, &mut cov, false);
            assert!(live.release_start(root));
            rows.push(snapshot(&live, &seq_of, "release", &[1, 2], 0, 0, &[], false, &req_full_abs));
            live.unpin(child).unwrap();
            rows.push(snapshot(&live, &seq_of, "unpin", &req_full_abs, 0, 0, &[], false, &req_full_abs));
            do_seek(&mut live, &mut seq_of, &mut rows, &mut cov, false);
            live.unpin(root).unwrap();
            rows.push(snapshot(&live, &seq_of, "unpin", &[1, 2], 0, 0, &[], false, &req_full_abs));
            // 3. Reuse ONLY Window slot 3 (Full slot 1 stays at its original generation!).
            live.records.reuse(Slot::new(3).unwrap(), BlockKind::Window);
            cov.reuse_window += 1;
            rows.push(snapshot(&live, &seq_of, "reuse-window", &[], 0, 3, &[], false, &req_full_abs));
            do_seek(&mut live, &mut seq_of, &mut rows, &mut cov, true);
            // 4. Stale twin via Full-only reuse, then replaced by fresh insert.
            let root2 = live.insert(None, 0, &enc([1, 2]), pair(&live, 3, 2)).unwrap();
            seq_of.insert(root2, vec![1, 2]);
            rows.push(snapshot(&live, &seq_of, "seal", &[1, 2], 3, 2, &[], false, &req_full_abs));
            live.retain_start(root2).unwrap();
            rows.push(snapshot(&live, &seq_of, "retain", &[1, 2], 0, 0, &[], false, &req_full_abs));
            live.records.reuse(Slot::new(3).unwrap(), BlockKind::Full);
            cov.reuse_full += 1;
            rows.push(snapshot(&live, &seq_of, "reuse-full", &[], 3, 0, &[], false, &req_full_abs));
            let fresh = pair(&live, 2, 1);
            let root3 = live.insert(None, 0, &enc([1, 2]), fresh).unwrap();
            assert_ne!(root3, root2, "the fresh block replaces the stale twin");
            seq_of.retain(|id, _| *id != root2);
            seq_of.insert(root3, vec![1, 2]);
            rows.push(snapshot(&live, &seq_of, "seal-twin", &[1, 2], 2, 1, &[], false, &req_full_abs));
            live.retain_start(root3).unwrap();
            rows.push(snapshot(&live, &seq_of, "retain", &[1, 2], 0, 0, &[], false, &req_full_abs));
            live.pin(root3).unwrap();
            rows.push(snapshot(&live, &seq_of, "pin", &[1, 2], 0, 0, &[], false, &req_full_abs));
            do_seek(&mut live, &mut seq_of, &mut rows, &mut cov, true);
        }

        let mut mutating = 0;
        let mut attempts = 0;
        let target_steps = if with_steered_preamble { 50 } else { 90 };
        while mutating < target_steps && attempts < 1200 {
            attempts += 1;
            let live_ids: Vec<SealedId> = live.tree.entries().into_iter().map(|(id, _)| id).collect();
            seq_of.retain(|id, _| live_ids.contains(id));
            match pick(12) {
                0 | 1 => {
                    // Bias half of seals toward the model request prefix <<1, 2>> / <<1, 2, 1, 2>> so
                    // unsteered seeds reliably build 2-block held chains as well as arbitrary branches.
                    let want_req_chain = pick(2) == 0;
                    let (parent, seq, block) = if want_req_chain {
                        let root_id = seq_of.iter().find(|(_, s)| **s == [1, 2]).map(|(&id, _)| id);
                        match root_id {
                            None => (None, vec![1, 2], [1u32, 2u32]),
                            Some(rid) if !seq_of.values().any(|s| *s == [1, 2, 1, 2]) => {
                                // Ensure live parent before sealing child; if rid is stale, fall through.
                                let rh = live.handle(rid).unwrap();
                                let ok = live.records.generation(rh.full().slot(), BlockKind::Full)
                                    == Some(rh.full().generation())
                                    && rh.window().map_or(true, |w| {
                                        live.records.generation(w.slot(), BlockKind::Window) == Some(w.generation())
                                    });
                                if !ok {
                                    continue;
                                }
                                (Some(rid), vec![1, 2, 1, 2], [1u32, 2u32])
                            }
                            _ => continue,
                        }
                    } else {
                        let parents: Vec<Option<SealedId>> = std::iter::once(None)
                            .chain(live_ids.iter().filter(|id| seq_of[id].len() == 2).map(|id| Some(*id)))
                            .collect();
                        let p = parents[pick(parents.len())];
                        if let Some(pid) = p {
                            let ph = live.handle(pid).unwrap();
                            let ok = live.records.generation(ph.full().slot(), BlockKind::Full)
                                == Some(ph.full().generation())
                                && ph.window().map_or(true, |w| {
                                    live.records.generation(w.slot(), BlockKind::Window) == Some(w.generation())
                                });
                            if !ok {
                                continue;
                            }
                        }
                        let base: Vec<u32> = p.map(|pid| seq_of[&pid].clone()).unwrap_or_default();
                        let blk = [1 + pick(2) as u32, 1 + pick(2) as u32];
                        let mut sq = base;
                        sq.extend(blk);
                        if seq_of.values().any(|s| *s == sq) {
                            continue;
                        }
                        (p, sq, blk)
                    };
                    let slot = 1 + pick(3) as u32;
                    let wslot = 1 + pick(3) as u32;
                    let pos = if parent.is_none() { 0 } else { BLOCK };
                    let id = live
                        .insert(parent, pos, &enc(block), pair(&live, slot, wslot))
                        .expect("a fresh seal at the current generations");
                    seq_of.insert(id, seq.clone());
                    rows.push(snapshot(&live, &seq_of, "seal", &seq, slot, wslot, &[], false, &req_full_abs));
                }
                2 => {
                    // Refused stale insert on either Full or Window independently.
                    let test_win = pick(2) == 0;
                    let slot = 1 + pick(3) as u32;
                    let block = [1 + pick(2) as u32, 1 + pick(2) as u32];
                    if test_win {
                        let wgen = live.records.generation(Slot::new(slot).unwrap(), BlockKind::Window).unwrap();
                        if wgen < 2 {
                            continue;
                        }
                        let fslot = 1 + (slot % 3);
                        let fgen = live.records.generation(Slot::new(fslot).unwrap(), BlockKind::Full).unwrap();
                        let got = live.insert(None, 0, &enc(block), handles_win(fslot, fgen, slot, wgen - 1));
                        assert_eq!(
                            got,
                            Err(LiveError::StaleEntry {
                                slot: Slot::new(slot).unwrap(),
                                claimed: wgen - 1,
                                current: Some(wgen),
                            })
                        );
                        cov.stale_win_insert_refused += 1;
                        rows.push(snapshot(
                            &live,
                            &seq_of,
                            "seal-stale",
                            &block.to_vec(),
                            fslot,
                            slot,
                            &[],
                            false,
                            &req_full_abs,
                        ));
                    } else {
                        let fgen = live.records.generation(Slot::new(slot).unwrap(), BlockKind::Full).unwrap();
                        if fgen < 2 {
                            continue;
                        }
                        let got = live.insert(None, 0, &enc(block), handles(slot, fgen - 1));
                        assert_eq!(
                            got,
                            Err(LiveError::StaleEntry {
                                slot: Slot::new(slot).unwrap(),
                                claimed: fgen - 1,
                                current: Some(fgen),
                            })
                        );
                        cov.stale_full_insert_refused += 1;
                        rows.push(snapshot(
                            &live,
                            &seq_of,
                            "seal-stale",
                            &block.to_vec(),
                            slot,
                            0,
                            &[],
                            false,
                            &req_full_abs,
                        ));
                    }
                }
                3 => {
                    let held: Vec<SealedId> = live_ids
                        .iter()
                        .copied()
                        .filter(|id| live.tree.start_holders(*id) < 2)
                        .collect();
                    if held.is_empty() {
                        continue;
                    }
                    let id = held[pick(held.len())];
                    live.retain_start(id).unwrap();
                    let seq = seq_of[&id].clone();
                    rows.push(snapshot(&live, &seq_of, "retain", &seq, 0, 0, &[], false, &req_full_abs));
                }
                4 => {
                    let held: Vec<SealedId> = live_ids
                        .iter()
                        .copied()
                        .filter(|id| live.tree.start_holders(*id) > 0)
                        .collect();
                    if held.is_empty() {
                        continue;
                    }
                    let id = held[pick(held.len())];
                    assert!(live.release_start(id));
                    let seq = seq_of[&id].clone();
                    rows.push(snapshot(&live, &seq_of, "release", &seq, 0, 0, &[], false, &req_full_abs));
                }
                5 => {
                    // Reuse ONLY Full on a slot not pinned on Full.
                    let slot = 1 + pick(3) as u32;
                    let gen = live.records.generation(Slot::new(slot).unwrap(), BlockKind::Full).unwrap();
                    if gen >= 3 {
                        continue;
                    }
                    let pinned_full: Vec<u32> = live
                        .tree
                        .entries()
                        .into_iter()
                        .filter(|(id, _)| live.pin_count(*id) > 0)
                        .map(|(_, h)| h.full().slot().get())
                        .collect();
                    if pinned_full.contains(&slot) {
                        continue;
                    }
                    live.records.reuse(Slot::new(slot).unwrap(), BlockKind::Full);
                    cov.reuse_full += 1;
                    rows.push(snapshot(&live, &seq_of, "reuse-full", &[], slot, 0, &[], false, &req_full_abs));
                }
                6 => {
                    // Reuse ONLY Window on a slot not pinned on Window.
                    let wslot = 1 + pick(3) as u32;
                    let gen = live.records.generation(Slot::new(wslot).unwrap(), BlockKind::Window).unwrap();
                    if gen >= 3 {
                        continue;
                    }
                    let pinned_win: Vec<u32> = live
                        .tree
                        .entries()
                        .into_iter()
                        .filter(|(id, _)| live.pin_count(*id) > 0)
                        .filter_map(|(_, h)| h.window().map(|w| w.slot().get()))
                        .collect();
                    if pinned_win.contains(&wslot) {
                        continue;
                    }
                    live.records.reuse(Slot::new(wslot).unwrap(), BlockKind::Window);
                    cov.reuse_window += 1;
                    rows.push(snapshot(&live, &seq_of, "reuse-window", &[], 0, wslot, &[], false, &req_full_abs));
                }
                7 => {
                    if live_ids.is_empty() {
                        continue;
                    }
                    let id = live_ids[pick(live_ids.len())];
                    let seq = seq_of[&id].clone();
                    let twin = live.handle(id).unwrap();
                    let (parent, pos) = if seq.len() == 2 {
                        (None, 0)
                    } else {
                        let pid = *seq_of.iter().find(|(_, s)| **s == seq[..2]).unwrap().0;
                        let ph = live.handle(pid).unwrap();
                        let parent_ok = live.records.generation(ph.full().slot(), BlockKind::Full)
                            == Some(ph.full().generation())
                            && ph.window().map_or(true, |w| {
                                live.records.generation(w.slot(), BlockKind::Window) == Some(w.generation())
                            });
                        if !parent_ok {
                            continue;
                        }
                        (Some(pid), BLOCK)
                    };
                    let block = [seq[seq.len() - 2], seq[seq.len() - 1]];
                    let current = |e: BlockEntry| live.records.generation(e.slot(), e.kind()) == Some(e.generation());
                    if current(twin.full()) && twin.window().map(current).unwrap_or(true) {
                        let got = live.insert(parent, pos, &enc(block), twin).unwrap();
                        assert_eq!(got, id, "the dedup returns the block already in the tree");
                        rows.push(snapshot(
                            &live,
                            &seq_of,
                            "seal-dup",
                            &seq,
                            twin.full().slot().get(),
                            twin.window().map(|w| w.slot().get()).unwrap_or(0),
                            &[],
                            false,
                            &req_full_abs,
                        ));
                    } else {
                        let slot = 1 + pick(3) as u32;
                        let wslot = 1 + pick(3) as u32;
                        let fresh = live.insert(parent, pos, &enc(block), pair(&live, slot, wslot)).unwrap();
                        assert_ne!(fresh, id, "the fresh block replaces the stale twin");
                        let remaining: Vec<SealedId> = live.tree.entries().into_iter().map(|(eid, _)| eid).collect();
                        seq_of.retain(|k, _| remaining.contains(k));
                        seq_of.insert(fresh, seq.clone());
                        rows.push(snapshot(&live, &seq_of, "seal-twin", &seq, slot, wslot, &[], false, &req_full_abs));
                    }
                }
                8 | 9 => {
                    let use_ext = pick(2) == 0;
                    do_seek(&mut live, &mut seq_of, &mut rows, &mut cov, use_ext);
                }
                10 => {
                    let cand: Vec<SealedId> = live_ids
                        .iter()
                        .copied()
                        .filter(|id| live.pin_count(*id) == 0)
                        .collect();
                    if cand.is_empty() {
                        continue;
                    }
                    let id = cand[pick(cand.len())];
                    if live.pin(id).is_err() {
                        continue;
                    }
                    let seq = seq_of[&id].clone();
                    rows.push(snapshot(&live, &seq_of, "pin", &seq, 0, 0, &[], false, &req_full_abs));
                }
                11 => {
                    let cand: Vec<SealedId> = live_ids
                        .iter()
                        .copied()
                        .filter(|id| live.pin_count(*id) > 0)
                        .collect();
                    if cand.is_empty() {
                        continue;
                    }
                    let id = cand[pick(cand.len())];
                    live.unpin(id).unwrap();
                    let seq = seq_of[&id].clone();
                    rows.push(snapshot(&live, &seq_of, "unpin", &seq, 0, 0, &[], false, &req_full_abs));
                }
                _ => {
                    if live_ids.is_empty() {
                        continue;
                    }
                    let id = live_ids[pick(live_ids.len())];
                    let seq = seq_of[&id].clone();
                    live.remove(id).unwrap();
                    rows.push(snapshot(&live, &seq_of, "remove", &seq, 0, 0, &[], false, &req_full_abs));
                }
            }
            mutating += 1;
        }
        assert_eq!(mutating, target_steps, "seed {seed:#x} completed all wandering steps");
        do_seek(&mut live, &mut seq_of, &mut rows, &mut cov, true);
        do_seek(&mut live, &mut seq_of, &mut rows, &mut cov, false);
        (rows, cov)
    }

    /// `specs/run_tlc.zsh` runs it before the trace check.
    #[test]
    #[ignore = "trace dump; run by specs/run_tlc.zsh"]
    fn prefix_trace_dump() {
        let out = std::env::var("INKY_PREFIX_TRACE_OUT")
            .unwrap_or_else(|_| "specs/PrefixTreeTraceData.tla".to_string());
        let base_seed: u64 = std::env::var("INKY_PREFIX_TRACE_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0x9e37_79b9_7f4a_7c15);
        let seeds = [
            (base_seed, true),
            (base_seed ^ 0x517c_c1b7_2722_0a95, false),
            (base_seed.wrapping_add(0x94d0_49bb_1331_11eb), false),
            (base_seed.wrapping_add(0xbf58_476d_1ce4_e5b9), false),
        ];
        let mut traces_tla = Vec::with_capacity(seeds.len());
        let mut total = WalkCoverage::default();
        let mut unsteered_win_evict = 0usize;
        let mut unsteered_full_evict = 0usize;
        for (idx, &(seed, steered)) in seeds.iter().enumerate() {
            let (rows, cov) = run_prefix_walk(seed, steered);
            assert!(cov.reuse_full > 0 && cov.reuse_window > 0, "trace {idx} missed independent kind reuse");
            if !steered {
                unsteered_win_evict += cov.stale_win_evicted_on_seek;
                unsteered_full_evict += cov.stale_full_evicted_on_seek;
            }
            total.reuse_full += cov.reuse_full;
            total.reuse_window += cov.reuse_window;
            total.stale_win_evicted_on_seek += cov.stale_win_evicted_on_seek;
            total.stale_full_evicted_on_seek += cov.stale_full_evicted_on_seek;
            total.deepest_two_blocks_over_one += cov.deepest_two_blocks_over_one;
            total.full_match_stepback += cov.full_match_stepback;
            total.stale_win_insert_refused += cov.stale_win_insert_refused;
            total.stale_full_insert_refused += cov.stale_full_insert_refused;
            traces_tla.push(format!("<<\n{}\n>>", rows.join(",\n")));
        }
        assert!(unsteered_win_evict > 0, "unsteered seeds must independently evict stale Window blocks on seek");
        assert!(unsteered_full_evict > 0, "unsteered seeds must independently evict stale Full blocks on seek");
        assert!(total.deepest_two_blocks_over_one > 0, "traces must exercise 2-block deepest over 1-block shallowest");
        assert!(total.full_match_stepback > 0, "traces must exercise exact full-match step-back");
        assert!(total.stale_win_insert_refused > 0, "traces must exercise refused stale Window insert");
        assert!(total.stale_full_insert_refused > 0, "traces must exercise refused stale Full insert");

        let body = format!(
            "------------------------- MODULE PrefixTreeTraceData -------------------------\n\
             \\* Generated by prefix::tests::prefix_trace_dump (specs/run_tlc.zsh runs it).\n\
             \\* Do not edit; the generator is the code under test.\n\
             EXTENDS PrefixTree\n\nTraces == <<\n{}\n>>\n\n=============================================================================\n",
            traces_tla.join(",\n")
        );
        std::fs::write(&out, body).unwrap();
    }

    #[test]
    fn namespace_isolates_prefix_lookups_and_deduplication() {
        let mut t: PrefixTree<u64> = PrefixTree::new();
        let toks = [block(7), block(8), vec![99]].concat();
        let a1 = t.insert_in_namespace(101, None, 0, &block(7), 11).unwrap();
        let a2 = t.insert_in_namespace(101, Some(a1), BLOCK, &block(8), 12).unwrap();
        t.retain_start(a1).unwrap();
        t.retain_start(a2).unwrap();

        // Default namespace 0 and another principal's namespace 202 see zero hits.
        assert_eq!(t.deepest_start(&toks).at, 0);
        assert_eq!(t.deepest_start_in_namespace(202, &toks).at, 0);

        // Principal 101 hits its deepest 2-block checkpoint.
        let m101 = t.deepest_start_in_namespace(101, &toks);
        assert_eq!(m101.at, 2 * BLOCK);
        assert_eq!(m101.blocks, vec![11, 12]);

        // Cross-namespace child insertion under another namespace's parent is refused.
        assert_eq!(
            t.insert_in_namespace(202, Some(a1), BLOCK, &block(8), 99),
            Err(InsertError::UnknownParent)
        );

        // Principal 202 inserting the same tokens gets a distinct node and handle.
        let b1 = t.insert_in_namespace(202, None, 0, &block(7), 21).unwrap();
        assert_ne!(a1, b1);
        t.retain_start(b1).unwrap();
        let m202 = t.deepest_start_in_namespace(202, &toks);
        assert_eq!(m202.at, BLOCK);
        assert_eq!(m202.blocks, vec![21]);
    }

    #[test]
    fn live_tree_distinguishes_full_from_window_and_deepest_across_seeds() {
        for s in 1u64..=16 {
            let (_rows, cov) = run_prefix_walk(s.wrapping_mul(0x9e37_79b9_7f4a_7c15), false);
            assert!(cov.reuse_full > 0 && cov.reuse_window > 0, "seed {s} must reuse Full and Window independently");
        }
        // Explicit deterministic check that reusing ONLY a Window slot invalidates the block even when Full is live.
        let mut live = LiveTree::new(FakeRecords::default());
        let h1 = handles_win(1, 1, 2, 1);
        let r = live.insert(None, 0, &enc([1, 2]), h1).unwrap();
        live.retain_start(r).unwrap();
        let c = live.insert(Some(r), BLOCK, &enc([1, 2]), handles_win(2, 1, 3, 1)).unwrap();
        live.retain_start(c).unwrap();
        let ext = [enc([1, 2]), enc([1, 2]), vec![1]].concat();
        assert_eq!(live.deepest_start(&ext).unwrap().at, 2 * BLOCK);
        // Reuse ONLY Window slot 3 (child's Window slot); child's Full slot 2 and root remain live!
        live.records.reuse(Slot::new(3).unwrap(), BlockKind::Window);
        let after = live.deepest_start(&ext).unwrap();
        assert_eq!(after.at, BLOCK, "stale Window on child must evict child while keeping live root");
    }


    #[test]
    fn entry_handles_come_back_as_johnnys_entries() {
        let full = BlockEntry::new(Slot::new(3).unwrap(), 1, 4096, BlockKind::Full, 2);
        let win = BlockEntry::new(Slot::new(5).unwrap(), 2, 8192, BlockKind::Window, 2);
        let full2 = BlockEntry::new(Slot::new(4).unwrap(), 1, 4096, BlockKind::Full, 2);
        let ha = BlockHandles::new(full, Some(win));
        let hb = BlockHandles::new(full2, None);
        let mut t: EntryTree = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), ha).unwrap();
        let b = t.insert(Some(a), BLOCK, &block(2), hb).unwrap();
        t.retain_start(b).unwrap();
        let m = t.deepest_start(&[block(1), block(2), block(3)].concat());
        assert_eq!(m.at, 2 * BLOCK);
        assert_eq!(m.blocks, vec![ha, hb]);
        assert_eq!(m.blocks[0].full().kind(), BlockKind::Full);
        assert_eq!(m.blocks[0].full().offset(), 4096);
        assert_eq!(m.blocks[0].full().slot().get(), 3);
        assert_eq!(m.blocks[0].window(), Some(win), "the Window entry rides on the same node");
        assert_eq!(m.blocks[0].window().unwrap().generation(), 2);
        assert_eq!(m.blocks[1].window(), None);
    }

    #[test]
    fn deepest_start_materializes_path_once_without_intermediate_clones() {
        let mut tree: PrefixTree<u32> = PrefixTree::new();
        let depth = 384usize;
        let mut tokens = Vec::with_capacity((depth + 2) * BLOCK);
        let mut parent = None;
        let mut expected_ids = Vec::with_capacity(depth);
        let mut expected_handles = Vec::with_capacity(depth);
        for b in 0..depth {
            let blk = block((b + 1) as u32);
            tokens.extend_from_slice(&blk);
            let id = tree.insert(parent, b * BLOCK, &blk, b as u32).unwrap();
            // Retain a start point at EVERY block so the old O(depth * starts) clone loop
            // would have cloned `ids` and `chain` 384 times (~73,920 elements copied).
            tree.retain_start(id).unwrap();
            expected_ids.push(id);
            expected_handles.push(b as u32);
            parent = Some(id);
        }
        // Append two unretained blocks + a non-matching tail block: deepest_start must
        // truncate cleanly back to `depth` without cloning at intermediate checkpoints.
        for extra in 0..2usize {
            let b = depth + extra;
            let blk = block((b + 1) as u32);
            tokens.extend_from_slice(&blk);
            let id = tree.insert(parent, b * BLOCK, &blk, b as u32).unwrap();
            parent = Some(id);
        }
        tokens.extend_from_slice(&block(999_999));

        let m = tree.deepest_start(&tokens);
        assert_eq!(m.at, depth * BLOCK);
        assert_eq!(m.start, expected_ids.last().copied());
        assert_eq!(m.ids, expected_ids);
        assert_eq!(m.blocks, expected_handles);

        // No retained starts along a matched chain -> empty Vecs with zero capacity retained.
        let mut no_starts: PrefixTree<u32> = PrefixTree::new();
        let a = no_starts.insert(None, 0, &block(7), 7).unwrap();
        let _b = no_starts.insert(Some(a), BLOCK, &block(8), 8).unwrap();
        let miss = no_starts.deepest_start(&[block(7), block(8)].concat());
        assert_eq!(miss.at, 0);
        assert_eq!(miss.start, None);
        assert!(miss.ids.is_empty() && miss.ids.capacity() == 0);
        assert!(miss.blocks.is_empty() && miss.blocks.capacity() == 0);
    }
}
