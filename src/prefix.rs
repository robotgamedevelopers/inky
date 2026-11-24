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
//! 16th block boundary a prefill crosses, a lane head): the tree records
//! which block ends are retained as start points, and a match returns the
//! deepest retained one on the matched path, with the block handles from
//! the root to it. Position 0 is always a start point and needs nothing.
//! The convolution state, the window blocks and the pool behind those
//! handles are the pool's and Johnny's interface's business, not this
//! file's: the matcher is behind the `PrefixMatch` trait so the scheduler
//! never names the concrete tree, and the block handle is the caller's
//! type. Nothing here hashes for the pool; it only matches tokens the
//! caller supplies.

use std::collections::HashMap;

/// Positions in one sealed block. Johnny's `blocks::BLOCK` when PR 49 v2 lands.
pub const BLOCK: usize = 64;

/// One sealed block in the tree. Ids are assigned by the tree and never reused.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SealedId(pub u32);

/// Why a block was not added. Nothing is added on any of these.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum InsertError {
    /// A sealed block is exactly `BLOCK` positions, never a partial tail.
    NotFullBlock(usize),
    /// `pos` must be the end of the parent chain (0 for a root block).
    WrongPosition { expected: usize, got: usize },
    /// The parent id names no block in this tree.
    UnknownParent,
}

/// The deepest start point a request matches.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct StartMatch<B> {
    /// Positions the start point covers: a multiple of `BLOCK`, or 0.
    pub at: usize,
    /// The tree block that ends at `at`; `None` only for position 0.
    pub start: Option<SealedId>,
    /// The caller's handles for the blocks from the root to `at`, in order.
    pub blocks: Vec<B>,
}

/// What the scheduler asks of a prefix matcher. The concrete tree in this
/// file is one implementation; the trait is the seam Rowan asked for while
/// the block interface (PR 49) is revised.
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
    start: bool,
    children: Vec<SealedId>,
}

/// FNV-1a over the parent, the position and the ordered token ids. A
/// lookup aid only: the exact token comparison always decides the match.
fn digest(parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |x: u64| h = (h ^ x).wrapping_mul(0x100000001b3);
    mix(parent.map(|p| p.0 as u64 + 1).unwrap_or(0));
    mix(pos as u64);
    for &t in tokens { mix(t as u64); }
    h
}

/// The prefix tree of sealed blocks.
pub struct PrefixTree<B> {
    nodes: Vec<Node<B>>,
    roots: Vec<SealedId>,
    by_digest: HashMap<u64, Vec<SealedId>>,
}

impl<B: Copy> PrefixTree<B> {
    pub fn new() -> PrefixTree<B> {
        PrefixTree { nodes: Vec::new(), roots: Vec::new(), by_digest: HashMap::new() }
    }

    fn node(&self, id: SealedId) -> Option<&Node<B>> { self.nodes.get(id.0 as usize) }

    /// Add one sealed block. `parent` is the block before it (`None` for
    /// the first block of a prefix) and `pos` its absolute start position.
    /// Adding the same block again (a racing publisher) returns the
    /// canonical block already in the tree, with its original handle:
    /// deduplication happens only after the exact token comparison.
    pub fn insert(&mut self, parent: Option<SealedId>, pos: usize, tokens: &[u32], handle: B) -> Result<SealedId, InsertError> {
        if tokens.len() != BLOCK { return Err(InsertError::NotFullBlock(tokens.len())); }
        let expected = match parent {
            None => 0,
            Some(p) => self.node(p).ok_or(InsertError::UnknownParent)?.pos + BLOCK,
        };
        if pos != expected { return Err(InsertError::WrongPosition { expected, got: pos }); }
        let d = digest(parent, pos, tokens);
        if let Some(bucket) = self.by_digest.get(&d) {
            for &id in bucket {
                let n = &self.nodes[id.0 as usize];
                if n.parent == parent && n.pos == pos && n.tokens.as_ref() == tokens { return Ok(id); }
            }
        }
        let id = SealedId(self.nodes.len() as u32);
        self.nodes.push(Node { parent, pos, digest: d, tokens: tokens.into(), handle, start: false, children: Vec::new() });
        match parent {
            None => self.roots.push(id),
            Some(p) => self.nodes[p.0 as usize].children.push(id),
        }
        self.by_digest.entry(d).or_default().push(id);
        Ok(id)
    }

    /// Keep the end of this block as a start point. Returns false when the
    /// id names no block.
    pub fn retain_start(&mut self, id: SealedId) -> bool {
        match self.nodes.get_mut(id.0 as usize) {
            Some(n) => { n.start = true; true }
            None => false,
        }
    }

    /// Drop a start point (its blocks stay in the tree as cold blocks).
    pub fn release_start(&mut self, id: SealedId) -> bool {
        match self.nodes.get_mut(id.0 as usize) {
            Some(n) => { let was = n.start; n.start = false; was }
            None => false,
        }
    }

    /// The child of `parent` (or a root) whose tokens are exactly this
    /// block. The digest bucket narrows the search; tokens decide.
    fn child(&self, parent: Option<SealedId>, pos: usize, tokens: &[u32]) -> Option<SealedId> {
        let bucket = self.by_digest.get(&digest(parent, pos, tokens))?;
        bucket.iter().copied().find(|&id| {
            let n = &self.nodes[id.0 as usize];
            n.parent == parent && n.pos == pos && n.tokens.as_ref() == tokens
        })
    }
}

impl<B: Copy> Default for PrefixTree<B> {
    fn default() -> Self { Self::new() }
}

impl<B: Copy> PrefixMatch<B> for PrefixTree<B> {
    fn deepest_start(&self, tokens: &[u32]) -> StartMatch<B> {
        let mut best = StartMatch { at: 0, start: None, blocks: Vec::new() };
        let mut parent = None;
        let mut chain: Vec<B> = Vec::new();
        let mut at = 0usize;
        while tokens.len() - at >= BLOCK {
            let Some(id) = self.child(parent, at, &tokens[at..at + BLOCK]) else { break };
            let n = &self.nodes[id.0 as usize];
            chain.push(n.handle);
            at += BLOCK;
            parent = Some(id);
            if n.start {
                best = StartMatch { at, start: Some(id), blocks: chain.clone() };
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(seed: u32) -> Vec<u32> { (0..BLOCK as u32).map(|i| seed * 1000 + i).collect() }

    fn tree3() -> (PrefixTree<u64>, Vec<SealedId>) {
        let mut t = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), 10).unwrap();
        let b = t.insert(Some(a), 64, &block(2), 20).unwrap();
        let c = t.insert(Some(b), 128, &block(3), 30).unwrap();
        (t, vec![a, b, c])
    }

    #[test]
    fn an_empty_tree_starts_at_zero() {
        let t: PrefixTree<u64> = PrefixTree::new();
        let m = t.deepest_start(&block(1));
        assert_eq!(m, StartMatch { at: 0, start: None, blocks: vec![] });
    }

    #[test]
    fn the_deepest_retained_start_point_wins() {
        let (mut t, ids) = tree3();
        t.retain_start(ids[0]);
        t.retain_start(ids[2]);
        let mut toks = block(1);
        toks.extend(block(2));
        toks.extend(block(3));
        toks.extend(block(9));
        let m = t.deepest_start(&toks);
        assert_eq!(m.at, 3 * BLOCK);
        assert_eq!(m.start, Some(ids[2]));
        assert_eq!(m.blocks, vec![10, 20, 30], "handles from the root to the start point, in order");
        // a released start point is skipped for the deeper retained one behind it
        t.release_start(ids[2]);
        let m = t.deepest_start(&toks);
        assert_eq!(m.at, BLOCK);
        assert_eq!(m.blocks, vec![10]);
    }

    #[test]
    fn a_block_that_differs_anywhere_stops_the_match() {
        let (mut t, ids) = tree3();
        t.retain_start(ids[2]);
        let mut toks = block(1);
        let mut second = block(2);
        second[63] ^= 1;
        toks.extend(second);
        toks.extend(block(3));
        let m = t.deepest_start(&toks);
        assert_eq!(m.at, 0, "the second block differs, so even the third block's start point is unreachable");
        // and a matched block with no retained start above it still starts at 0
        let (mut t2, ids2) = tree3();
        t2.retain_start(ids2[1]);
        let m = t2.deepest_start(&[block(1), block(7)].concat());
        assert_eq!(m.at, 0);
    }

    #[test]
    fn equal_local_tokens_under_different_parents_are_different_blocks() {
        let mut t: PrefixTree<u64> = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), 10).unwrap();
        let x = t.insert(None, 0, &block(5), 50).unwrap();
        // the same second block tokens under both histories
        let a2 = t.insert(Some(a), 64, &block(2), 20).unwrap();
        let x2 = t.insert(Some(x), 64, &block(2), 60).unwrap();
        assert_ne!(a2, x2);
        t.retain_start(a2);
        t.retain_start(x2);
        let m = t.deepest_start(&[block(5), block(2)].concat());
        assert_eq!(m.blocks, vec![50, 60], "the match follows the parent chain, never the local tokens alone");
        let m = t.deepest_start(&[block(1), block(2)].concat());
        assert_eq!(m.blocks, vec![10, 20]);
    }

    #[test]
    fn racing_publishers_dedupe_only_on_exact_tokens() {
        let mut t: PrefixTree<u64> = PrefixTree::new();
        let a = t.insert(None, 0, &block(1), 10).unwrap();
        let again = t.insert(None, 0, &block(1), 99).unwrap();
        assert_eq!(again, a, "the same block again is the canonical one, with its first handle");
        let other = t.insert(None, 0, &block(2), 20).unwrap();
        assert_ne!(other, a, "different tokens are a different root, not a dedupe");
    }

    #[test]
    fn unsealed_and_misplaced_blocks_never_enter() {
        let mut t: PrefixTree<u64> = PrefixTree::new();
        assert_eq!(t.insert(None, 0, &block(1)[..63], 10), Err(InsertError::NotFullBlock(63)), "a partial tail is unsealed and invisible");
        assert_eq!(t.insert(None, 64, &block(1), 10), Err(InsertError::WrongPosition { expected: 0, got: 64 }));
        assert_eq!(t.insert(Some(SealedId(7)), 64, &block(1), 10), Err(InsertError::UnknownParent));
        let a = t.insert(None, 0, &block(1), 10).unwrap();
        assert_eq!(t.insert(Some(a), 0, &block(2), 20), Err(InsertError::WrongPosition { expected: 64, got: 0 }));
        assert!(!t.retain_start(SealedId(7)));
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
        // the caller's start points: every 16th block boundary (position 1024)
        t.retain_start(ids[15]);
        let mut toks: Vec<u32> = (1..=17u32).flat_map(block).collect();
        let m = t.deepest_start(&toks);
        assert_eq!(m.at, 16 * BLOCK);
        // a trailing partial block matches nothing further
        toks.truncate(16 * BLOCK + 63);
        let m = t.deepest_start(&toks);
        assert_eq!(m.at, 16 * BLOCK, "63 tokens into block 17 is not a block");
        assert_eq!(m.blocks.len(), 16);
    }
}
