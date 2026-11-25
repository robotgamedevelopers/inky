//! Standalone entry point for the CPU build of the production serving cache.
#![allow(dead_code)]

extern crate libc;

mod block_cache;
mod blockbook;
mod blocks;
mod prefix;
mod spill;

mod engine {
    // The checker extracts EXT from src/engine.rs to avoid linking Metal.
    include!("extent.rs");
}

fn main() {
    let path = std::env::args_os().nth(1).expect("trace output path");
    block_cache::trace::generate(std::path::Path::new(&path));
}
