//! Thin, zero-overhead-ish wrapper around objc2-metal for compute work.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

/// Check every submitted command before returning the first failure. Buffers
/// use unretained references, so callers must finish draining before releasing resources.
pub fn wait_all<T>(
    commands: impl IntoIterator<Item = T>,
    mut wait: impl FnMut(T) -> Result<(), String>,
) -> Result<(), String> {
    let mut result = Ok(());
    for command in commands {
        result = result.and(wait(command));
    }
    result
}

/// Expose the oldest command's result only after it completes. On failure,
/// finish the commands already queued after it before returning the error.
pub fn wait_or_drain<T>(
    first: T,
    pending: impl IntoIterator<Item = T>,
    mut wait: impl FnMut(T) -> Result<(), String>,
) -> Result<(), String> {
    if let Err(error) = wait(first) {
        let _ = wait_all(pending, wait);
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
std::thread_local! {
    static INJECTED_WAIT_FAILURE: std::cell::Cell<Option<(&'static str, usize)>> = const { std::cell::Cell::new(None) };
    static WAIT_STAGE_CALLS: std::cell::RefCell<HashMap<String, usize>> = std::cell::RefCell::new(HashMap::new());
}

#[cfg(test)]
pub fn inject_wait_failure(stage: &'static str, failing_call_index: usize) {
    WAIT_STAGE_CALLS.with(|m| m.borrow_mut().clear());
    INJECTED_WAIT_FAILURE.with(|f| f.set(Some((stage, failing_call_index))));
}

#[cfg(test)]
pub fn clear_wait_failure() {
    WAIT_STAGE_CALLS.with(|m| m.borrow_mut().clear());
    INJECTED_WAIT_FAILURE.with(|f| f.set(None));
}

pub fn wait_completed(cb: &CmdBuf, stage: &str, pos: usize) -> Result<(), String> {
    cb.waitUntilCompleted();
    #[cfg(test)]
    if let Some((want_stage, want_idx)) = INJECTED_WAIT_FAILURE.with(|f| f.get()) {
        if stage == want_stage {
            let idx = WAIT_STAGE_CALLS.with(|m| {
                let mut map = m.borrow_mut();
                let c = map.entry(stage.to_string()).or_insert(0);
                let cur = *c;
                *c += 1;
                cur
            });
            if idx == want_idx {
                INJECTED_WAIT_FAILURE.with(|f| f.set(None));
                return Err(format!("Metal {stage} at token {pos} failed (InjectedTestError)"));
            }
        }
    }
    let status = cb.status();
    if status == MTLCommandBufferStatus::Completed { return Ok(()); }
    let detail = cb.error().map(|err| format!(": {}", err.localizedDescription())).unwrap_or_default();
    Err(format!("Metal {stage} at token {pos} failed ({status:?}){detail}"))
}

pub struct Gpu {
    pub dev: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    lib: Retained<ProtocolObject<dyn MTLLibrary>>,
    psos: std::cell::RefCell<HashMap<String, Pso>>,
}

impl Gpu {
    pub fn new(src: &str, kernels: &[&str]) -> Gpu {
        let dev = MTLCreateSystemDefaultDevice().expect("no Metal device");
        let queue = dev.newCommandQueue().expect("queue");
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Fast);
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let lib = dev
            .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
            .unwrap_or_else(|e| panic!("Metal compile failed:\n{}", e.localizedDescription()));
        let g = Gpu { dev, queue, lib, psos: Default::default() };
        for k in kernels { g.pso(k); }
        g
    }

    /// pipeline state by kernel name (compiled on first use; production kernels are pre-built at load)
    pub fn pso(&self, name: &str) -> Pso {
        if let Some(p) = self.psos.borrow().get(name) { return p.clone(); }
        let f = self.lib.newFunctionWithName(&NSString::from_str(name)).unwrap_or_else(|| panic!("kernel {name} missing"));
        let p = self.dev.newComputePipelineStateWithFunction_error(&f).unwrap_or_else(|e| panic!("pso {name}: {}", e.localizedDescription()));
        self.psos.borrow_mut().insert(name.to_string(), p.clone());
        p
    }

    pub fn buffer(&self, len: usize) -> Buf {
        self.dev.newBufferWithLength_options(len.max(16), MTLResourceOptions::StorageModeShared).expect("alloc buffer")
    }

    pub fn buffer_with(&self, data: &[u8]) -> Buf {
        let b = self.buffer(data.len());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents().as_ptr() as *mut u8, data.len()); }
        b
    }

    /// Wrap existing page-aligned memory (e.g. an mmapped model file) without copying.
    pub fn buffer_nocopy(&self, ptr: *mut u8, len: usize) -> Buf {
        unsafe {
            self.dev
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(ptr as *mut c_void).unwrap(), len, MTLResourceOptions::StorageModeShared, None)
                .expect("no-copy buffer (needs page-aligned ptr/len)")
        }
    }

    pub fn cmd(&self) -> CmdBuf { self.queue.commandBufferWithUnretainedReferences().expect("cmdbuf") }

    /// Metal's recommended maximum working set size in bytes, queried directly from MTLDevice.
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.dev.recommendedMaxWorkingSetSize()
    }

    /// Bytes currently allocated on the Metal device.
    pub fn current_allocated_size(&self) -> usize {
        self.dev.currentAllocatedSize()
    }
}

pub struct Enc {
    pub e: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Enc {
    pub fn new(cb: &CmdBuf) -> Enc {
        Enc { e: cb.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent).expect("encoder") }
    }
    #[inline] pub fn pso(&self, p: &Pso) { self.e.setComputePipelineState(p); }
    #[inline] pub fn buf(&self, b: &Buf, off: usize, idx: usize) { unsafe { self.e.setBuffer_offset_atIndex(Some(b), off, idx); } }
    #[inline] pub fn bytes<T>(&self, v: &T, idx: usize) {
        unsafe { self.e.setBytes_length_atIndex(NonNull::new(v as *const T as *mut c_void).unwrap(), std::mem::size_of::<T>(), idx); }
    }
    #[inline] pub fn bytes_slice<T>(&self, v: &[T], idx: usize) {
        unsafe { self.e.setBytes_length_atIndex(NonNull::new(v.as_ptr() as *mut c_void).unwrap(), std::mem::size_of_val(v), idx); }
    }
    #[inline] pub fn dispatch(&self, grid: (usize, usize, usize), tg: usize) {
        self.e.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 }, MTLSize { width: tg, height: 1, depth: 1 });
    }
    /// grid size read on the GPU from 3 uint32 at `off` bytes into `b` (written by an earlier kernel of the same encoder)
    #[inline] pub fn dispatch_indirect(&self, b: &Buf, off: usize, tg: usize) {
        unsafe { self.e.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(b, off, MTLSize { width: tg, height: 1, depth: 1 }); }
    }
    #[inline] pub fn barrier(&self) { self.e.memoryBarrierWithScope(MTLBarrierScope::Buffers); }
    pub fn end(self) { self.e.endEncoding(); }
}
