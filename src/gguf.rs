//! Zero-copy GGUF reader: mmaps the file and parses metadata + the tensor table.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::os::unix::io::AsRawFd;
use std::ptr::NonNull;

pub const T_F32: u32 = 0;
pub const T_Q5_0: u32 = 6;
pub const T_Q8_0: u32 = 8;
pub const T_Q4_K: u32 = 12;
pub const T_Q6_K: u32 = 14;
pub const T_BF16: u32 = 30;
/// Native expert weight formats (chadlyb/erisinky): NVFP4 and MXFP8 blocks of 32 weights
pub const T_NV4: u32 = 200;
pub const T_MX8: u32 = 201;

/// (bytes per block, elements per block)
fn type_block(t: u32) -> Option<(usize, usize)> {
    Some(match t {
        0 => (4, 1),
        1 | 30 => (2, 1),
        2 => (18, 32),
        6 => (22, 32),
        8 => (34, 32),
        12 => (144, 256),
        13 => (176, 256),
        14 => (210, 256),
        200 => (18, 32),
        201 => (34, 32),
        _ => return None,
    })
}

pub fn invalid_model(reason: impl std::fmt::Display) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, format!("invalid model: {reason}"))
}

#[derive(Debug, Clone)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    B(bool),
    S(String),
    AI(Vec<i64>),
    #[allow(dead_code)]
    AF(Vec<f64>),
    AS(Vec<String>),
    AB(Vec<bool>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub ne: Vec<usize>,
    pub ty: u32,
    /// Byte offset from the start of the mapped file.
    pub file_off: usize,
    pub row_bytes: usize,
    nbytes: usize,
}

impl Tensor {
    fn from_decl(name: &str, ne: Vec<usize>, ty: u32, file_off: usize) -> io::Result<Self> {
        let (block_bytes, block_elems) = type_block(ty)
            .ok_or_else(|| invalid_model(format_args!("tensor {name} has unsupported ggml type {ty}")))?;
        if !(1..=4).contains(&ne.len()) || ne.contains(&0) {
            return Err(invalid_model(format_args!("tensor {name} has invalid dimensions {ne:?}")));
        }
        if ne[0] % block_elems != 0 {
            return Err(invalid_model(format_args!("tensor {name} has an incomplete quantization block in each row")));
        }
        let row_blocks = ne[0] / block_elems;
        let overflow = || invalid_model(format_args!("tensor {name} byte size overflows"));
        let row_bytes = row_blocks.checked_mul(block_bytes).ok_or_else(overflow)?;
        let nbytes = ne[1..].iter().try_fold(row_bytes, |n, &dim| n.checked_mul(dim))
            .ok_or_else(overflow)?;
        Ok(Self { ne, ty, file_off, row_bytes, nbytes })
    }

    fn locate(&mut self, name: &str, data_off: usize, align: usize, file_len: usize) -> io::Result<()> {
        if self.file_off % align != 0 {
            return Err(invalid_model(format_args!("tensor {name} has an unaligned offset")));
        }
        let out_of_bounds = || invalid_model(format_args!("tensor {name} extends beyond the file"));
        let start = data_off.checked_add(self.file_off).ok_or_else(out_of_bounds)?;
        let end = start.checked_add(self.nbytes).ok_or_else(out_of_bounds)?;
        if end > file_len {
            return Err(out_of_bounds());
        }
        self.file_off = start;
        Ok(())
    }

    pub fn check(&self, name: &str, shape: &[usize], types: &[u32]) -> io::Result<()> {
        if self.ne != shape || !types.contains(&self.ty) {
            return Err(invalid_model(format_args!(
                "tensor {name} has shape {:?} and ggml type {}; expected {shape:?} and one of {types:?}",
                self.ne, self.ty,
            )));
        }
        Ok(())
    }
}

pub struct Gguf {
    _file: File,
    map: NonNull<u8>,
    map_len: usize,
    file_len: usize,
    pub kv: HashMap<String, Value>,
    tensors: HashMap<String, Tensor>,
}

unsafe impl Send for Gguf {}
unsafe impl Sync for Gguf {}

impl Drop for Gguf {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.map.as_ptr().cast(), self.map_len) };
    }
}

struct Rd<'a> { rest: &'a [u8] }
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let (bytes, rest) = self.rest.split_at_checked(n)
            .ok_or_else(|| invalid_model("truncated GGUF header"))?;
        self.rest = rest;
        Ok(bytes)
    }
    fn u8(&mut self) -> io::Result<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> io::Result<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> io::Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> io::Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn size(&mut self) -> io::Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| invalid_model("GGUF size exceeds the address space"))
    }
    fn str(&mut self) -> io::Result<String> {
        let n = self.size()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn count(&mut self, min_bytes: usize) -> io::Result<usize> {
        let n = self.size()?;
        if n > self.rest.len() / min_bytes {
            return Err(invalid_model("GGUF item count exceeds the remaining file"));
        }
        Ok(n)
    }
    fn integer(&mut self, t: u32) -> io::Result<Value> {
        Ok(match t {
            0 => Value::U(self.u8()? as u64),
            1 => Value::I(self.u8()? as i8 as i64),
            2 => Value::U(self.u16()? as u64),
            3 => Value::I(self.u16()? as i16 as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            _ => return Err(invalid_model(format_args!("unsupported GGUF value type {t}"))),
        })
    }
    fn value(&mut self, t: u32) -> io::Result<Value> {
        Ok(match t {
            6 => Value::F(f32::from_bits(self.u32()?) as f64),
            7 => Value::B(self.u8()? != 0),
            8 => Value::S(self.str()?),
            9 => self.array()?,
            12 => Value::F(f64::from_bits(self.u64()?)),
            _ => self.integer(t)?,
        })
    }
    fn array(&mut self) -> io::Result<Value> {
        let et = self.u32()?;
        let min_bytes = match et {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4 | 5 | 6 => 4,
            8 | 10 | 11 | 12 => 8,
            _ => return Err(invalid_model(format_args!("unsupported GGUF array element type {et}"))),
        };
        let n = self.count(min_bytes)?;
        Ok(match et {
            6 => Value::AF((0..n).map(|_| self.u32().map(|b| f32::from_bits(b) as f64)).collect::<io::Result<_>>()?),
            7 => Value::AB((0..n).map(|_| self.u8().map(|b| b != 0)).collect::<io::Result<_>>()?),
            8 => Value::AS((0..n).map(|_| self.str()).collect::<io::Result<_>>()?),
            12 => Value::AF((0..n).map(|_| self.u64().map(f64::from_bits)).collect::<io::Result<_>>()?),
            _ => Value::AI((0..n).map(|_| match self.integer(et)? {
                Value::U(u) => Ok(u as i64),
                Value::I(i) => Ok(i),
                _ => unreachable!(),
            }).collect::<io::Result<_>>()?),
        })
    }
}

fn parse_tables(bytes: &[u8]) -> io::Result<(HashMap<String, Value>, HashMap<String, Tensor>)> {
    let mut r = Rd { rest: bytes };
    if r.take(4)? != b"GGUF" {
        return Err(invalid_model("expected GGUF magic"));
    }
    let ver = r.u32()?;
    if !matches!(ver, 2 | 3) {
        return Err(invalid_model(format_args!("unsupported GGUF version {ver}")));
    }
    let nt = r.count(32)?; // name length, rank, at least one dimension, type, offset
    let nkv = r.count(13)?; // key length, type, at least one byte of value
    let mut kv = HashMap::new();
    for _ in 0..nkv {
        let k = r.str()?;
        if k.is_empty() {
            return Err(invalid_model("empty GGUF metadata key"));
        }
        let ty = r.u32()?;
        let v = r.value(ty)?;
        if kv.insert(k, v).is_some() {
            return Err(invalid_model("duplicate GGUF metadata key"));
        }
    }
    // The Q4_K Metal kernels read 16-byte vectors directly from the mapping.
    let align = match kv.get("general.alignment") {
        Some(Value::U(a)) => usize::try_from(*a).ok().filter(|a| *a >= 16 && a.is_power_of_two())
            .ok_or_else(|| invalid_model("invalid GGUF alignment"))?,
        None => 32,
        _ => return Err(invalid_model("invalid GGUF alignment")),
    };
    let mut tensors = HashMap::new();
    for _ in 0..nt {
        let name = r.str()?;
        if name.is_empty() {
            return Err(invalid_model("empty GGUF tensor name"));
        }
        let nd = r.u32()?;
        if !(1..=4).contains(&nd) {
            return Err(invalid_model(format_args!("tensor {name} has invalid rank {nd}")));
        }
        let ne = (0..nd).map(|_| r.size()).collect::<io::Result<Vec<_>>>()?;
        let ty = r.u32()?;
        let off = r.size()?;
        let t = Tensor::from_decl(&name, ne, ty, off)?;
        if tensors.insert(name, t).is_some() {
            return Err(invalid_model("duplicate GGUF tensor name"));
        }
    }
    if !tensors.is_empty() {
        let data_off = (bytes.len() - r.rest.len()).checked_next_multiple_of(align)
            .ok_or_else(|| invalid_model("GGUF tensor data offset overflows"))?;
        for (name, t) in &mut tensors {
            t.locate(name, data_off, align, bytes.len())?;
        }
    }
    Ok((kv, tensors))
}

impl Gguf {
    pub fn open(path: &str) -> io::Result<Gguf> {
        let file = File::open(path)?;
        let file_len = usize::try_from(file.metadata()?.len()).ok()
            .filter(|&n| n >= 24 && n <= isize::MAX as usize)
            .ok_or_else(|| invalid_model("invalid GGUF file size"))?;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = usize::try_from(page).ok().filter(|&p| p != 0)
            .ok_or_else(|| io::Error::other("could not determine mmap page size"))?;
        let map_len = file_len.checked_next_multiple_of(page)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| invalid_model("GGUF mapping exceeds the address space"))?;
        let map = unsafe {
            libc::mmap(std::ptr::null_mut(), map_len, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0)
        };
        if map == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let Some(map_ptr) = NonNull::new(map.cast()) else {
            unsafe { libc::munmap(map, map_len) };
            return Err(io::Error::other("GGUF mapping returned a null address"));
        };
        // Own the mapping before parsing so it is unmapped on any error.
        let mut g = Gguf { _file: file, map: map_ptr, map_len, file_len, kv: HashMap::new(), tensors: HashMap::new() };
        (g.kv, g.tensors) = parse_tables(g.data())?;
        Ok(g)
    }

    pub fn ptr(&self) -> *mut u8 { self.map.as_ptr() }
    pub fn map_len(&self) -> usize { self.map_len }
    pub fn file_len(&self) -> usize { self.file_len }

    fn data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.map.as_ptr(), self.file_len) }
    }

    pub fn uint(&self, k: &str) -> io::Result<u64> {
        match self.kv.get(k) {
            Some(Value::U(u)) => Ok(*u),
            Some(Value::I(i)) if *i >= 0 => Ok(*i as u64),
            _ => Err(invalid_model(format_args!("{k} must be a nonnegative integer"))),
        }
    }
    pub fn u(&self, k: &str) -> u64 {
        match self.kv.get(k) {
            Some(Value::B(b)) => u64::from(*b),
            _ => self.uint(k).unwrap(),
        }
    }
    pub fn f(&self, k: &str) -> io::Result<f32> {
        let f = match self.kv.get(k) {
            Some(Value::F(f)) => *f as f32,
            Some(Value::I(i)) => *i as f32,
            _ => self.uint(k)? as f32,
        };
        if !f.is_finite() {
            return Err(invalid_model(format_args!("{k} must be finite")));
        }
        Ok(f)
    }
    pub fn tensor(&self, name: &str, shape: &[usize], types: &[u32]) -> io::Result<&Tensor> {
        let t = self.opt(name).ok_or_else(|| invalid_model(format_args!("missing tensor {name}")))?;
        t.check(name, shape, types)?;
        Ok(t)
    }
    pub fn opt(&self, name: &str) -> Option<&Tensor> { self.tensors.get(name) }
    pub fn bytes(&self, t: &Tensor) -> &[u8] {
        &self.data()[t.file_off..][..t.nbytes]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_rows_require_whole_blocks_and_checked_sizes() {
        let t = Tensor::from_decl("test", vec![256, 2], T_Q4_K, 0).unwrap();
        assert_eq!(t.row_bytes, 144);
        assert_eq!(t.nbytes, 288);
        assert!(t.check("test", &[256, 2], &[T_Q4_K]).is_ok());
        assert!(t.check("test", &[512, 1], &[T_Q4_K]).is_err());
        assert!(t.check("test", &[256, 2], &[T_Q6_K]).is_err());

        // Each row must be independently addressable even if the total fills a block.
        assert!(Tensor::from_decl("test", vec![128, 2], T_Q4_K, 0).is_err());
        assert!(Tensor::from_decl("test", vec![256, 0], T_Q4_K, 0).is_err());
        assert!(Tensor::from_decl("test", vec![usize::MAX, 2], T_F32, 0).is_err());
        assert!(Tensor::from_decl("test", vec![32, usize::MAX], T_Q8_0, 0).is_err());
    }

    #[test]
    fn tensor_extent_requires_alignment_and_must_fit_actual_file() {
        let mut t = Tensor::from_decl("test", vec![256], T_Q4_K, 32).unwrap();
        assert!(t.locate("test", 64, 32, 239).is_err());
        assert!(t.locate("test", 64, 32, 240).is_ok());
        assert_eq!(t.file_off, 96);

        let mut t = Tensor::from_decl("test", vec![256], T_Q4_K, 1).unwrap();
        assert!(t.locate("test", 64, 32, 240).is_err());

        let mut t = Tensor::from_decl("test", vec![256], T_Q4_K, usize::MAX - 31).unwrap();
        assert!(t.locate("test", 32, 32, usize::MAX).is_err());
        assert!(t.locate("test", 0, 32, usize::MAX).is_err());
    }

    #[test]
    fn parses_complete_gguf_and_rejects_truncation() {
        let mut file = b"GGUF".to_vec();
        file.extend(3u32.to_le_bytes());
        file.extend(1u64.to_le_bytes()); // tensors
        file.extend(0u64.to_le_bytes()); // metadata entries
        file.extend(4u64.to_le_bytes()); // tensor name bytes
        file.extend(b"test");
        file.extend(1u32.to_le_bytes()); // rank
        file.extend(256u64.to_le_bytes());
        file.extend(T_Q4_K.to_le_bytes());
        file.extend(0u64.to_le_bytes()); // offset relative to tensor data
        let offset = file.len().next_multiple_of(32);
        file.resize(offset + 144, 0);

        let (_, tensors) = parse_tables(&file).unwrap();
        assert_eq!(tensors["test"].file_off, offset);
        assert_eq!(tensors["test"].nbytes, 144);
        for end in [1, 16, 24, 50, file.len() - 1] {
            assert_eq!(parse_tables(&file[..end]).unwrap_err().kind(), ErrorKind::InvalidData);
        }
    }

    #[test]
    fn native_nv4_and_mx8_block_geometry() {
        let nv4 = Tensor::from_decl("gu", vec![2304, 2304, 128], T_NV4, 0).unwrap();
        assert_eq!(nv4.row_bytes, 2304 / 32 * 18);
        assert_eq!(nv4.nbytes, nv4.row_bytes * 2304 * 128);
        assert!(Tensor::from_decl("bad_nv4", vec![30], T_NV4, 0).is_err());

        let mx8 = Tensor::from_decl("dn", vec![1152, 2304, 128], T_MX8, 0).unwrap();
        assert_eq!(mx8.row_bytes, 1152 / 32 * 34);
        assert_eq!(mx8.nbytes, mx8.row_bytes * 2304 * 128);
        assert!(Tensor::from_decl("bad_mx8", vec![33], T_MX8, 0).is_err());
    }

    #[test]
    fn rejects_duplicate_or_empty_metadata_keys_and_tensor_names() {
        let mut dup_kv = b"GGUF".to_vec();
        dup_kv.extend(3u32.to_le_bytes());
        dup_kv.extend(0u64.to_le_bytes()); // tensors
        dup_kv.extend(2u64.to_le_bytes()); // metadata entries
        for _ in 0..2 {
            dup_kv.extend(16u64.to_le_bytes());
            dup_kv.extend(b"inkserve.experts");
            dup_kv.extend(8u32.to_le_bytes());
            dup_kv.extend(2u64.to_le_bytes());
            dup_kv.extend(b"ok");
        }
        assert!(parse_tables(&dup_kv).unwrap_err().to_string().contains("duplicate GGUF metadata key"));

        let mut dup_tensor = b"GGUF".to_vec();
        dup_tensor.extend(3u32.to_le_bytes());
        dup_tensor.extend(2u64.to_le_bytes()); // tensors
        dup_tensor.extend(0u64.to_le_bytes()); // metadata entries
        for idx in 0..2u64 {
            let name = b"blk.0.ffn_gate_up_exps.gscale";
            dup_tensor.extend((name.len() as u64).to_le_bytes());
            dup_tensor.extend(name);
            dup_tensor.extend(1u32.to_le_bytes());
            dup_tensor.extend(128u64.to_le_bytes());
            dup_tensor.extend(T_F32.to_le_bytes());
            dup_tensor.extend((idx * 512).to_le_bytes());
        }
        let off = dup_tensor.len().next_multiple_of(32);
        dup_tensor.resize(off + 1024, 0);
        assert!(parse_tables(&dup_tensor).unwrap_err().to_string().contains("duplicate GGUF tensor name"));
    }
}
