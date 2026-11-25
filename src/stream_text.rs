/// Decode complete UTF-8 sequences and replace malformed bytes, retaining only an
/// incomplete trailing sequence until more bytes arrive or the segment ends.
pub fn stream_text(pend: &mut Vec<u8>, end: bool, emit: impl FnOnce(&str)) {
    let mut consumed = 0;
    while consumed < pend.len() {
        match std::str::from_utf8(&pend[consumed..]) {
            Ok(_) => {
                consumed = pend.len();
                break;
            }
            Err(e) => {
                consumed += e.valid_up_to();
                match e.error_len() {
                    Some(n) => consumed += n,
                    None if end => {
                        consumed = pend.len();
                        break;
                    }
                    None => break,
                }
            }
        }
    }
    if consumed != 0 {
        emit(&String::from_utf8_lossy(&pend[..consumed]));
        pend.drain(..consumed);
    }
}
