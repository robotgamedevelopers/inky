#![no_main]

#[allow(dead_code)]
#[path = "../../src/gguf.rs"]
mod gguf;
#[allow(dead_code)]
#[path = "../../src/unicode_tables.rs"]
mod unicode_tables;
#[allow(dead_code)]
#[path = "../../src/tokenizer.rs"]
mod tokenizer;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(g) = gguf::Gguf::open_bytes(data) {
        let _ = g.uint("general.alignment");
        let _ = g.uint_or_bool("tokenizer.ggml.eos_token_id");
        let _ = g.f("inkling2_loop.final_logit_softcapping");
        if let Some(t) = g.opt("test") {
            let _ = g.bytes(t);
        }
        if let Ok(tk) = tokenizer::Tokenizer::try_load(&g) {
            let ids = tk.encode("Hello <|message_user|> world 12345!\n\té", true);
            for id in ids {
                let _ = tk.decode_bytes(id);
            }
        }
    }
});
