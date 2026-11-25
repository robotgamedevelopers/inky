// Reference oracle: run llama.cpp on a prompt one token at a time (greedy), dump token ids + full logits.
// usage: ref_logits model.gguf "prompt" n_gen out_prefix [batch=0|1]
#include "llama.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool parse_nonneg_int(const char * s, int * out) {
    if (!s || !*s) return false;
    errno = 0;
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || !end || *end != '\0' || v < 0 || v > INT_MAX) return false;
    *out = static_cast<int>(v);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 6) {
        std::fprintf(stderr, "usage: %s model prompt n_gen out_prefix [batch]\n", argv[0]);
        return 2;
    }
    std::string prompt = argv[2];
    int n_gen = 0;
    if (!parse_nonneg_int(argv[3], &n_gen)) {
        std::fprintf(stderr, "error: invalid n_gen %s (must be nonnegative integer)\n", argv[3]);
        return 2;
    }
    std::string out = argv[4];
    if (out.empty()) {
        std::fprintf(stderr, "error: out_prefix must not be empty\n");
        return 2;
    }
    int batch_arg = 0;
    if (argc > 5 && (!parse_nonneg_int(argv[5], &batch_arg) || batch_arg > 1)) {
        std::fprintf(stderr, "error: batch must be 0 or 1\n");
        return 2;
    }
    bool batch_mode = batch_arg != 0;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "error: failed to load model from %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
        std::fprintf(stderr, "error: model %s has no vocabulary\n", argv[1]);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (prompt.size() > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "error: prompt exceeds maximum supported byte length\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    int raw_n = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (raw_n == INT_MIN) {
        std::fprintf(stderr, "error: tokenization overflowed\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    int n = raw_n < 0 ? -raw_n : raw_n;
    if (n <= 0) {
        std::fprintf(stderr, "error: prompt tokenized to %d tokens\n", n);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    if (n > INT_MAX - n_gen) {
        std::fprintf(stderr, "error: prompt + n_gen overflows\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::vector<llama_token> toks(n);
    int tok_rc = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), toks.data(), n, true, true);
    if (tok_rc != n) {
        std::fprintf(stderr, "error: llama_tokenize returned %d, expected %d\n", tok_rc, n);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(std::max(4096, n + n_gen + 64));
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    if (std::getenv("REF_NO_FA")) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to initialize llama_context (n_ctx=%u)\n", cp.n_ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const int nv = llama_vocab_n_tokens(vocab);
    if (nv <= 0) {
        std::fprintf(stderr, "error: invalid vocabulary size %d\n", nv);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::string logits_path = out + ".logits";
    std::string tokens_path = out + ".tokens";
    FILE * fl = std::fopen(logits_path.c_str(), "wb");
    FILE * ft = std::fopen(tokens_path.c_str(), "w");
    if (!fl || !ft) {
        std::fprintf(stderr, "error: failed to open output files for prefix %s\n", out.c_str());
        if (fl) std::fclose(fl);
        if (ft) std::fclose(ft);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    if (!batch) {
        std::fprintf(stderr, "error: llama_batch_ext_init returned null\n");
        std::fclose(fl);
        std::fclose(ft);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    auto cleanup = [&](int code) {
        llama_batch_ext_free(batch);
        std::fclose(fl);
        std::fclose(ft);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return code;
    };

    int pos = 0;
    auto run = [&](const llama_token * t, int cnt) -> bool {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < cnt; i++) {
            int idx = llama_batch_ext_add_token(batch, 0, t[i]);
            llama_pos p = pos + i;
            llama_batch_ext_set_pos(batch, idx, &p);
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            std::fprintf(stderr, "error: llama_process decode failed at pos %d\n", pos);
            return false;
        }
        for (int i = 0; i < cnt; i++) {
            const float * row = llama_get_logits_ith(ctx, i);
            if (!row) {
                std::fprintf(stderr, "error: null logits row at batch index %d\n", i);
                return false;
            }
            if (std::fwrite(row, sizeof(float), static_cast<size_t>(nv), fl) != static_cast<size_t>(nv)) {
                std::fprintf(stderr, "error: short write to %s\n", logits_path.c_str());
                return false;
            }
            if (std::fprintf(ft, "%d\n", t[i]) <= 0) {
                std::fprintf(stderr, "error: failed writing token to %s\n", tokens_path.c_str());
                return false;
            }
        }
        pos += cnt;
        return true;
    };

    if (batch_mode) {
        for (int i = 0; i < n; i += 512) {
            if (!run(&toks[i], std::min(512, n - i))) return cleanup(1);
        }
    } else {
        for (int i = 0; i < n; i++) {
            if (!run(&toks[i], 1)) return cleanup(1);
        }
    }

    llama_token cur = 0;
    for (int g = 0; g < n_gen; g++) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (!lg) {
            std::fprintf(stderr, "error: null logits at generation step %d\n", g);
            return cleanup(1);
        }
        int best = 0;
        for (int v = 1; v < nv; v++) {
            if (lg[v] > lg[best]) best = v;
        }
        cur = best;
        char buf[256];
        int k = llama_token_to_piece(vocab, cur, buf, sizeof buf, 0, true);
        if (k > 0) {
            std::fwrite(buf, 1, std::min<size_t>(static_cast<size_t>(k), sizeof buf), stderr);
        }
        if (!run(&cur, 1)) return cleanup(1);
    }

    if (std::fflush(fl) != 0 || std::fflush(ft) != 0 || std::ferror(fl) || std::ferror(ft)) {
        std::fprintf(stderr, "error: output flush failed for prefix %s\n", out.c_str());
        return cleanup(1);
    }
    std::fprintf(stderr, "\n[ref] prompt tokens %d, generated %d, vocab %d\n", n, n_gen, nv);
    return cleanup(0);
}
