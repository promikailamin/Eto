package eto.nn;

import eto.gguf.GgufWriter;
import eto.tokenizer.ByteTokenizer;

import java.io.IOException;
import java.nio.file.Path;

/** Exports a trained {@link TinyLlama} as a standard llama-architecture GGUF. */
public final class ModelIO {
    private ModelIO() {
    }

    public static void saveGguf(TinyLlama model, ByteTokenizer tok, Path path) throws IOException {
        saveGguf(model, tok, path, "eto-tiny");
    }

    public static void saveGguf(TinyLlama model, ByteTokenizer tok, Path path, String name) throws IOException {
        ModelConfig cfg = model.cfg;
        GgufWriter w = new GgufWriter(path);
        w.alignment(32);
        w.add("general.architecture", "llama");
        w.add("general.name", name);
        w.add("general.file_type", 0);
        w.add("general.alignment", 32);

        w.add("llama.context_length", cfg.ctxLen);
        w.add("llama.embedding_length", cfg.dim);
        w.add("llama.block_count", cfg.nLayers);
        w.add("llama.feed_forward_length", cfg.ffnDim);
        w.add("llama.attention.head_count", cfg.nHeads);
        w.add("llama.attention.head_count_kv", cfg.nHeads);
        w.add("llama.attention.layer_norm_rms_epsilon", cfg.rmsEps);
        w.add("llama.rope.dimension_count", cfg.headDim);
        w.add("llama.rope.freq_base", cfg.ropeTheta);
        w.add("llama.vocab_size", cfg.vocabSize);

        w.add("tokenizer.ggml.model", "gpt2");
        w.add("tokenizer.ggml.tokens", tok.ggufTokens());
        w.add("tokenizer.ggml.token_type", tok.ggufTokenTypes());
        w.add("tokenizer.ggml.merges", new String[0]);
        w.add("tokenizer.ggml.bos_token_id", tok.bosId());
        w.add("tokenizer.ggml.eos_token_id", tok.eosId());
        w.add("tokenizer.ggml.padding_token_id", tok.eosId());
        w.add("tokenizer.ggml.add_bos_token", true);
        w.add("tokenizer.ggml.add_eos_token", false);
        w.add("tokenizer.ggml.add_prefix_space", false);

        for (TensorParam p : model.params()) {
            long[] dims = p.isVector()
                    ? new long[] {p.cols()}
                    : new long[] {p.cols(), p.rows()};
            float[] data = p.w();
            // GGUF's llama architecture stores q/k rows in the interleaved
            // layout used by llama.cpp's "normal" RoPE. The in-memory model
            // uses the Hugging Face half-split layout, so permute on export.
            if (p.name().endsWith("attn_q.weight") || p.name().endsWith("attn_k.weight")) {
                data = permuteRopeRows(data, p.rows(), p.cols(), cfg.headDim);
            }
            w.addTensor(p.name(), dims, data);
        }
        w.write();
    }

    /**
     * Converts [head][i] + [head][i + half] row pairs into adjacent
     * [head][2i] + [head][2i + 1] pairs, matching llama.cpp's RoPE layout.
     */
    private static float[] permuteRopeRows(float[] src, int rows, int cols, int headDim) {
        float[] dst = new float[src.length];
        int half = headDim / 2;
        for (int h = 0; h < rows / headDim; h++) {
            int base = h * headDim;
            for (int i = 0; i < half; i++) {
                System.arraycopy(src, (base + i) * cols, dst, (base + 2 * i) * cols, cols);
                System.arraycopy(src, (base + i + half) * cols, dst, (base + 2 * i + 1) * cols, cols);
            }
        }
        return dst;
    }
}
