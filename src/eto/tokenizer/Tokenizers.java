package eto.tokenizer;

import eto.gguf.GgufReader;

/** Chooses and constructs the tokenizer described by a GGUF file. */
public final class Tokenizers {
    private Tokenizers() {
    }

    public static Tokenizer fromGguf(GgufReader r) {
        String[] tokens = r.getStringArray("tokenizer.ggml.tokens");
        if (tokens == null || tokens.length == 0) {
            throw new IllegalArgumentException("GGUF has no tokenizer.ggml.tokens");
        }
        String model = r.getString("tokenizer.ggml.model", "llama");
        int[] types = r.getIntArray("tokenizer.ggml.token_type");
        int bos = r.getInt("tokenizer.ggml.bos_token_id", find(tokens, "<s>", 1));
        int eos = r.getInt("tokenizer.ggml.eos_token_id", find(tokens, "</s>", 2));
        boolean addBos = r.getBoolean("tokenizer.ggml.add_bos_token", !"gpt2".equals(model));

        if ("gpt2".equals(model)) {
            String[] merges = r.getStringArray("tokenizer.ggml.merges");
            String pre = r.getString("tokenizer.ggml.pre", null);
            return new Gpt2Tokenizer(tokens, types, merges, bos, eos, addBos, pre);
        }
        if ("llama".equals(model) || "spm".equals(model)) {
            float[] scores = r.getFloatArray("tokenizer.ggml.scores");
            String[] merges = r.getStringArray("tokenizer.ggml.merges");
            boolean addPrefix = r.getBoolean("tokenizer.ggml.add_prefix_space", true);
            return new SpmTokenizer(tokens, types, scores, merges, bos, eos, addBos, addPrefix);
        }
        throw new IllegalArgumentException("unsupported tokenizer.ggml.model = " + model);
    }

    private static int find(String[] tokens, String text, int fallback) {
        for (int i = 0; i < tokens.length; i++) {
            if (text.equals(tokens[i])) {
                return i;
            }
        }
        return fallback;
    }
}
