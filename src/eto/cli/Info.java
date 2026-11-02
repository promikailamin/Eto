package eto.cli;

import eto.gguf.GgufReader;
import eto.gguf.Quant;
import eto.infer.LlamaModel;
import eto.tokenizer.Tokenizer;
import eto.tokenizer.Tokenizers;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Prints GGUF metadata, tokenizer information and tensor types. */
public final class Info {
    private Info() {
    }

    public static void main(String[] args) throws Exception {
        Args a = Args.parse(args);
        if (a.flag("help") || !a.has("model")) {
            System.out.println("Usage: ./eto info --model MODEL.gguf");
            return;
        }
        Path path = Path.of(a.get("model"));
        try (GgufReader r = GgufReader.open(path)) {
            System.out.println("file      : " + path);
            System.out.println("size      : " + String.format("%,d", Files.size(path)) + " bytes");
            System.out.println("gguf      : v" + r.version());
            System.out.println("tensors   : " + r.tensors().size());
            System.out.println("metadata  :");
            for (Map.Entry<String, Object> e : r.metadata().entrySet()) {
                String key = e.getKey();
                if (key.equals("tokenizer.ggml.tokens") || key.equals("tokenizer.ggml.merges")
                        || key.equals("tokenizer.ggml.token_type") || key.equals("tokenizer.ggml.scores")) {
                    Object v = e.getValue();
                    int n = v instanceof Object[] arr ? arr.length : v instanceof int[] arr ? arr.length
                            : v instanceof float[] arr ? arr.length : -1;
                    System.out.println("  " + key + " = <array length " + n + ">");
                    continue;
                }
                System.out.println("  " + key + " = " + summary(e.getValue()));
            }

            try {
                Tokenizer tok = Tokenizers.fromGguf(r);
                System.out.println("tokenizer : " + tok.family() + ", vocab=" + tok.vocabSize()
                        + ", bos=" + tok.bosId() + ", eos=" + tok.eosId()
                        + ", addBos=" + tok.defaultAddBos());
            } catch (RuntimeException e) {
                System.out.println("tokenizer : unsupported (" + e.getMessage() + ")");
            }

            long params = 0;
            System.out.println("tensor list:");
            for (GgufReader.TensorInfo ti : r.tensors().values()) {
                params += ti.count();
                String flag = Quant.supported(ti.type) ? "" : "  [UNSUPPORTED]";
                System.out.printf("  %-34s %-6s %s  %d values%s%n",
                        ti.name, Quant.name(ti.type), java.util.Arrays.toString(ti.dims), ti.count(), flag);
            }
            System.out.println("parameters: " + String.format("%,d", params));

            try (LlamaModel m = LlamaModel.open(path)) {
                System.out.println("config    : " + m.cfg);
                System.out.println("output    : " + (m.cfg.tiedOutput ? "tied embeddings" : "separate output.weight"));
            } catch (RuntimeException e) {
                System.out.println("inference : not loadable: " + e.getMessage());
            }
        }
    }

    private static String summary(Object v) {
        if (v instanceof Object[] a) {
            return "<array length " + a.length + ">";
        }
        if (v instanceof int[] a) {
            return "<array length " + a.length + ">";
        }
        if (v instanceof long[] a) {
            return "<array length " + a.length + ">";
        }
        if (v instanceof float[] a) {
            return "<array length " + a.length + ">";
        }
        if (v instanceof double[] a) {
            return "<array length " + a.length + ">";
        }
        if (v instanceof boolean[] a) {
            return "<array length " + a.length + ">";
        }
        if (v instanceof String s && s.length() > 120) {
            return s.substring(0, 117) + "...";
        }
        return String.valueOf(v);
    }
}
