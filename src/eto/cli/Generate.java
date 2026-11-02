package eto.cli;

import eto.infer.Generator;
import eto.infer.LlamaModel;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/** One-shot completion: ./generate.sh --model model.gguf --prompt "..." */
public final class Generate {
    private Generate() {
    }

    public static void main(String[] args) throws Exception {
        Args a = Args.parse(args);
        if (a.flag("help")) {
            usage();
            return;
        }
        Path path = Path.of(a.get("model", "models/eto-tiny.gguf"));
        String prompt = a.get("prompt", "");
        int tokens = a.getInt("tokens", 200);
        int ctx = a.getInt("ctx", 0);
        boolean noBos = a.flag("no-bos");
        float temp = a.flag("greedy") ? 0f : a.getFloat("temp", 0.8f);

        try (LlamaModel model = LlamaModel.open(path)) {
            Generator.Params p = new Generator.Params(
                    temp,
                    a.getInt("top-k", 40),
                    a.getFloat("top-p", 0.95f),
                    a.getFloat("repeat", model.tokenizer.vocabSize() <= 300 ? 1.0f : 1.1f),
                    a.getLong("seed", 42));
            if (ctx <= 0) {
                ctx = model.cfg.ctxLen;
            }
            System.out.print(prompt);
            int[] ids = model.tokenizer.encode(prompt, !noBos && model.tokenizer.defaultAddBos());
            Generator.Result result = Generator.generate(model, ids, tokens, ctx, p, (id, bytes) -> {
                System.out.write(bytes, 0, bytes.length);
                System.out.flush();
            });
            System.out.println();
            if (result.ids().length == 0) {
                System.err.println("[no tokens generated]");
            }
        }
    }

    private static void usage() {
        System.out.println("""
                Usage: ./generate.sh --model MODEL.gguf [options]

                  --prompt TEXT   prompt (default empty)
                  --tokens N      maximum new tokens (default 200)
                  --ctx N         context window (default model context)
                  --temp F        temperature, 0 = greedy (default 0.8)
                  --top-k N       top-k sampling (default 40)
                  --top-p F       nucleus sampling (default 0.95)
                  --repeat F      repetition penalty (default 1.1; 1.0 for byte-level models)
                  --seed N        RNG seed (default 42)
                  --greedy        shortcut for --temp 0
                  --no-bos        do not add BOS
                """);
    }
}
