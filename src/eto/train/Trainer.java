package eto.train;

import eto.nn.AdamW;
import eto.nn.ModelConfig;
import eto.nn.ModelIO;
import eto.nn.TensorParam;
import eto.nn.TinyLlama;
import eto.tokenizer.ByteTokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/** Training loop over random byte windows with AdamW and cosine decay. */
public final class Trainer {
    public record Options(
            Path data,
            Path out,
            String name,
            int steps,
            int batch,
            int ctx,
            int dim,
            int layers,
            int heads,
            int ffn,
            float lr,
            long seed,
            float std) {
    }

    private Trainer() {
    }

    public static void train(Options o) throws IOException {
        byte[] bytes = Files.readAllBytes(o.data);
        if (bytes.length < 16) {
            throw new IllegalArgumentException("training data is too small: " + o.data);
        }
        ByteTokenizer tok = ByteTokenizer.standard();
        int[] corpus = new int[bytes.length + 2];
        corpus[0] = tok.bosId();
        for (int i = 0; i < bytes.length; i++) {
            corpus[i + 1] = bytes[i] & 0xFF;
        }
        corpus[corpus.length - 1] = tok.eosId();
        if (corpus.length <= o.ctx + 1) {
            throw new IllegalArgumentException("training data must contain at least ctx+2 bytes; got "
                    + bytes.length + " bytes for ctx=" + o.ctx);
        }

        ModelConfig cfg = new ModelConfig(tok.vocabSize(), o.dim, o.layers, o.heads, o.ffn, o.ctx, 1e-5f, 10000f);
        TinyLlama model = new TinyLlama(cfg);
        Random rng = new Random(o.seed);
        model.initRandom(rng, o.std);
        AdamW adam = AdamW.defaults();
        List<TensorParam> params = model.params();

        System.out.println("data      : " + o.data + " (" + bytes.length + " bytes, " + corpus.length + " tokens)");
        System.out.println("model     : " + cfg);
        System.out.println("parameters: " + String.format("%,d", cfg.paramCount()));
        System.out.println("steps     : " + o.steps + "  batch=" + o.batch + "  lr=" + o.lr);
        System.out.println();

        long t0 = System.nanoTime();
        int window = o.ctx;
        int[] batchTokens = new int[window];
        int warmup = Math.min(20, Math.max(1, o.steps / 10));
        int reportEvery = Math.max(1, o.steps / 20);
        float lastLoss = 0f;

        for (int step = 0; step < o.steps; step++) {
            adam.zeroGrad(params);
            double batchLoss = 0.0;
            for (int b = 0; b < o.batch; b++) {
                int start = rng.nextInt(corpus.length - window);
                System.arraycopy(corpus, start, batchTokens, 0, window);
                batchLoss += model.lossAndBackward(batchTokens);
            }
            float invBatch = 1f / o.batch;
            for (TensorParam p : params) {
                float[] g = p.g();
                for (int i = 0; i < g.length; i++) {
                    g[i] *= invBatch;
                }
            }
            lastLoss = (float) (batchLoss / o.batch);

            float lr;
            if (step < warmup) {
                lr = o.lr * (step + 1) / (float) warmup;
            } else {
                float progress = (step - warmup) / (float) Math.max(1, o.steps - warmup);
                lr = o.lr * (0.1f + 0.9f * 0.5f * (1f + (float) Math.cos(Math.PI * progress)));
            }
            adam.step(params, lr);

            if (step == 0 || (step + 1) % reportEvery == 0 || step == o.steps - 1) {
                double secs = (System.nanoTime() - t0) / 1e9;
                System.out.printf("step %5d/%d  loss %.4f  lr %.6f  %.1fs%n",
                        step + 1, o.steps, lastLoss, lr, secs);
            }
        }

        ModelIO.saveGguf(model, tok, o.out, o.name);
        long size = Files.size(o.out);
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println();
        System.out.printf("saved %s (%.2f MiB, %.1fs, final loss %.4f)%n",
                o.out, size / 1048576.0, secs, lastLoss);
        System.out.println("run it with: ./runner.sh --model " + o.out);
    }
}
