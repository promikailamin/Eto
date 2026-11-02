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
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
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
            float std,
            Path checkpoint,
            Path resume,
            int checkpointEvery) {
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
        model.initRandom(new Random(o.seed), o.std);
        AdamW adam = AdamW.defaults();
        List<TensorParam> params = model.params();
        byte[] dataHash = sha256(bytes);
        int startStep = 0;
        if (o.resume != null) {
            startStep = TrainingCheckpoint.load(o.resume, model, o.batch, o.seed, dataHash, params);
            adam.restoreStepCount(startStep);
        }
        if (o.checkpoint != null && o.checkpointEvery <= 0) {
            throw new IllegalArgumentException("checkpoint interval must be positive");
        }

        System.out.println("data      : " + o.data + " (" + bytes.length + " bytes, " + corpus.length + " tokens)");
        System.out.println("model     : " + cfg);
        System.out.println("parameters: " + String.format("%,d", cfg.paramCount()));
        System.out.println("steps     : " + o.steps + "  batch=" + o.batch + "  lr=" + o.lr);
        if (o.resume != null) {
            System.out.println("resumed   : " + o.resume + " (optimizer step " + startStep + ")");
        }
        System.out.println();

        // Chat turns start with "user:" at the beginning of a line. At inference the prompt
        // always starts with BOS followed by such a turn, so half of the training windows
        // are aligned to a turn start (BOS + turn) instead of a random byte offset.
        int[] turnStarts = findTurnStarts(corpus);
        System.out.println("turn starts: " + turnStarts.length);

        long t0 = System.nanoTime();
        int window = o.ctx;
        int[] batchTokens = new int[window];
        int warmup = Math.min(20, Math.max(1, o.steps / 10));
        int reportEvery = Math.max(1, o.steps / 20);
        float lastLoss = 0f;

        for (int step = 0; step < o.steps; step++) {
            int globalStep = Math.addExact(startStep, step);
            Random rng = new Random(o.seed ^ (0x9E3779B97F4A7C15L * (globalStep + 1L)));
            adam.zeroGrad(params);
            double batchLoss = 0.0;
            for (int b = 0; b < o.batch; b++) {
                if (turnStarts.length > 0 && rng.nextBoolean()) {
                    int start = turnStarts[rng.nextInt(turnStarts.length)];
                    batchTokens[0] = tok.bosId();
                    int n = Math.min(window - 1, corpus.length - start);
                    System.arraycopy(corpus, start, batchTokens, 1, n);
                    for (int i = n + 1; i < window; i++) {
                        batchTokens[i] = tok.eosId();
                    }
                } else {
                    int start = rng.nextInt(corpus.length - window);
                    System.arraycopy(corpus, start, batchTokens, 0, window);
                }
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
            if (o.checkpoint != null && ((step + 1) % o.checkpointEvery == 0 || step == o.steps - 1)) {
                TrainingCheckpoint.save(o.checkpoint, model, o.batch, o.seed, dataHash,
                        adam.stepCount(), params);
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

    /** Indices (into the token array) where a line starts with "user:". */
    private static int[] findTurnStarts(int[] corpus) {
        byte[] key = "user:".getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        int[] out = new int[1024];
        int count = 0;
        for (int i = 1; i + key.length < corpus.length - 1; i++) {
            if (i > 1 && corpus[i - 1] != '\n') {
                continue;
            }
            boolean match = true;
            for (int k = 0; k < key.length; k++) {
                if (corpus[i + k] != key[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                if (count == out.length) {
                    out = java.util.Arrays.copyOf(out, count * 2);
                }
                out[count++] = i;
            }
        }
        return java.util.Arrays.copyOf(out, count);
    }

    private static byte[] sha256(byte[] bytes) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(bytes);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }
}
