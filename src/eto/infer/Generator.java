package eto.infer;

import eto.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/** Shared autoregressive generation loop. */
public final class Generator {
    public record Params(float temperature, int topK, float topP, float repeatPenalty, long seed) {
        public static Params defaults() {
            return new Params(0.8f, 40, 0.95f, 1.1f, 42);
        }
    }

    public interface Sink {
        void token(int id, byte[] bytes);
    }

    public record Result(int[] ids, String text) {
    }

    private Generator() {
    }

    public static Result generate(
            LlamaModel model,
            int[] prompt,
            int maxNew,
            int maxCtx,
            Params params,
            Sink sink) {
        Tokenizer tok = model.tokenizer;
        int ctx = Math.min(maxCtx, model.cfg.ctxLen);
        int[] ids = prompt.length == 0 ? new int[] {tok.bosId()} : prompt.clone();
        if (ids.length >= ctx) {
            int[] trimmed = new int[ctx];
            trimmed[0] = tok.bosId();
            System.arraycopy(ids, ids.length - (ctx - 1), trimmed, 1, ctx - 1);
            ids = trimmed;
        }

        Session session = model.newSession(ctx);
        float[] logits = null;
        for (int i = 0; i < ids.length; i++) {
            logits = session.forward(ids[i], i == ids.length - 1);
        }

        int[] history = new int[ctx];
        System.arraycopy(ids, 0, history, 0, ids.length);
        int histLen = ids.length;
        List<Integer> generated = new ArrayList<>();
        Random rng = new Random(params.seed);

        for (int step = 0; step < maxNew && session.position() < ctx; step++) {
            int from = Math.max(0, histLen - 64);
            int[] recent = Arrays.copyOfRange(history, from, histLen);
            int next = Sampler.sample(
                    logits,
                    params.temperature,
                    params.topK,
                    params.topP,
                    params.repeatPenalty,
                    recent,
                    recent.length,
                    tok,
                    rng);
            if (next == tok.eosId()) {
                break;
            }
            generated.add(next);
            history[histLen++] = next;
            if (sink != null) {
                sink.token(next, tok.decodeToken(next));
            }
            if (session.position() >= ctx) {
                break;
            }
            logits = session.forward(next, true);
        }

        int[] out = new int[generated.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = generated.get(i);
        }
        return new Result(out, tok.decode(out));
    }
}
