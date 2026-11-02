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
        return generate(model, prompt, maxNew, maxCtx, params, sink, new String[0]);
    }

    /** True when {@code text} ends with a non-empty prefix of any stop string. */
    private static boolean endsWithStopPrefix(String text, String[] stops) {
        for (String stop : stops) {
            for (int n = Math.min(stop.length() - 1, text.length()); n > 0; n--) {
                if (text.endsWith(stop.substring(0, n))) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Same as the plain overload, but generation stops as soon as the output contains one of
     * {@code stops} (for example "\nuser:" for chat models that have no end-of-turn token).
     * The stop string itself is never passed to the sink or returned.
     */
    public static Result generate(
            LlamaModel model,
            int[] prompt,
            int maxNew,
            int maxCtx,
            Params params,
            Sink sink,
            String[] stops) {
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
        java.io.ByteArrayOutputStream pending = new java.io.ByteArrayOutputStream();
        java.io.ByteArrayOutputStream all = new java.io.ByteArrayOutputStream();
        boolean stopped = false;
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
            byte[] piece = tok.decodeToken(next);
            if (stops.length == 0) {
                if (sink != null) {
                    sink.token(next, piece);
                }
            } else {
                pending.write(piece, 0, piece.length);
                String text = new String(pending.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
                int cut = -1;
                for (String stop : stops) {
                    int at = text.indexOf(stop);
                    if (at >= 0 && (cut < 0 || at < cut)) {
                        cut = at;
                    }
                }
                if (cut >= 0) {
                    byte[] keep = text.substring(0, cut).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                    all.write(keep, 0, keep.length);
                    if (sink != null && keep.length > 0) {
                        sink.token(next, keep);
                    }
                    stopped = true;
                    break;
                }
                if (!endsWithStopPrefix(text, stops) && !text.endsWith("\uFFFD")) {
                    byte[] flush = pending.toByteArray();
                    all.write(flush, 0, flush.length);
                    if (sink != null) {
                        sink.token(next, flush);
                    }
                    pending.reset();
                }
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
        if (stops.length > 0) {
            if (!stopped && pending.size() > 0) {
                byte[] rest = pending.toByteArray();
                all.write(rest, 0, rest.length);
                if (sink != null) {
                    sink.token(-1, rest);
                }
            }
            return new Result(out, all.toString(java.nio.charset.StandardCharsets.UTF_8).strip());
        }
        return new Result(out, tok.decode(out));
    }
}
