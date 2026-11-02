package eto.infer;

import eto.tokenizer.Tokenizer;

import java.util.Random;

/** Temperature / top-k / top-p / repetition-penalty sampler. */
public final class Sampler {
    private Sampler() {
    }

    public static int sample(
            float[] logits,
            float temperature,
            int topK,
            float topP,
            float repeatPenalty,
            int[] recent,
            int recentCount,
            Tokenizer tokenizer,
            Random rng) {
        // Never emit BOS or other control tokens except EOS.
        int eos = tokenizer.eosId();
        for (int id : tokenizer.specialIds()) {
            if (id != eos && id >= 0 && id < logits.length) {
                logits[id] = Float.NEGATIVE_INFINITY;
            }
        }
        if (repeatPenalty > 1.0f && recent != null) {
            for (int i = 0; i < recentCount; i++) {
                int id = recent[i];
                if (id < 0 || id >= logits.length || Float.isInfinite(logits[id])) {
                    continue;
                }
                if (logits[id] > 0f) {
                    logits[id] /= repeatPenalty;
                } else {
                    logits[id] *= repeatPenalty;
                }
            }
        }

        if (temperature <= 0f) {
            int best = -1;
            float bestVal = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < logits.length; i++) {
                if (logits[i] > bestVal) {
                    bestVal = logits[i];
                    best = i;
                }
            }
            if (best < 0) {
                throw new IllegalStateException("no tokens available to sample");
            }
            return best;
        }

        int v = logits.length;
        int k = topK > 0 ? Math.min(topK, v) : v;
        int[] idx = new int[k];
        float[] val = new float[k];
        java.util.Arrays.fill(val, Float.NEGATIVE_INFINITY);
        int used = 0;
        for (int i = 0; i < v; i++) {
            float l = logits[i];
            if (Float.isNaN(l) || l == Float.NEGATIVE_INFINITY) {
                continue;
            }
            if (used < k) {
                int p = used++;
                while (p > 0 && val[p - 1] < l) {
                    val[p] = val[p - 1];
                    idx[p] = idx[p - 1];
                    p--;
                }
                val[p] = l;
                idx[p] = i;
            } else if (l > val[k - 1]) {
                int p = k - 1;
                while (p > 0 && val[p - 1] < l) {
                    val[p] = val[p - 1];
                    idx[p] = idx[p - 1];
                    p--;
                }
                val[p] = l;
                idx[p] = i;
            }
        }
        if (used == 0) {
            throw new IllegalStateException("all token probabilities are zero");
        }

        float max = val[0];
        double sum = 0.0;
        double[] probs = new double[used];
        for (int i = 0; i < used; i++) {
            double p = Math.exp((val[i] - max) / temperature);
            probs[i] = p;
            sum += p;
        }
        int keep = used;
        if (topP > 0f && topP < 1f) {
            double cumulative = 0.0;
            for (int i = 0; i < used; i++) {
                cumulative += probs[i] / sum;
                if (cumulative >= topP) {
                    keep = i + 1;
                    break;
                }
            }
        }
        double total = 0.0;
        for (int i = 0; i < keep; i++) {
            total += probs[i];
        }
        double r = rng.nextDouble() * total;
        double accum = 0.0;
        for (int i = 0; i < keep; i++) {
            accum += probs[i];
            if (r <= accum) {
                return idx[i];
            }
        }
        return idx[keep - 1];
    }
}
