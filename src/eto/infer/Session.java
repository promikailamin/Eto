package eto.infer;

import eto.gguf.GgufTensor;

import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * One inference session with a KV cache. Create one per conversation.
 * The session itself is single-threaded; heavy matrix-vector products are
 * parallelized across rows with a shared fork-join pool.
 */
public final class Session {
    public final LlamaModel model;
    public final int maxCtx;

    private static final int THREADS = threadCount();
    private static final ForkJoinPool POOL = THREADS > 1 ? new ForkJoinPool(THREADS) : null;
    /** Rows below this are cheaper to run on the calling thread. */
    private static final long PARALLEL_WORK = 1L << 19;

    private final int dim;
    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final int valueDim;
    private final int ffnDim;
    private final int group;
    private final int kvStride;
    private final int vStride;
    private final int attnDim;

    private final float[][] kCache;
    private final float[][] vCache;
    private final float[] scores;

    private final float[] x;
    private final float[] h;
    private final float[] q;
    private final float[] k;
    private final float[] v;
    private final float[] attn;
    private final float[] tmp;
    private final float[] gate;
    private final float[] up;
    private final float[] hidden;
    private final float[] normed;
    private final float[] logits;

    private int pos;

    Session(LlamaModel model, int maxCtx) {
        if (maxCtx < 2) {
            throw new IllegalArgumentException("context must be at least 2");
        }
        this.model = model;
        this.maxCtx = maxCtx;
        LlamaModel.Config c = model.cfg;
        this.dim = c.dim;
        this.nHeads = c.nHeads;
        this.nKvHeads = c.nKvHeads;
        this.headDim = c.headDim;
        this.valueDim = c.valueDim;
        this.ffnDim = c.ffnDim;
        this.group = nHeads / nKvHeads;
        this.kvStride = nKvHeads * headDim;
        this.vStride = nKvHeads * valueDim;
        this.attnDim = nHeads * valueDim;

        this.kCache = new float[c.nLayers][maxCtx * kvStride];
        this.vCache = new float[c.nLayers][maxCtx * vStride];
        this.scores = new float[maxCtx];
        this.x = new float[dim];
        this.h = new float[dim];
        this.q = new float[nHeads * headDim];
        this.k = new float[kvStride];
        this.v = new float[vStride];
        this.attn = new float[attnDim];
        this.tmp = new float[Math.max(dim, ffnDim)];
        this.gate = new float[ffnDim];
        this.up = new float[ffnDim];
        this.hidden = new float[ffnDim];
        this.normed = new float[dim];
        this.logits = new float[c.vocabSize];
    }

    public int position() {
        return pos;
    }

    public int remaining() {
        return maxCtx - pos;
    }

    public void reset() {
        pos = 0;
    }

    /**
     * Runs one token through the model. When {@code needLogits} is false the
     * final projection is skipped (useful while prefilling a prompt).
     */
    public float[] forward(int tokenId, boolean needLogits) {
        if (pos >= maxCtx) {
            throw new IllegalStateException("context window is full");
        }
        model.tokenEmbd.rowInto(tokenId, x);
        for (int li = 0; li < model.layers.length; li++) {
            LlamaModel.Layer layer = model.layers[li];

            rmsNorm(x, layer.attnNormW, model.cfg.rmsEps, h);

            matvec(layer.wq, h, q);
            addBias(q, layer.qBias);
            matvec(layer.wk, h, k);
            addBias(k, layer.kBias);
            matvec(layer.wv, h, v);
            addBias(v, layer.vBias);

            rope(q, pos, nHeads, headDim);
            rope(k, pos, nKvHeads, headDim);

            float[] kc = kCache[li];
            float[] vc = vCache[li];
            System.arraycopy(k, 0, kc, pos * kvStride, kvStride);
            System.arraycopy(v, 0, vc, pos * vStride, vStride);

            attention(kc, vc, pos);

            matvec(layer.wo, attn, tmp);
            addBiasPrefix(tmp, layer.oBias, dim);
            addInPlace(x, tmp, dim);

            rmsNorm(x, layer.ffnNormW, model.cfg.rmsEps, h);
            matvec(layer.wGate, h, gate);
            matvec(layer.wUp, h, up);
            for (int i = 0; i < ffnDim; i++) {
                hidden[i] = silu(gate[i]) * up[i];
            }
            matvec(layer.wDown, hidden, tmp);
            addInPlace(x, tmp, dim);
        }
        pos++;
        if (!needLogits) {
            return null;
        }
        rmsNorm(x, model.outputNorm, model.cfg.rmsEps, normed);
        matvec(model.output, normed, logits);
        return logits;
    }

    private void attention(float[] kc, float[] vc, int at) {
        float invSqrt = (float) (1.0 / Math.sqrt(headDim));
        for (int hh = 0; hh < nHeads; hh++) {
            int kvh = hh / group;
            int qOff = hh * headDim;
            int kOff = kvh * headDim;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j <= at; j++) {
                float sc = dot(q, qOff, kc, j * kvStride + kOff, headDim) * invSqrt;
                scores[j] = sc;
                if (sc > max) {
                    max = sc;
                }
            }
            float sum = 0f;
            for (int j = 0; j <= at; j++) {
                float e = (float) Math.exp(scores[j] - max);
                scores[j] = e;
                sum += e;
            }
            float inv = 1f / sum;
            int outOff = hh * valueDim;
            java.util.Arrays.fill(attn, outOff, outOff + valueDim, 0f);
            int vOff = kvh * valueDim;
            for (int j = 0; j <= at; j++) {
                float p = scores[j] * inv;
                int base = j * vStride + vOff;
                for (int i = 0; i < valueDim; i++) {
                    attn[outOff + i] += p * vc[base + i];
                }
            }
        }
    }

    private void rope(float[] vec, int position, int heads, int headDimLocal) {
        int half = headDimLocal / 2;
        double p = position / (double) model.cfg.ropeScale;
        boolean neox = model.cfg.ropeNeox;
        for (int i = 0; i < half; i++) {
            double freq = 1.0 / Math.pow(model.cfg.ropeTheta, (2.0 * i) / headDimLocal);
            float ang = (float) (p * freq);
            float cs = (float) Math.cos(ang);
            float sn = (float) Math.sin(ang);
            for (int hh = 0; hh < heads; hh++) {
                int base = hh * headDimLocal;
                int i1 = neox ? base + i : base + 2 * i;
                int i2 = neox ? base + i + half : base + 2 * i + 1;
                float x1 = vec[i1];
                float x2 = vec[i2];
                vec[i1] = x1 * cs - x2 * sn;
                vec[i2] = x1 * sn + x2 * cs;
            }
        }
    }

    private static void rmsNorm(float[] in, float[] w, float eps, float[] out) {
        int n = in.length;
        float s = 0f;
        for (int i = 0; i < n; i++) {
            s += in[i] * in[i];
        }
        float r = (float) (1.0 / Math.sqrt(s / n + eps));
        for (int i = 0; i < n; i++) {
            out[i] = in[i] * r * w[i];
        }
    }

    private static int threadCount() {
        int n = Integer.getInteger("eto.threads", Runtime.getRuntime().availableProcessors());
        return Math.max(1, Math.min(n, 64));
    }

    private static void matvec(GgufTensor w, float[] in, float[] out) {
        int rows = (int) w.rows();
        if (POOL == null || rows < 2 || (long) rows * w.ne0() < PARALLEL_WORK) {
            for (int o = 0; o < rows; o++) {
                out[o] = w.dotRow(o, in);
            }
            return;
        }
        POOL.submit(() -> IntStream.range(0, rows).parallel()
                .forEach(o -> out[o] = w.dotRow(o, in))).join();
    }

    private static float dot(float[] a, int ao, float[] b, int bo, int n) {
        float s = 0f;
        for (int i = 0; i < n; i++) {
            s += a[ao + i] * b[bo + i];
        }
        return s;
    }

    private static void addBias(float[] y, float[] bias) {
        if (bias == null) {
            return;
        }
        for (int i = 0; i < bias.length; i++) {
            y[i] += bias[i];
        }
    }

    private static void addBiasPrefix(float[] y, float[] bias, int n) {
        if (bias == null) {
            return;
        }
        for (int i = 0; i < n && i < bias.length; i++) {
            y[i] += bias[i];
        }
    }

    private static void addInPlace(float[] a, float[] b, int n) {
        for (int i = 0; i < n; i++) {
            a[i] += b[i];
        }
    }

    private static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }
}
