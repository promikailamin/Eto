package eto.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A small llama-style decoder used for training and gradient checks.
 *
 * Its GGUF export uses the conventional llama tensor names, so the generic
 * inference engine can load it exactly like a third-party llama model.
 */
public final class TinyLlama {
    public final ModelConfig cfg;
    public final Mat tokenEmbd;
    public final Vec outputNorm;
    public final Mat output;
    public final Block[] blocks;

    public TinyLlama(ModelConfig cfg) {
        this.cfg = cfg;
        int d = cfg.dim;
        tokenEmbd = new Mat("token_embd.weight", cfg.vocabSize, d);
        outputNorm = new Vec("output_norm.weight", d);
        output = new Mat("output.weight", cfg.vocabSize, d);
        blocks = new Block[cfg.nLayers];
        for (int i = 0; i < cfg.nLayers; i++) {
            blocks[i] = new Block("blk." + i + ".", cfg);
        }
    }

    public void initRandom(Random rng, float std) {
        Init.normal(rng, tokenEmbd.w, std);
        Init.normal(rng, output.w, std);
        Init.fill(outputNorm.w, 1f);
        float stdResidual = (float) (std / Math.sqrt(2.0 * cfg.nLayers));
        for (Block b : blocks) {
            b.init(rng, std, stdResidual);
        }
    }

    public List<TensorParam> params() {
        List<TensorParam> out = new ArrayList<>();
        out.add(tokenEmbd);
        out.add(outputNorm);
        out.add(output);
        for (Block b : blocks) {
            b.addParams(out);
        }
        return out;
    }

    public void zeroGrad() {
        for (TensorParam p : params()) {
            java.util.Arrays.fill(p.g(), 0f);
        }
    }

    // ---------------------------------------------------------------- forward

    public static final class Cache {
        public final int T;
        public final int[] tokens;
        public final float[][] xFinal;
        public final float[][] normed;
        public final BlockCache[] blocks;

        Cache(int t, int[] tokens, float[][] xFinal, float[][] normed, BlockCache[] blocks) {
            this.T = t;
            this.tokens = tokens;
            this.xFinal = xFinal;
            this.normed = normed;
            this.blocks = blocks;
        }
    }

    public static final class BlockCache {
        public int T;
        public float[][] input;
        public float[][] ln1;
        public float[][] q;
        public float[][] k;
        public float[][] v;
        public float[][][] probs;
        public float[][] attn;
        public float[][] xMid;
        public float[][] ln2;
        public float[][] gate;
        public float[][] up;
        public float[][] hidden;
        public float[][] out;
    }

    public Cache forward(int[] tokens) {
        int t = tokens.length;
        if (t < 2 || t > cfg.ctxLen) {
            throw new IllegalArgumentException("sequence length " + t + " outside [2," + cfg.ctxLen + "]");
        }
        int d = cfg.dim;
        float[][] x = new float[t][d];
        for (int i = 0; i < t; i++) {
            System.arraycopy(tokenEmbd.w, tokens[i] * d, x[i], 0, d);
        }
        BlockCache[] bcs = new BlockCache[blocks.length];
        for (int li = 0; li < blocks.length; li++) {
            bcs[li] = blockForward(blocks[li], x);
            x = bcs[li].out;
        }
        float[][] normed = Ops.rmsNorm(x, outputNorm.w, cfg.rmsEps);
        return new Cache(t, tokens, x, normed, bcs);
    }

    private BlockCache blockForward(Block b, float[][] x) {
        int t = x.length;
        int d = cfg.dim;
        int f = cfg.ffnDim;
        BlockCache c = new BlockCache();
        c.T = t;
        c.input = x;
        c.ln1 = Ops.rmsNorm(x, b.attnNorm.w, cfg.rmsEps);
        c.q = Ops.linear(b.wq, c.ln1);
        c.k = Ops.linear(b.wk, c.ln1);
        c.v = Ops.linear(b.wv, c.ln1);
        Ops.rope(c.q, cfg.nHeads, cfg.headDim, cfg.ropeTheta);
        Ops.rope(c.k, cfg.nHeads, cfg.headDim, cfg.ropeTheta);

        int heads = cfg.nHeads;
        int hd = cfg.headDim;
        float scale = (float) (1.0 / Math.sqrt(hd));
        c.probs = new float[heads][t][t];
        c.attn = new float[t][d];
        for (int h = 0; h < heads; h++) {
            int off = h * hd;
            for (int ti = 0; ti < t; ti++) {
                float[] pt = c.probs[h][ti];
                float max = Float.NEGATIVE_INFINITY;
                for (int s = 0; s <= ti; s++) {
                    float sc = Ops.dot(c.q[ti], off, c.k[s], off, hd) * scale;
                    pt[s] = sc;
                    if (sc > max) {
                        max = sc;
                    }
                }
                float sum = 0f;
                for (int s = 0; s <= ti; s++) {
                    float e = (float) Math.exp(pt[s] - max);
                    pt[s] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                for (int s = 0; s <= ti; s++) {
                    pt[s] *= inv;
                }
                for (int i = 0; i < hd; i++) {
                    float acc = 0f;
                    for (int s = 0; s <= ti; s++) {
                        acc += pt[s] * c.v[s][off + i];
                    }
                    c.attn[ti][off + i] = acc;
                }
            }
        }

        c.xMid = Ops.add(x, Ops.linear(b.wo, c.attn));
        c.ln2 = Ops.rmsNorm(c.xMid, b.ffnNorm.w, cfg.rmsEps);
        c.gate = Ops.linear(b.wGate, c.ln2);
        c.up = Ops.linear(b.wUp, c.ln2);
        c.hidden = new float[t][f];
        for (int ti = 0; ti < t; ti++) {
            for (int j = 0; j < f; j++) {
                c.hidden[ti][j] = Ops.silu(c.gate[ti][j]) * c.up[ti][j];
            }
        }
        c.out = Ops.add(c.xMid, Ops.linear(b.wDown, c.hidden));
        return c;
    }

    public float[][] allLogits(Cache c) {
        int t = c.T;
        int d = cfg.dim;
        int v = cfg.vocabSize;
        float[][] logits = new float[t][v];
        float[] w = output.w;
        for (int ti = 0; ti < t; ti++) {
            float[] n = c.normed[ti];
            float[] row = logits[ti];
            for (int vi = 0; vi < v; vi++) {
                row[vi] = Ops.dot(w, vi * d, n, 0, d);
            }
        }
        return logits;
    }

    public float[] lastLogits(int[] tokens) {
        Cache c = forward(tokens);
        int ti = c.T - 1;
        int d = cfg.dim;
        int v = cfg.vocabSize;
        float[] logits = new float[v];
        float[] n = c.normed[ti];
        for (int vi = 0; vi < v; vi++) {
            logits[vi] = Ops.dot(output.w, vi * d, n, 0, d);
        }
        return logits;
    }

    // ------------------------------------------------------------------- loss

    public float loss(int[] tokens) {
        Cache c = forward(tokens);
        float[][] logits = allLogits(c);
        int n = c.T - 1;
        int v = cfg.vocabSize;
        double total = 0.0;
        for (int ti = 0; ti < n; ti++) {
            float[] row = logits[ti];
            int target = tokens[ti + 1];
            float max = Float.NEGATIVE_INFINITY;
            for (int vi = 0; vi < v; vi++) {
                if (row[vi] > max) {
                    max = row[vi];
                }
            }
            double sum = 0.0;
            for (int vi = 0; vi < v; vi++) {
                sum += Math.exp(row[vi] - max);
            }
            total += max + Math.log(sum) - row[target];
        }
        return (float) (total / n);
    }

    /**
     * Computes the mean next-token cross entropy and accumulates gradients
     * into every parameter. Gradients are not zeroed by this method.
     */
    public float lossAndBackward(int[] tokens) {
        Cache c = forward(tokens);
        int t = c.T;
        int d = cfg.dim;
        int v = cfg.vocabSize;
        float[][] logits = allLogits(c);
        int n = t - 1;
        double total = 0.0;
        float inv = 1f / n;
        for (int ti = 0; ti < n; ti++) {
            float[] row = logits[ti];
            int target = tokens[ti + 1];
            float max = Float.NEGATIVE_INFINITY;
            for (int vi = 0; vi < v; vi++) {
                if (row[vi] > max) {
                    max = row[vi];
                }
            }
            double sum = 0.0;
            for (int vi = 0; vi < v; vi++) {
                sum += Math.exp(row[vi] - max);
            }
            total += max + Math.log(sum) - row[target];
            float invSum = (float) (1.0 / sum);
            for (int vi = 0; vi < v; vi++) {
                row[vi] = (float) (Math.exp(row[vi] - max) * invSum);
            }
            row[target] -= 1f;
            for (int vi = 0; vi < v; vi++) {
                row[vi] *= inv;
            }
        }
        // The last position has no next-token target; allLogits left it zero.
        java.util.Arrays.fill(logits[t - 1], 0f);

        float[][] dNormed = new float[t][d];
        float[] ow = output.w;
        float[] og = output.g;
        for (int ti = 0; ti < t; ti++) {
            float[] dl = logits[ti];
            float[] normed = c.normed[ti];
            for (int vi = 0; vi < v; vi++) {
                float g = dl[vi];
                if (g == 0f) {
                    continue;
                }
                int base = vi * d;
                for (int i = 0; i < d; i++) {
                    og[base + i] += g * normed[i];
                    dNormed[ti][i] += g * ow[base + i];
                }
            }
        }

        float[][] dx = Ops.rmsNormBackward(c.xFinal, outputNorm, dNormed, cfg.rmsEps);
        for (int li = blocks.length - 1; li >= 0; li--) {
            dx = blockBackward(blocks[li], c.blocks[li], dx);
        }
        for (int ti = 0; ti < t; ti++) {
            int base = tokens[ti] * d;
            for (int i = 0; i < d; i++) {
                tokenEmbd.g[base + i] += dx[ti][i];
            }
        }
        return (float) (total / n);
    }

    private float[][] blockBackward(Block b, BlockCache c, float[][] dOut) {
        int t = c.T;
        int d = cfg.dim;
        int f = cfg.ffnDim;

        // SwiGLU feed-forward
        float[][] dHidden = Ops.linearBackward(b.wDown, c.hidden, dOut);
        float[][] dGate = new float[t][f];
        float[][] dUp = new float[t][f];
        for (int ti = 0; ti < t; ti++) {
            for (int j = 0; j < f; j++) {
                float g = c.gate[ti][j];
                float sg = Ops.sigmoid(g);
                float silu = g * sg;
                float dh = dHidden[ti][j];
                dUp[ti][j] = dh * silu;
                float dSilu = dh * c.up[ti][j];
                dGate[ti][j] = dSilu * sg * (1f + g * (1f - sg));
            }
        }
        float[][] dLn2gate = Ops.linearBackward(b.wGate, c.ln2, dGate);
        float[][] dLn2up = Ops.linearBackward(b.wUp, c.ln2, dUp);
        float[][] dLn2 = Ops.add(dLn2gate, dLn2up);
        float[][] dXmidMlp = Ops.rmsNormBackward(c.xMid, b.ffnNorm, dLn2, cfg.rmsEps);
        float[][] dXmid = Ops.add(dOut, dXmidMlp);

        // Causal self-attention
        float[][] dAttn = Ops.linearBackward(b.wo, c.attn, dXmid);
        float[][] dq = new float[t][d];
        float[][] dk = new float[t][d];
        float[][] dv = new float[t][d];
        int heads = cfg.nHeads;
        int hd = cfg.headDim;
        float scale = (float) (1.0 / Math.sqrt(hd));
        float[] dp = new float[t];
        for (int h = 0; h < heads; h++) {
            int off = h * hd;
            for (int ti = 0; ti < t; ti++) {
                float dotdp = 0f;
                float[] dat = dAttn[ti];
                for (int s = 0; s <= ti; s++) {
                    float p = c.probs[h][ti][s];
                    if (p == 0f) {
                        dp[s] = 0f;
                        continue;
                    }
                    float dvAcc = 0f;
                    for (int i = 0; i < hd; i++) {
                        dvAcc += dat[off + i] * c.v[s][off + i];
                    }
                    dp[s] = dvAcc;
                    dotdp += dvAcc * p;
                    for (int i = 0; i < hd; i++) {
                        dv[s][off + i] += p * dat[off + i];
                    }
                }
                for (int s = 0; s <= ti; s++) {
                    float p = c.probs[h][ti][s];
                    float ds = p * (dp[s] - dotdp) * scale;
                    if (ds == 0f) {
                        continue;
                    }
                    for (int i = 0; i < hd; i++) {
                        dq[ti][off + i] += ds * c.k[s][off + i];
                        dk[s][off + i] += ds * c.q[ti][off + i];
                    }
                }
            }
        }
        Ops.ropeBackward(dq, heads, hd, cfg.ropeTheta);
        Ops.ropeBackward(dk, heads, hd, cfg.ropeTheta);

        float[][] dLn1q = Ops.linearBackward(b.wq, c.ln1, dq);
        float[][] dLn1k = Ops.linearBackward(b.wk, c.ln1, dk);
        float[][] dLn1v = Ops.linearBackward(b.wv, c.ln1, dv);
        float[][] dLn1 = Ops.add(Ops.add(dLn1q, dLn1k), dLn1v);
        float[][] dXinAttn = Ops.rmsNormBackward(c.input, b.attnNorm, dLn1, cfg.rmsEps);
        return Ops.add(dXmid, dXinAttn);
    }
}
