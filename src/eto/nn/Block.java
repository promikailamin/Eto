package eto.nn;

import java.util.List;
import java.util.Random;

/** One transformer block with pre-norm attention and a SwiGLU feed-forward net. */
public final class Block {
    public final Vec attnNorm;
    public final Mat wq;
    public final Mat wk;
    public final Mat wv;
    public final Mat wo;

    public final Vec ffnNorm;
    public final Mat wGate;
    public final Mat wUp;
    public final Mat wDown;

    public Block(String prefix, ModelConfig cfg) {
        int d = cfg.dim;
        int f = cfg.ffnDim;
        attnNorm = new Vec(prefix + "attn_norm.weight", d);
        wq = new Mat(prefix + "attn_q.weight", d, d);
        wk = new Mat(prefix + "attn_k.weight", d, d);
        wv = new Mat(prefix + "attn_v.weight", d, d);
        wo = new Mat(prefix + "attn_output.weight", d, d);

        ffnNorm = new Vec(prefix + "ffn_norm.weight", d);
        wGate = new Mat(prefix + "ffn_gate.weight", f, d);
        wUp = new Mat(prefix + "ffn_up.weight", f, d);
        wDown = new Mat(prefix + "ffn_down.weight", d, f);
    }

    public void init(Random rng, float std, float stdResidual) {
        Init.fill(attnNorm.w, 1f);
        Init.fill(ffnNorm.w, 1f);
        Init.normal(rng, wq.w, std);
        Init.normal(rng, wk.w, std);
        Init.normal(rng, wv.w, std);
        Init.normal(rng, wo.w, stdResidual);
        Init.normal(rng, wGate.w, std);
        Init.normal(rng, wUp.w, std);
        Init.normal(rng, wDown.w, stdResidual);
    }

    public void addParams(List<TensorParam> out) {
        out.add(attnNorm);
        out.add(wq);
        out.add(wk);
        out.add(wv);
        out.add(wo);
        out.add(ffnNorm);
        out.add(wGate);
        out.add(wUp);
        out.add(wDown);
    }
}
