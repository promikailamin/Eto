package eto.nn;

import java.util.List;

/** AdamW with decoupled weight decay and global-norm gradient clipping. */
public final class AdamW {
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private final float maxGradNorm;
    private int step;

    public AdamW(float beta1, float beta2, float eps, float weightDecay, float maxGradNorm) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.maxGradNorm = maxGradNorm;
    }

    public static AdamW defaults() {
        return new AdamW(0.9f, 0.99f, 1e-8f, 0.01f, 1.0f);
    }

    public int stepCount() {
        return step;
    }

    public void restoreStepCount(int step) {
        if (step < 0) {
            throw new IllegalArgumentException("optimizer step cannot be negative");
        }
        this.step = step;
    }

    public void zeroGrad(List<TensorParam> params) {
        for (TensorParam p : params) {
            java.util.Arrays.fill(p.g(), 0f);
        }
    }

    public void step(List<TensorParam> params, float lr) {
        step++;
        clip(params);
        float bc1 = 1f - (float) Math.pow(beta1, step);
        float bc2 = 1f - (float) Math.pow(beta2, step);
        for (TensorParam p : params) {
            float[] w = p.w();
            float[] g = p.g();
            float[] m = p.m();
            float[] v = p.v();
            boolean decay = !p.isVector();
            for (int i = 0; i < w.length; i++) {
                float gi = g[i];
                m[i] = beta1 * m[i] + (1f - beta1) * gi;
                v[i] = beta2 * v[i] + (1f - beta2) * gi * gi;
                float mh = m[i] / bc1;
                float vh = v[i] / bc2;
                if (decay) {
                    w[i] -= lr * weightDecay * w[i];
                }
                w[i] -= lr * mh / (float) (Math.sqrt(vh) + eps);
            }
        }
    }

    private void clip(List<TensorParam> params) {
        double sum = 0.0;
        for (TensorParam p : params) {
            for (float g : p.g()) {
                sum += (double) g * g;
            }
        }
        double norm = Math.sqrt(sum);
        if (norm > maxGradNorm) {
            float scale = (float) (maxGradNorm / (norm + 1e-12));
            for (TensorParam p : params) {
                float[] g = p.g();
                for (int i = 0; i < g.length; i++) {
                    g[i] *= scale;
                }
            }
        }
    }
}
