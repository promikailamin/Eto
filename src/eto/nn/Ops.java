package eto.nn;

/**
 * Small, dependency-free tensor math used by the trainer.
 *
 * Activations are {@code float[seq][features]}. Weight matrices are
 * {@code Mat} with row-major storage. No BLAS, no native code.
 */
public final class Ops {
    private Ops() {
    }

    public static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    public static float silu(float x) {
        return x * sigmoid(x);
    }

    public static float dot(float[] a, int ao, float[] b, int bo, int n) {
        float s = 0f;
        for (int i = 0; i < n; i++) {
            s += a[ao + i] * b[bo + i];
        }
        return s;
    }

    /** y[t][o] = sum_i W[o][i] * x[t][i] */
    public static float[][] linear(Mat w, float[][] x) {
        int t = x.length;
        int in = w.cols;
        int out = w.rows;
        float[][] y = new float[t][out];
        float[] ww = w.w;
        for (int ti = 0; ti < t; ti++) {
            float[] xt = x[ti];
            float[] yt = y[ti];
            for (int o = 0; o < out; o++) {
                int base = o * in;
                float s = 0f;
                for (int i = 0; i < in; i++) {
                    s += ww[base + i] * xt[i];
                }
                yt[o] = s;
            }
        }
        return y;
    }

    /**
     * Backprop for {@link #linear}. Accumulates weight gradients into {@code w.g}
     * and returns the gradient with respect to the input.
     */
    public static float[][] linearBackward(Mat w, float[][] x, float[][] dy) {
        int t = x.length;
        int in = w.cols;
        int out = w.rows;
        float[][] dx = new float[t][in];
        float[] ww = w.w;
        float[] gg = w.g;
        for (int ti = 0; ti < t; ti++) {
            float[] xt = x[ti];
            float[] dyt = dy[ti];
            float[] dxt = dx[ti];
            for (int o = 0; o < out; o++) {
                float d = dyt[o];
                if (d == 0f) {
                    continue;
                }
                int base = o * in;
                for (int i = 0; i < in; i++) {
                    gg[base + i] += d * xt[i];
                    dxt[i] += d * ww[base + i];
                }
            }
        }
        return dx;
    }

    public static float[][] add(float[][] a, float[][] b) {
        int t = a.length;
        int d = a[0].length;
        float[][] out = new float[t][d];
        for (int ti = 0; ti < t; ti++) {
            float[] at = a[ti];
            float[] bt = b[ti];
            float[] ot = out[ti];
            for (int i = 0; i < d; i++) {
                ot[i] = at[i] + bt[i];
            }
        }
        return out;
    }

    /** RMSNorm forward: y = x / sqrt(mean(x^2) + eps) * weight */
    public static float[][] rmsNorm(float[][] x, float[] weight, float eps) {
        int t = x.length;
        int d = x[0].length;
        float[][] y = new float[t][d];
        for (int ti = 0; ti < t; ti++) {
            float[] xt = x[ti];
            float[] yt = y[ti];
            float s = 0f;
            for (int i = 0; i < d; i++) {
                s += xt[i] * xt[i];
            }
            float r = (float) (1.0 / Math.sqrt(s / d + eps));
            for (int i = 0; i < d; i++) {
                yt[i] = xt[i] * r * weight[i];
            }
        }
        return y;
    }

    /** Backprop for {@link #rmsNorm}. Accumulates weight gradients into {@code norm.g}. */
    public static float[][] rmsNormBackward(float[][] x, Vec norm, float[][] dy, float eps) {
        int t = x.length;
        int d = x[0].length;
        float[][] dx = new float[t][d];
        float[] w = norm.w;
        float[] dw = norm.g;
        for (int ti = 0; ti < t; ti++) {
            float[] xt = x[ti];
            float[] dyt = dy[ti];
            float[] dxt = dx[ti];
            float s = 0f;
            for (int i = 0; i < d; i++) {
                s += xt[i] * xt[i];
            }
            float r = (float) (1.0 / Math.sqrt(s / d + eps));
            float c = 0f;
            for (int i = 0; i < d; i++) {
                dw[i] += dyt[i] * xt[i] * r;
                c += dyt[i] * w[i] * xt[i];
            }
            float rr = r * r;
            for (int i = 0; i < d; i++) {
                dxt[i] = r * (dyt[i] * w[i] - xt[i] * rr * c / d);
            }
        }
        return dx;
    }

    /**
     * In-place RoPE (llama / GPT-NeoX rotate-half layout). Rows are token
     * positions, columns are {@code heads * headDim}.
     */
    public static void rope(float[][] x, int heads, int headDim, float theta) {
        int t = x.length;
        int half = headDim / 2;
        for (int ti = 0; ti < t; ti++) {
            float[] row = x[ti];
            for (int i = 0; i < half; i++) {
                double freq = 1.0 / Math.pow(theta, (2.0 * i) / headDim);
                float ang = (float) (ti * freq);
                float c = (float) Math.cos(ang);
                float s = (float) Math.sin(ang);
                for (int h = 0; h < heads; h++) {
                    int base = h * headDim;
                    float x1 = row[base + i];
                    float x2 = row[base + i + half];
                    row[base + i] = x1 * c - x2 * s;
                    row[base + i + half] = x1 * s + x2 * c;
                }
            }
        }
    }

    /** Backprop through {@link #rope}; rotation is orthogonal, so use the inverse. */
    public static void ropeBackward(float[][] dx, int heads, int headDim, float theta) {
        int t = dx.length;
        int half = headDim / 2;
        for (int ti = 0; ti < t; ti++) {
            float[] row = dx[ti];
            for (int i = 0; i < half; i++) {
                double freq = 1.0 / Math.pow(theta, (2.0 * i) / headDim);
                float ang = (float) (ti * freq);
                float c = (float) Math.cos(ang);
                float s = (float) Math.sin(ang);
                for (int h = 0; h < heads; h++) {
                    int base = h * headDim;
                    float d1 = row[base + i];
                    float d2 = row[base + i + half];
                    row[base + i] = d1 * c + d2 * s;
                    row[base + i + half] = -d1 * s + d2 * c;
                }
            }
        }
    }
}
