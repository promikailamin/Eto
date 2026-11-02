package eto.nn;

/** A parameter vector (RMSNorm weights, biases, ...). */
public final class Vec implements TensorParam {
    private final String name;
    public final int n;

    public final float[] w;
    public final float[] g;
    public final float[] m;
    public final float[] v;

    public Vec(String name, int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("bad vector size " + n + " for " + name);
        }
        this.name = name;
        this.n = n;
        this.w = new float[n];
        this.g = new float[n];
        this.m = new float[n];
        this.v = new float[n];
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public int rows() {
        return 1;
    }

    @Override
    public int cols() {
        return n;
    }

    @Override
    public float[] w() {
        return w;
    }

    @Override
    public float[] g() {
        return g;
    }

    @Override
    public float[] m() {
        return m;
    }

    @Override
    public float[] v() {
        return v;
    }

    public void fill(float x) {
        java.util.Arrays.fill(w, x);
    }

    public void zeroGrad() {
        java.util.Arrays.fill(g, 0f);
    }
}
