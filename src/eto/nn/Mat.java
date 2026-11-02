package eto.nn;

/** A dense weight matrix stored as {@code rows * cols} floats, row-major. */
public final class Mat implements TensorParam {
    private final String name;
    public final int rows;
    public final int cols;

    public final float[] w;
    public final float[] g;
    public final float[] m;
    public final float[] v;

    public Mat(String name, int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("bad matrix shape " + rows + "x" + cols + " for " + name);
        }
        this.name = name;
        this.rows = rows;
        this.cols = cols;
        int n = rows * cols;
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
        return rows;
    }

    @Override
    public int cols() {
        return cols;
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

    public void zeroGrad() {
        java.util.Arrays.fill(g, 0f);
    }
}
