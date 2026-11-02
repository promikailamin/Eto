package eto.nn;

/**
 * A parameter tensor. All model parameters are either a matrix (rows x cols)
 * or a vector (1 x n). The data is stored row-major with the last dimension
 * contiguous, which is exactly the layout GGUF expects (ne[0] fastest).
 */
public interface TensorParam {
    String name();

    int rows();

    int cols();

    float[] w();

    float[] g();

    float[] m();

    float[] v();

    default int length() {
        return rows() * cols();
    }

    default boolean isVector() {
        return rows() == 1;
    }
}
