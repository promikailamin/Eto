package eto.nn;

/** Hyper-parameters of the tiny llama-style model. */
public final class ModelConfig {
    public final int vocabSize;
    public final int dim;
    public final int nLayers;
    public final int nHeads;
    public final int ffnDim;
    public final int ctxLen;
    public final float rmsEps;
    public final float ropeTheta;
    public final int headDim;

    public ModelConfig(
            int vocabSize,
            int dim,
            int nLayers,
            int nHeads,
            int ffnDim,
            int ctxLen,
            float rmsEps,
            float ropeTheta) {
        if (vocabSize <= 0 || dim <= 0 || nLayers <= 0 || nHeads <= 0 || ffnDim <= 0 || ctxLen <= 1) {
            throw new IllegalArgumentException("model dimensions must be positive");
        }
        if (dim % nHeads != 0) {
            throw new IllegalArgumentException("dim (" + dim + ") must be divisible by heads (" + nHeads + ")");
        }
        if ((dim / nHeads) % 2 != 0) {
            throw new IllegalArgumentException("head dimension must be even for RoPE");
        }
        this.vocabSize = vocabSize;
        this.dim = dim;
        this.nLayers = nLayers;
        this.nHeads = nHeads;
        this.ffnDim = ffnDim;
        this.ctxLen = ctxLen;
        this.rmsEps = rmsEps;
        this.ropeTheta = ropeTheta;
        this.headDim = dim / nHeads;
    }

    public long paramCount() {
        long n = (long) vocabSize * dim * 2L + dim;
        long block = (long) dim * 4L * dim + (long) 3L * dim * ffnDim + 2L * dim;
        return n + block * nLayers;
    }

    @Override
    public String toString() {
        return "ModelConfig{vocab=" + vocabSize + ", dim=" + dim + ", layers=" + nLayers
                + ", heads=" + nHeads + ", ffn=" + ffnDim + ", ctx=" + ctxLen + "}";
    }
}
