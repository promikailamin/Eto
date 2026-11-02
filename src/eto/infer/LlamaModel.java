package eto.infer;

import eto.gguf.GgufReader;
import eto.gguf.GgufTensor;
import eto.gguf.Quant;
import eto.tokenizer.Tokenizer;
import eto.tokenizer.Tokenizers;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;

/**
 * A llama-style GGUF model: RMSNorm, RoPE, grouped-query attention, SwiGLU.
 *
 * Supports GGUF architectures "llama", "mistral" and "qwen2" (optional q/k/v
 * biases), F32/F16 and the common legacy + K quantizations. Tensors stay
 * quantized in the memory map; rows are dequantized on demand.
 */
public final class LlamaModel implements Closeable {
    public final GgufReader reader;
    public final Tokenizer tokenizer;
    public final Config cfg;

    public final GgufTensor tokenEmbd;
    public final GgufTensor output;
    public final float[] outputNorm;
    public final Layer[] layers;

    public static final class Config {
        public final String arch;
        public final int vocabSize;
        public final int dim;
        public final int nLayers;
        public final int nHeads;
        public final int nKvHeads;
        public final int headDim;
        public final int valueDim;
        public final int ffnDim;
        public final int ctxLen;
        public final float rmsEps;
        public final float ropeTheta;
        public final float ropeScale;
        public final boolean ropeNeox;
        public final boolean tiedOutput;

        Config(
                String arch,
                int vocabSize,
                int dim,
                int nLayers,
                int nHeads,
                int nKvHeads,
                int headDim,
                int valueDim,
                int ffnDim,
                int ctxLen,
                float rmsEps,
                float ropeTheta,
                float ropeScale,
                boolean ropeNeox,
                boolean tiedOutput) {
            this.arch = arch;
            this.vocabSize = vocabSize;
            this.dim = dim;
            this.nLayers = nLayers;
            this.nHeads = nHeads;
            this.nKvHeads = nKvHeads;
            this.headDim = headDim;
            this.valueDim = valueDim;
            this.ffnDim = ffnDim;
            this.ctxLen = ctxLen;
            this.rmsEps = rmsEps;
            this.ropeTheta = ropeTheta;
            this.ropeScale = ropeScale;
            this.ropeNeox = ropeNeox;
            this.tiedOutput = tiedOutput;
        }

        @Override
        public String toString() {
            return arch + " dim=" + dim + " layers=" + nLayers + " heads=" + nHeads
                    + "/" + nKvHeads + " headDim=" + headDim + " ffn=" + ffnDim
                    + " ctx=" + ctxLen + " vocab=" + vocabSize;
        }
    }

    public static final class Layer {
        public final GgufTensor attnNorm;
        public final GgufTensor wq;
        public final GgufTensor wk;
        public final GgufTensor wv;
        public final GgufTensor wo;
        public final GgufTensor ffnNorm;
        public final GgufTensor wGate;
        public final GgufTensor wUp;
        public final GgufTensor wDown;

        public final float[] qBias;
        public final float[] kBias;
        public final float[] vBias;
        public final float[] oBias;
        public final float[] attnNormW;
        public final float[] ffnNormW;

        Layer(
                GgufReader r,
                int i,
                GgufTensor wGate,
                GgufTensor wUp,
                GgufTensor wDown,
                float[] qBias,
                float[] kBias,
                float[] vBias,
                float[] oBias) {
            this.attnNorm = r.requireTensor("blk." + i + ".attn_norm.weight");
            this.wq = r.requireTensor("blk." + i + ".attn_q.weight");
            this.wk = r.requireTensor("blk." + i + ".attn_k.weight");
            this.wv = r.requireTensor("blk." + i + ".attn_v.weight");
            this.wo = r.requireTensor("blk." + i + ".attn_output.weight");
            this.ffnNorm = r.requireTensor("blk." + i + ".ffn_norm.weight");
            this.wGate = wGate;
            this.wUp = wUp;
            this.wDown = wDown;
            this.qBias = qBias;
            this.kBias = kBias;
            this.vBias = vBias;
            this.oBias = oBias;
            this.attnNormW = vector(r, this.attnNorm);
            this.ffnNormW = vector(r, this.ffnNorm);
        }
    }

    private LlamaModel(GgufReader reader, Tokenizer tok, Config cfg,
            GgufTensor tokenEmbd, GgufTensor output, float[] outputNorm, Layer[] layers) {
        this.reader = reader;
        this.tokenizer = tok;
        this.cfg = cfg;
        this.tokenEmbd = tokenEmbd;
        this.output = output;
        this.outputNorm = outputNorm;
        this.layers = layers;
    }

    public static LlamaModel open(Path path) throws IOException {
        GgufReader r = GgufReader.open(path);
        try {
            String arch = r.getString("general.architecture", "llama");
            if (!isSupportedArch(arch)) {
                throw new IllegalArgumentException("unsupported GGUF architecture: " + arch
                        + " (supported: llama, mistral, qwen2)");
            }
            Tokenizer tok = Tokenizers.fromGguf(r);
            int vocab = tok.vocabSize();
            int dim = r.getInt(arch + ".embedding_length", 0);
            int nLayers = r.getInt(arch + ".block_count", 0);
            int nHeads = r.getInt(arch + ".attention.head_count", 0);
            int nKvHeads = r.getInt(arch + ".attention.head_count_kv", nHeads);
            if (dim <= 0 || nLayers <= 0 || nHeads <= 0 || nKvHeads <= 0) {
                throw new IllegalArgumentException("GGUF is missing llama model dimensions");
            }
            if (nHeads % nKvHeads != 0) {
                throw new IllegalArgumentException("attention.head_count (" + nHeads
                        + ") must be divisible by head_count_kv (" + nKvHeads + ")");
            }
            int headDim = r.getInt(arch + ".attention.key_length", dim / nHeads);
            int valueDim = r.getInt(arch + ".attention.value_length", headDim);
            if (headDim % 2 != 0) {
                throw new IllegalArgumentException("head dimension must be even for RoPE");
            }
            int ffn = r.getInt(arch + ".feed_forward_length", 0);
            if (ffn <= 0) {
                throw new IllegalArgumentException("GGUF is missing " + arch + ".feed_forward_length");
            }
            int ctx = r.getInt(arch + ".context_length", 2048);
            float eps = r.getFloat(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
            float ropeTheta = r.getFloat(arch + ".rope.freq_base", 10000f);
            String scalingType = r.getString(arch + ".rope.scaling.type", "none");
            float ropeScale = 1f;
            if ("linear".equals(scalingType)) {
                ropeScale = r.getFloat(arch + ".rope.scaling.factor", 1f);
            } else if (!"none".equals(scalingType) && scalingType != null) {
                System.err.println("warning: unsupported rope scaling '" + scalingType
                        + "'; using unscaled RoPE");
            }

            GgufTensor tokenEmbd = r.requireTensor("token_embd.weight");
            GgufTensor output = r.tensor("output.weight");
            boolean tied = output == null;
            if (tied) {
                output = tokenEmbd;
            }
            GgufTensor outputNorm = r.requireTensor("output_norm.weight");
            float[] outNorm = vector(r, outputNorm);

            if (tokenEmbd.dims.length < 2 || tokenEmbd.ne0() != dim) {
                throw new IllegalArgumentException("token_embd.weight shape does not match embedding_length");
            }
            if (tokenEmbd.rows() != vocab) {
                throw new IllegalArgumentException("token_embd.weight rows " + tokenEmbd.rows()
                        + " do not match tokenizer vocab " + vocab);
            }

            Layer[] layers = new Layer[nLayers];
            for (int i = 0; i < nLayers; i++) {
                String p = "blk." + i + ".";
                GgufTensor gate = r.tensor(p + "ffn_gate.weight");
                if (gate == null) {
                    throw new IllegalArgumentException("unsupported model: missing " + p + "ffn_gate.weight "
                            + "(only SwiGLU llama-family models are supported)");
                }
                GgufTensor up = r.requireTensor(p + "ffn_up.weight");
                GgufTensor down = r.requireTensor(p + "ffn_down.weight");
                float[] qb = optionalVector(r, p + "attn_q.bias");
                float[] kb = optionalVector(r, p + "attn_k.bias");
                float[] vb = optionalVector(r, p + "attn_v.bias");
                float[] ob = optionalVector(r, p + "attn_output.bias");
                layers[i] = new Layer(r, i, gate, up, down, qb, kb, vb, ob);

                requireShape(layers[i].wq, nHeads * headDim, dim, p + "attn_q.weight");
                requireShape(layers[i].wk, nKvHeads * headDim, dim, p + "attn_k.weight");
                requireShape(layers[i].wv, nKvHeads * valueDim, dim, p + "attn_v.weight");
                requireShape(layers[i].wo, dim, nHeads * headDim, p + "attn_output.weight");
                requireShape(gate, ffn, dim, p + "ffn_gate.weight");
                requireShape(up, ffn, dim, p + "ffn_up.weight");
                requireShape(down, dim, ffn, p + "ffn_down.weight");
                requireVector(layers[i].attnNorm, dim, p + "attn_norm.weight");
                requireVector(layers[i].ffnNorm, dim, p + "ffn_norm.weight");
            }

            Config cfg = new Config(arch, vocab, dim, nLayers, nHeads, nKvHeads, headDim, valueDim,
                    ffn, ctx, eps, ropeTheta, ropeScale, usesNeoxRope(arch), tied);
            checkSupportedTypes(r);
            return new LlamaModel(r, tok, cfg, tokenEmbd, output, outNorm, layers);
        } catch (RuntimeException e) {
            r.close();
            throw e;
        }
    }

    private static boolean isSupportedArch(String arch) {
        return "llama".equals(arch) || "mistral".equals(arch) || "qwen2".equals(arch);
    }

    /**
     * GGUF conversion permutes LLaMA-family q/k rows so that RoPE rotates
     * adjacent values, while Qwen2/3 keep the HF half-split layout. These are
     * the two rope types llama.cpp calls NORM and NEOX.
     */
    private static boolean usesNeoxRope(String arch) {
        return arch.startsWith("qwen");
    }

    private static void checkSupportedTypes(GgufReader r) {
        for (GgufReader.TensorInfo ti : r.tensors().values()) {
            if (!Quant.supported(ti.type)) {
                throw new IllegalArgumentException("unsupported tensor type " + Quant.name(ti.type)
                        + " for tensor " + ti.name + " (supported: F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/"
                        + "Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K)");
            }
        }
    }

    private static void requireShape(GgufTensor t, long rows, long cols, String what) {
        if (t == null) {
            throw new IllegalArgumentException("missing tensor " + what);
        }
        if (t.dims.length != 2 || t.dims[0] != cols || t.dims[1] != rows) {
            throw new IllegalArgumentException("tensor " + what + " has shape " + java.util.Arrays.toString(t.dims)
                    + ", expected [" + cols + ", " + rows + "]");
        }
    }

    private static void requireVector(GgufTensor t, long n, String what) {
        if (t == null) {
            throw new IllegalArgumentException("missing tensor " + what);
        }
        if (t.dims.length != 1 || t.dims[0] != n) {
            throw new IllegalArgumentException("tensor " + what + " has shape " + java.util.Arrays.toString(t.dims)
                    + ", expected [" + n + "]");
        }
    }

    private static float[] vector(GgufReader r, GgufTensor t) {
        if (t == null) {
            throw new IllegalArgumentException("missing tensor");
        }
        if (t.dims.length != 1) {
            throw new IllegalArgumentException("tensor " + t.name + " is not a vector");
        }
        return t.all();
    }

    private static float[] optionalVector(GgufReader r, String name) {
        GgufTensor t = r.tensor(name);
        return t == null ? null : vector(r, t);
    }

    public Session newSession(int maxCtx) {
        return new Session(this, Math.min(maxCtx, cfg.ctxLen));
    }

    @Override
    public void close() throws IOException {
        reader.close();
    }
}
