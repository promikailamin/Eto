package eto.gguf;

import java.nio.ByteBuffer;

/**
 * A lazily accessed tensor inside a GGUF file. Quantized rows are dequantized
 * on demand, so a large model is never expanded to full F32 in memory.
 */
public final class GgufTensor {
    public final String name;
    public final int type;
    public final long[] dims;
    public final long offset;

    private final ByteBuffer data;

    public GgufTensor(String name, int type, long[] dims, long offset, ByteBuffer data) {
        this.name = name;
        this.type = type;
        this.dims = dims;
        this.offset = offset;
        this.data = data;
    }

    public int ne0() {
        return (int) dims[0];
    }

    public long count() {
        long n = 1;
        for (long d : dims) {
            n *= d;
        }
        return n;
    }

    public long rows() {
        return count() / ne0();
    }

    public int rowBytes() {
        int bs = Quant.blockSize(type);
        int ne0 = ne0();
        if (ne0 % bs != 0) {
            throw new IllegalArgumentException(
                    "tensor " + name + " row length " + ne0 + " is not a multiple of " + bs);
        }
        return Quant.typeSize(type) * (ne0 / bs);
    }

    public long byteSize() {
        return (long) rowBytes() * rows();
    }

    private int offsetFor(long row) {
        long off = offset + row * rowBytes();
        if (off < 0 || off > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor " + name + " offset does not fit in memory map");
        }
        return (int) off;
    }

    /** Reads one row (or a full vector) as floats. */
    public void rowInto(long row, float[] out) {
        int n = ne0();
        if (out.length < n) {
            throw new IllegalArgumentException("output must hold " + n + " floats");
        }
        Quant.dequantize(type, data, offsetFor(row), out, n);
    }

    public float[] row(long row) {
        float[] out = new float[ne0()];
        rowInto(row, out);
        return out;
    }

    public float[] all() {
        long n = count();
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor too large for a single array: " + name);
        }
        float[] out = new float[(int) n];
        Quant.dequantize(type, data, offsetFor(0), out, (int) n);
        return out;
    }

    private int u8(int i) {
        return data.get(i) & 0xFF;
    }

    private int i8(int i) {
        return data.get(i);
    }

    private float f16(int i) {
        return Quant.halfToFloat(data.getShort(i));
    }

    /**
     * Fused dequantize-and-dot for one row. Avoids materializing the row, which
     * roughly halves the work in the hot matrix-vector path. Safe to call from
     * multiple threads.
     */
    public float dotRow(long row, float[] x) {
        int n = ne0();
        int off = offsetFor(row);
        return switch (type) {
            case Gguf.GGML_F32 -> dotF32(off, x, n);
            case Gguf.GGML_F16 -> dotF16(off, x, n);
            case Gguf.GGML_Q4_0 -> dotQ4_0(off, x, n);
            case Gguf.GGML_Q4_1 -> dotQ4_1(off, x, n);
            case Gguf.GGML_Q5_0 -> dotQ5_0(off, x, n);
            case Gguf.GGML_Q5_1 -> dotQ5_1(off, x, n);
            case Gguf.GGML_Q8_0 -> dotQ8_0(off, x, n);
            case Gguf.GGML_Q2_K -> dotQ2K(off, x, n);
            case Gguf.GGML_Q3_K -> dotQ3K(off, x, n);
            case Gguf.GGML_Q4_K -> dotQ4K(off, x, n);
            case Gguf.GGML_Q5_K -> dotQ5K(off, x, n);
            case Gguf.GGML_Q6_K -> dotQ6K(off, x, n);
            case Gguf.GGML_Q8_K -> dotQ8K(off, x, n);
            default -> throw new IllegalArgumentException("unsupported tensor type " + Quant.name(type));
        };
    }

    private float dotF32(int off, float[] x, int n) {
        float s = 0f;
        for (int i = 0; i < n; i++) {
            s += Float.intBitsToFloat(data.getInt(off + 4 * i)) * x[i];
        }
        return s;
    }

    private float dotF16(int off, float[] x, int n) {
        float s = 0f;
        for (int i = 0; i < n; i++) {
            s += Quant.halfToFloat(data.getShort(off + 2 * i)) * x[i];
        }
        return s;
    }

    private float dotQ4_0(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 32; b++, xi += 32) {
            int bo = off + b * 18;
            float d = f16(bo);
            float s0 = 0f;
            float s1 = 0f;
            for (int j = 0; j < 16; j++) {
                int q = u8(bo + 2 + j);
                s0 += ((q & 0x0F) - 8) * x[xi + j];
                s1 += ((q >> 4) - 8) * x[xi + j + 16];
            }
            acc += d * (s0 + s1);
        }
        return acc;
    }

    private float dotQ4_1(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 32; b++, xi += 32) {
            int bo = off + b * 20;
            float d = f16(bo);
            float m = f16(bo + 2);
            float s = 0f;
            float xs = 0f;
            for (int j = 0; j < 16; j++) {
                int q = u8(bo + 4 + j);
                s += (q & 0x0F) * x[xi + j] + (q >> 4) * x[xi + j + 16];
                xs += x[xi + j] + x[xi + j + 16];
            }
            acc += d * s + m * xs;
        }
        return acc;
    }

    private float dotQ5_0(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 32; b++, xi += 32) {
            int bo = off + b * 22;
            float d = f16(bo);
            int qh = data.getInt(bo + 2);
            float s0 = 0f;
            float s1 = 0f;
            for (int j = 0; j < 16; j++) {
                int q = u8(bo + 6 + j);
                int xh0 = ((qh >>> j) << 4) & 0x10;
                int xh1 = (qh >>> (j + 12)) & 0x10;
                s0 += (((q & 0x0F) | xh0) - 16) * x[xi + j];
                s1 += (((q >> 4) | xh1) - 16) * x[xi + j + 16];
            }
            acc += d * (s0 + s1);
        }
        return acc;
    }

    private float dotQ5_1(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 32; b++, xi += 32) {
            int bo = off + b * 24;
            float d = f16(bo);
            float m = f16(bo + 2);
            int qh = data.getInt(bo + 4);
            float s = 0f;
            float xs = 0f;
            for (int j = 0; j < 16; j++) {
                int q = u8(bo + 8 + j);
                int xh0 = ((qh >>> j) << 4) & 0x10;
                int xh1 = (qh >>> (j + 12)) & 0x10;
                s += (((q & 0x0F) | xh0) * x[xi + j])
                        + (((q >> 4) | xh1) * x[xi + j + 16]);
                xs += x[xi + j] + x[xi + j + 16];
            }
            acc += d * s + m * xs;
        }
        return acc;
    }

    private float dotQ8_0(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 32; b++, xi += 32) {
            int bo = off + b * 34;
            float d = f16(bo);
            float s = 0f;
            for (int j = 0; j < 32; j++) {
                s += i8(bo + 2 + j) * x[xi + j];
            }
            acc += d * s;
        }
        return acc;
    }

    private float dotQ2K(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 256; b++) {
            int bo = off + b * 84;
            float d = f16(bo + 80);
            float min = f16(bo + 82);
            int is = 0;
            for (int nn = 0; nn < 256; nn += 128) {
                int shift = 0;
                int qb = bo + 16 + (nn / 128) * 32;
                for (int j = 0; j < 4; j++) {
                    int sc = u8(bo + is++);
                    float dl = d * (sc & 0x0F);
                    float ml = min * (sc >> 4);
                    float sum = 0f;
                    float xs = 0f;
                    for (int l = 0; l < 16; l++) {
                        float xv = x[xi++];
                        sum += ((u8(qb + l) >> shift) & 3) * xv;
                        xs += xv;
                    }
                    acc += dl * sum - ml * xs;
                    sc = u8(bo + is++);
                    dl = d * (sc & 0x0F);
                    ml = min * (sc >> 4);
                    sum = 0f;
                    xs = 0f;
                    for (int l = 0; l < 16; l++) {
                        float xv = x[xi++];
                        sum += ((u8(qb + l + 16) >> shift) & 3) * xv;
                        xs += xv;
                    }
                    acc += dl * sum - ml * xs;
                    shift += 2;
                }
            }
        }
        return acc;
    }

    private float dotQ3K(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 256; b++) {
            int bo = off + b * 110;
            float dAll = f16(bo + 108);
            int hmBase = bo;
            int qBase = bo + 32;

            int a0 = data.getInt(bo + 96);
            int a1 = data.getInt(bo + 100);
            int tmp = data.getInt(bo + 104);
            int mask1 = 0x03030303;
            int mask2 = 0x0f0f0f0f;
            int aux0 = (a0 & mask2) | ((tmp & mask1) << 4);
            int aux1 = (a1 & mask2) | (((tmp >>> 2) & mask1) << 4);
            int aux2 = ((a0 >>> 4) & mask2) | (((tmp >>> 4) & mask1) << 4);
            int aux3 = ((a1 >>> 4) & mask2) | (((tmp >>> 6) & mask1) << 4);
            int[] aux = {aux0, aux1, aux2, aux3};

            int is = 0;
            int m = 1;
            for (int nn = 0; nn < 256; nn += 128) {
                int shift = 0;
                int qb = qBase + (nn / 128) * 32;
                for (int j = 0; j < 4; j++) {
                    int sc0 = (byte) (aux[is >>> 2] >>> (8 * (is & 3)));
                    is++;
                    float dl = dAll * (sc0 - 32);
                    float sum = 0f;
                    for (int l = 0; l < 16; l++) {
                        int qv = (u8(qb + l) >> shift) & 3;
                        int high = (u8(hmBase + l) & m) != 0 ? 0 : 4;
                        sum += (qv - high) * x[xi++];
                    }
                    acc += dl * sum;
                    int sc1 = (byte) (aux[is >>> 2] >>> (8 * (is & 3)));
                    is++;
                    dl = dAll * (sc1 - 32);
                    sum = 0f;
                    for (int l = 0; l < 16; l++) {
                        int qv = (u8(qb + l + 16) >> shift) & 3;
                        int high = (u8(hmBase + l + 16) & m) != 0 ? 0 : 4;
                        sum += (qv - high) * x[xi++];
                    }
                    acc += dl * sum;
                    shift += 2;
                    m <<= 1;
                }
            }
        }
        return acc;
    }

    private static void scaleMinK4(ByteBuffer data, int qOff, int j, int[] out) {
        if (j < 4) {
            out[0] = data.get(qOff + j) & 63;
            out[1] = data.get(qOff + j + 4) & 63;
        } else {
            int b0 = data.get(qOff + j + 4) & 0xFF;
            out[0] = (b0 & 0x0F) | (((data.get(qOff + j - 4) & 0xFF) >>> 6) << 4);
            out[1] = (b0 >>> 4) | (((data.get(qOff + j) & 0xFF) >>> 6) << 4);
        }
    }

    private float dotQ4K(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        int[] sc = new int[2];
        for (int b = 0; b < n / 256; b++) {
            int bo = off + b * 144;
            float d = f16(bo);
            float min = f16(bo + 2);
            int scales = bo + 4;
            int qBase = bo + 16;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                scaleMinK4(data, scales, is, sc);
                float d1 = d * sc[0];
                float m1 = min * sc[1];
                scaleMinK4(data, scales, is + 1, sc);
                float d2 = d * sc[0];
                float m2 = min * sc[1];
                int qb = qBase + (j / 64) * 32;
                float w1 = 0f;
                float w2 = 0f;
                float xs1 = 0f;
                float xs2 = 0f;
                for (int l = 0; l < 32; l++) {
                    int q = u8(qb + l);
                    w1 += (q & 0x0F) * x[xi + l];
                    w2 += (q >> 4) * x[xi + l + 32];
                    xs1 += x[xi + l];
                    xs2 += x[xi + l + 32];
                }
                acc += d1 * w1 - m1 * xs1 + d2 * w2 - m2 * xs2;
                xi += 64;
                is += 2;
            }
        }
        return acc;
    }

    private float dotQ5K(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        int[] sc = new int[2];
        for (int b = 0; b < n / 256; b++) {
            int bo = off + b * 176;
            float d = f16(bo);
            float min = f16(bo + 2);
            int scales = bo + 4;
            int qhBase = bo + 16;
            int qBase = bo + 48;
            int is = 0;
            int u1 = 1;
            int u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                scaleMinK4(data, scales, is, sc);
                float d1 = d * sc[0];
                float m1 = min * sc[1];
                scaleMinK4(data, scales, is + 1, sc);
                float d2 = d * sc[0];
                float m2 = min * sc[1];
                int qb = qBase + (j / 64) * 32;
                float w1 = 0f;
                float w2 = 0f;
                float xs1 = 0f;
                float xs2 = 0f;
                for (int l = 0; l < 32; l++) {
                    int q = u8(qb + l);
                    int h1 = (u8(qhBase + l) & u1) != 0 ? 16 : 0;
                    int h2 = (u8(qhBase + l) & u2) != 0 ? 16 : 0;
                    w1 += ((q & 0x0F) + h1) * x[xi + l];
                    w2 += ((q >> 4) + h2) * x[xi + l + 32];
                    xs1 += x[xi + l];
                    xs2 += x[xi + l + 32];
                }
                acc += d1 * w1 - m1 * xs1 + d2 * w2 - m2 * xs2;
                xi += 64;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        return acc;
    }

    private float dotQ6K(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 256; b++) {
            int bo = off + b * 210;
            float d = f16(bo + 208);
            int qlBase = bo;
            int qhBase = bo + 128;
            int scBase = bo + 192;
            for (int nn = 0; nn < 256; nn += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int qh = u8(qhBase + l);
                    int q1 = ((u8(qlBase + l) & 0x0F) | (((qh >> 0) & 3) << 4)) - 32;
                    int q2 = ((u8(qlBase + 32 + l) & 0x0F) | (((qh >> 2) & 3) << 4)) - 32;
                    int q3 = ((u8(qlBase + l) >> 4) | (((qh >> 4) & 3) << 4)) - 32;
                    int q4 = ((u8(qlBase + 32 + l) >> 4) | (((qh >> 6) & 3) << 4)) - 32;
                    acc += d * (i8(scBase + is) * q1 * x[xi + l]
                            + i8(scBase + is + 2) * q2 * x[xi + 32 + l]
                            + i8(scBase + is + 4) * q3 * x[xi + 64 + l]
                            + i8(scBase + is + 6) * q4 * x[xi + 96 + l]);
                }
                xi += 128;
                qlBase += 64;
                qhBase += 32;
                scBase += 8;
            }
        }
        return acc;
    }

    private float dotQ8K(int off, float[] x, int n) {
        float acc = 0f;
        int xi = 0;
        for (int b = 0; b < n / 256; b++) {
            int bo = off + b * 292;
            float d = Float.intBitsToFloat(data.getInt(bo));
            float s = 0f;
            for (int j = 0; j < 256; j++) {
                s += i8(bo + 4 + j) * x[xi + j];
            }
            acc += d * s;
            xi += 256;
        }
        return acc;
    }

    @Override
    public String toString() {
        return name + " " + java.util.Arrays.toString(dims) + " " + Quant.name(type) + " @" + offset;
    }
}
