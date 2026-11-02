package eto.gguf;

import java.nio.ByteBuffer;

/**
 * Pure-Java dequantizers for the ggml tensor formats most commonly found in
 * GGUF files. Ported from the reference implementations in ggml-quants.c.
 */
public final class Quant {
    private Quant() {
    }

    public static int blockSize(int type) {
        return switch (type) {
            case Gguf.GGML_F32, Gguf.GGML_F16 -> 1;
            case Gguf.GGML_Q4_0, Gguf.GGML_Q4_1, Gguf.GGML_Q5_0, Gguf.GGML_Q5_1, Gguf.GGML_Q8_0 -> 32;
            case Gguf.GGML_Q2_K, Gguf.GGML_Q3_K, Gguf.GGML_Q4_K, Gguf.GGML_Q5_K, Gguf.GGML_Q6_K, Gguf.GGML_Q8_K -> 256;
            default -> throw new IllegalArgumentException("unsupported tensor type " + type);
        };
    }

    public static int typeSize(int type) {
        return switch (type) {
            case Gguf.GGML_F32 -> 4;
            case Gguf.GGML_F16 -> 2;
            case Gguf.GGML_Q4_0 -> 18;
            case Gguf.GGML_Q4_1 -> 20;
            case Gguf.GGML_Q5_0 -> 22;
            case Gguf.GGML_Q5_1 -> 24;
            case Gguf.GGML_Q8_0 -> 34;
            case Gguf.GGML_Q2_K -> 84;
            case Gguf.GGML_Q3_K -> 110;
            case Gguf.GGML_Q4_K -> 144;
            case Gguf.GGML_Q5_K -> 176;
            case Gguf.GGML_Q6_K -> 210;
            case Gguf.GGML_Q8_K -> 292;
            default -> throw new IllegalArgumentException("unsupported tensor type " + type);
        };
    }

    public static String name(int type) {
        return switch (type) {
            case Gguf.GGML_F32 -> "F32";
            case Gguf.GGML_F16 -> "F16";
            case Gguf.GGML_Q4_0 -> "Q4_0";
            case Gguf.GGML_Q4_1 -> "Q4_1";
            case Gguf.GGML_Q5_0 -> "Q5_0";
            case Gguf.GGML_Q5_1 -> "Q5_1";
            case Gguf.GGML_Q8_0 -> "Q8_0";
            case Gguf.GGML_Q2_K -> "Q2_K";
            case Gguf.GGML_Q3_K -> "Q3_K";
            case Gguf.GGML_Q4_K -> "Q4_K";
            case Gguf.GGML_Q5_K -> "Q5_K";
            case Gguf.GGML_Q6_K -> "Q6_K";
            case Gguf.GGML_Q8_K -> "Q8_K";
            default -> "unknown(" + type + ")";
        };
    }

    public static boolean supported(int type) {
        return switch (type) {
            case Gguf.GGML_F32, Gguf.GGML_F16,
                 Gguf.GGML_Q4_0, Gguf.GGML_Q4_1, Gguf.GGML_Q5_0, Gguf.GGML_Q5_1, Gguf.GGML_Q8_0,
                 Gguf.GGML_Q2_K, Gguf.GGML_Q3_K, Gguf.GGML_Q4_K, Gguf.GGML_Q5_K, Gguf.GGML_Q6_K,
                 Gguf.GGML_Q8_K -> true;
            default -> false;
        };
    }

    public static float halfToFloat(short h) {
        int bits = h & 0xFFFF;
        int sign = (bits >>> 15) & 1;
        int exp = (bits >>> 10) & 0x1F;
        int mant = bits & 0x3FF;
        int fbits;
        if (exp == 0) {
            if (mant == 0) {
                fbits = sign << 31;
            } else {
                exp = 127 - 15 + 1;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x3FF;
                fbits = (sign << 31) | (exp << 23) | (mant << 13);
            }
        } else if (exp == 31) {
            fbits = (sign << 31) | (0xFF << 23) | (mant << 13);
        } else {
            fbits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        return Float.intBitsToFloat(fbits);
    }

    private static int u8(ByteBuffer b, int i) {
        return b.get(i) & 0xFF;
    }

    private static int i8(ByteBuffer b, int i) {
        return b.get(i);
    }

    private static int u16(ByteBuffer b, int i) {
        return b.getShort(i) & 0xFFFF;
    }

    private static float f16(ByteBuffer b, int i) {
        return halfToFloat(b.getShort(i));
    }

    /**
     * Dequantize {@code n} values of the given type starting at {@code byteOffset}
     * into {@code y[0..n)}. {@code n} must be a multiple of the block size.
     */
    public static void dequantize(int type, ByteBuffer b, int byteOffset, float[] y, int n) {
        switch (type) {
            case Gguf.GGML_F32 -> deqF32(b, byteOffset, y, n);
            case Gguf.GGML_F16 -> deqF16(b, byteOffset, y, n);
            case Gguf.GGML_Q4_0 -> deqQ4_0(b, byteOffset, y, n);
            case Gguf.GGML_Q4_1 -> deqQ4_1(b, byteOffset, y, n);
            case Gguf.GGML_Q5_0 -> deqQ5_0(b, byteOffset, y, n);
            case Gguf.GGML_Q5_1 -> deqQ5_1(b, byteOffset, y, n);
            case Gguf.GGML_Q8_0 -> deqQ8_0(b, byteOffset, y, n);
            case Gguf.GGML_Q2_K -> deqQ2K(b, byteOffset, y, n);
            case Gguf.GGML_Q3_K -> deqQ3K(b, byteOffset, y, n);
            case Gguf.GGML_Q4_K -> deqQ4K(b, byteOffset, y, n);
            case Gguf.GGML_Q5_K -> deqQ5K(b, byteOffset, y, n);
            case Gguf.GGML_Q6_K -> deqQ6K(b, byteOffset, y, n);
            case Gguf.GGML_Q8_K -> deqQ8K(b, byteOffset, y, n);
            default -> throw new IllegalArgumentException("unsupported tensor type " + name(type));
        }
    }

    private static void deqF32(ByteBuffer b, int off, float[] y, int n) {
        for (int i = 0; i < n; i++) {
            y[i] = Float.intBitsToFloat(b.getInt(off + 4 * i));
        }
    }

    private static void deqF16(ByteBuffer b, int off, float[] y, int n) {
        for (int i = 0; i < n; i++) {
            y[i] = f16(b, off + 2 * i);
        }
    }

    private static void deqQ4_0(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 32;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 18;
            float d = f16(b, bo);
            int base = i * 32;
            for (int j = 0; j < 16; j++) {
                int q = u8(b, bo + 2 + j);
                y[base + j] = ((q & 0x0F) - 8) * d;
                y[base + j + 16] = ((q >> 4) - 8) * d;
            }
        }
    }

    private static void deqQ4_1(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 32;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 20;
            float d = f16(b, bo);
            float m = f16(b, bo + 2);
            int base = i * 32;
            for (int j = 0; j < 16; j++) {
                int q = u8(b, bo + 4 + j);
                y[base + j] = (q & 0x0F) * d + m;
                y[base + j + 16] = (q >> 4) * d + m;
            }
        }
    }

    private static void deqQ5_0(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 32;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 22;
            float d = f16(b, bo);
            int qh = b.getInt(bo + 2);
            int base = i * 32;
            for (int j = 0; j < 16; j++) {
                int q = u8(b, bo + 6 + j);
                int xh0 = ((qh >>> j) << 4) & 0x10;
                int xh1 = (qh >>> (j + 12)) & 0x10;
                int x0 = ((q & 0x0F) | xh0) - 16;
                int x1 = ((q >> 4) | xh1) - 16;
                y[base + j] = x0 * d;
                y[base + j + 16] = x1 * d;
            }
        }
    }

    private static void deqQ5_1(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 32;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 24;
            float d = f16(b, bo);
            float m = f16(b, bo + 2);
            int qh = b.getInt(bo + 4);
            int base = i * 32;
            for (int j = 0; j < 16; j++) {
                int q = u8(b, bo + 8 + j);
                int xh0 = ((qh >>> j) << 4) & 0x10;
                int xh1 = (qh >>> (j + 12)) & 0x10;
                int x0 = (q & 0x0F) | xh0;
                int x1 = (q >> 4) | xh1;
                y[base + j] = x0 * d + m;
                y[base + j + 16] = x1 * d + m;
            }
        }
    }

    private static void deqQ8_0(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 32;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 34;
            float d = f16(b, bo);
            int base = i * 32;
            for (int j = 0; j < 32; j++) {
                y[base + j] = i8(b, bo + 2 + j) * d;
            }
        }
    }

    private static void deqQ2K(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 256;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 84;
            float d = f16(b, bo + 80);
            float min = f16(b, bo + 82);
            int is = 0;
            int yp = i * 256;
            for (int nn = 0; nn < 256; nn += 128) {
                int shift = 0;
                int qb = bo + 16 + (nn / 128) * 32;
                for (int j = 0; j < 4; j++) {
                    int sc = u8(b, bo + is++);
                    float dl = d * (sc & 0x0F);
                    float ml = min * (sc >> 4);
                    for (int l = 0; l < 16; l++) {
                        y[yp++] = dl * ((u8(b, qb + l) >> shift) & 3) - ml;
                    }
                    sc = u8(b, bo + is++);
                    dl = d * (sc & 0x0F);
                    ml = min * (sc >> 4);
                    for (int l = 0; l < 16; l++) {
                        y[yp++] = dl * ((u8(b, qb + l + 16) >> shift) & 3) - ml;
                    }
                    shift += 2;
                }
            }
        }
    }

    private static void deqQ3K(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 256;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 110;
            float dAll = f16(b, bo + 108);
            int hmBase = bo;
            int qBase = bo + 32;

            int[] aux = new int[4];
            aux[0] = b.getInt(bo + 96);
            aux[1] = b.getInt(bo + 100);
            int tmp = b.getInt(bo + 104);
            aux[2] = tmp;
            int mask1 = 0x03030303;
            int mask2 = 0x0f0f0f0f;
            aux[2] = ((aux[0] >>> 4) & mask2) | (((tmp >>> 4) & mask1) << 4);
            aux[3] = ((aux[1] >>> 4) & mask2) | (((tmp >>> 6) & mask1) << 4);
            aux[0] = (aux[0] & mask2) | (((tmp) & mask1) << 4);
            aux[1] = (aux[1] & mask2) | (((tmp >>> 2) & mask1) << 4);

            int is = 0;
            int yp = i * 256;
            int m = 1;
            for (int nn = 0; nn < 256; nn += 128) {
                int shift = 0;
                int qb = qBase + (nn / 128) * 32;
                for (int j = 0; j < 4; j++) {
                    int sc0 = (byte) (aux[is >>> 2] >>> (8 * (is & 3)));
                    is++;
                    float dl = dAll * (sc0 - 32);
                    for (int l = 0; l < 16; l++) {
                        int qv = (u8(b, qb + l) >> shift) & 3;
                        int high = (u8(b, hmBase + l) & m) != 0 ? 0 : 4;
                        y[yp++] = dl * (qv - high);
                    }
                    int sc1 = (byte) (aux[is >>> 2] >>> (8 * (is & 3)));
                    is++;
                    dl = dAll * (sc1 - 32);
                    for (int l = 0; l < 16; l++) {
                        int qv = (u8(b, qb + l + 16) >> shift) & 3;
                        int high = (u8(b, hmBase + l + 16) & m) != 0 ? 0 : 4;
                        y[yp++] = dl * (qv - high);
                    }
                    shift += 2;
                    m <<= 1;
                }
            }
        }
    }

    private static void getScaleMinK4(ByteBuffer b, int qOff, int j, int[] out) {
        if (j < 4) {
            out[0] = u8(b, qOff + j) & 63;
            out[1] = u8(b, qOff + j + 4) & 63;
        } else {
            out[0] = (u8(b, qOff + j + 4) & 0x0F) | ((u8(b, qOff + j - 4) >>> 6) << 4);
            out[1] = (u8(b, qOff + j + 4) >>> 4) | ((u8(b, qOff + j) >>> 6) << 4);
        }
    }

    private static void deqQ4K(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 256;
        int[] sc = new int[2];
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 144;
            float d = f16(b, bo);
            float min = f16(b, bo + 2);
            int scales = bo + 4;
            int qBase = bo + 16;
            int yp = i * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                getScaleMinK4(b, scales, is, sc);
                float d1 = d * sc[0];
                float m1 = min * sc[1];
                getScaleMinK4(b, scales, is + 1, sc);
                float d2 = d * sc[0];
                float m2 = min * sc[1];
                int qb = qBase + (j / 64) * 32;
                for (int l = 0; l < 32; l++) {
                    y[yp++] = d1 * (u8(b, qb + l) & 0x0F) - m1;
                }
                for (int l = 0; l < 32; l++) {
                    y[yp++] = d2 * (u8(b, qb + l) >> 4) - m2;
                }
                is += 2;
            }
        }
    }

    private static void deqQ5K(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 256;
        int[] sc = new int[2];
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 176;
            float d = f16(b, bo);
            float min = f16(b, bo + 2);
            int scales = bo + 4;
            int qhBase = bo + 16;
            int qBase = bo + 48;
            int yp = i * 256;
            int is = 0;
            int u1 = 1;
            int u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                getScaleMinK4(b, scales, is, sc);
                float d1 = d * sc[0];
                float m1 = min * sc[1];
                getScaleMinK4(b, scales, is + 1, sc);
                float d2 = d * sc[0];
                float m2 = min * sc[1];
                int qb = qBase + (j / 64) * 32;
                for (int l = 0; l < 32; l++) {
                    int q = u8(b, qb + l);
                    int high = (u8(b, qhBase + l) & u1) != 0 ? 16 : 0;
                    y[yp++] = d1 * ((q & 0x0F) + high) - m1;
                }
                for (int l = 0; l < 32; l++) {
                    int q = u8(b, qb + l);
                    int high = (u8(b, qhBase + l) & u2) != 0 ? 16 : 0;
                    y[yp++] = d2 * ((q >> 4) + high) - m2;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    private static void deqQ6K(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 256;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 210;
            float d = f16(b, bo + 208);
            int qlBase = bo;
            int qhBase = bo + 128;
            int scBase = bo + 192;
            int yp = i * 256;
            for (int nn = 0; nn < 256; nn += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int qh = u8(b, qhBase + l);
                    int q1 = ((u8(b, qlBase + l) & 0x0F) | (((qh >> 0) & 3) << 4)) - 32;
                    int q2 = ((u8(b, qlBase + 32 + l) & 0x0F) | (((qh >> 2) & 3) << 4)) - 32;
                    int q3 = ((u8(b, qlBase + l) >> 4) | (((qh >> 4) & 3) << 4)) - 32;
                    int q4 = ((u8(b, qlBase + 32 + l) >> 4) | (((qh >> 6) & 3) << 4)) - 32;
                    y[yp + l] = d * i8(b, scBase + is) * q1;
                    y[yp + 32 + l] = d * i8(b, scBase + is + 2) * q2;
                    y[yp + 64 + l] = d * i8(b, scBase + is + 4) * q3;
                    y[yp + 96 + l] = d * i8(b, scBase + is + 6) * q4;
                }
                yp += 128;
                qlBase += 64;
                qhBase += 32;
                scBase += 8;
            }
        }
    }

    private static void deqQ8K(ByteBuffer b, int off, float[] y, int n) {
        int nb = n / 256;
        for (int i = 0; i < nb; i++) {
            int bo = off + i * 292;
            float d = Float.intBitsToFloat(b.getInt(bo));
            int base = i * 256;
            for (int j = 0; j < 256; j++) {
                y[base + j] = i8(b, bo + 4 + j) * d;
            }
        }
    }
}
