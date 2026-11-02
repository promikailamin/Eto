package eto.gguf;

/** GGUF format constants. */
public final class Gguf {
    private Gguf() {
    }

    public static final int MAGIC = 0x46554747; // "GGUF" little-endian
    public static final int VERSION = 3;
    public static final int DEFAULT_ALIGNMENT = 32;

    // metadata value types
    public static final int META_UINT8 = 0;
    public static final int META_INT8 = 1;
    public static final int META_UINT16 = 2;
    public static final int META_INT16 = 3;
    public static final int META_UINT32 = 4;
    public static final int META_INT32 = 5;
    public static final int META_FLOAT32 = 6;
    public static final int META_BOOL = 7;
    public static final int META_STRING = 8;
    public static final int META_ARRAY = 9;
    public static final int META_UINT64 = 10;
    public static final int META_INT64 = 11;
    public static final int META_FLOAT64 = 12;

    // ggml tensor types
    public static final int GGML_F32 = 0;
    public static final int GGML_F16 = 1;
    public static final int GGML_Q4_0 = 2;
    public static final int GGML_Q4_1 = 3;
    public static final int GGML_Q5_0 = 6;
    public static final int GGML_Q5_1 = 7;
    public static final int GGML_Q8_0 = 8;
    public static final int GGML_Q2_K = 10;
    public static final int GGML_Q3_K = 11;
    public static final int GGML_Q4_K = 12;
    public static final int GGML_Q5_K = 13;
    public static final int GGML_Q6_K = 14;
    public static final int GGML_Q8_K = 15;

    public static long align(long value, long alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }
}
