package eto.gguf;

import java.io.Closeable;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/** Pure-Java GGUF reader with support for GGUF v2/v3. */
public final class GgufReader implements Closeable {
    public static final class TensorInfo {
        public final String name;
        public final long[] dims;
        public final int type;
        public final long offset;

        TensorInfo(String name, long[] dims, int type, long offset) {
            this.name = name;
            this.dims = dims;
            this.type = type;
            this.offset = offset;
        }

        public long count() {
            long n = 1;
            for (long d : dims) {
                n *= d;
            }
            return n;
        }

        @Override
        public String toString() {
            return name + " " + java.util.Arrays.toString(dims) + " " + Quant.name(type) + " @" + offset;
        }
    }

    private final int version;
    private final Map<String, Object> metadata;
    private final Map<String, TensorInfo> tensors;
    private final FileChannel channel;
    private final long dataStart;
    private final long fileSize;
    private final Map<String, GgufTensor> mappedTensors = new HashMap<>();

    private GgufReader(
            int version,
            Map<String, Object> metadata,
            Map<String, TensorInfo> tensors,
            FileChannel channel,
            long dataStart,
            long fileSize) {
        this.version = version;
        this.metadata = metadata;
        this.tensors = tensors;
        this.channel = channel;
        this.dataStart = dataStart;
        this.fileSize = fileSize;
    }

    public static GgufReader open(Path path) throws IOException {
        Map<String, Object> meta = new LinkedHashMap<>();
        Map<String, TensorInfo> infos = new LinkedHashMap<>();
        long alignment;
        long dataOffset;
        long fileSize;
        int version;

        try (RandomAccessFile raf = new RandomAccessFile(path.toFile(), "r")) {
            int magic = readI32(raf);
            if (magic != Gguf.MAGIC) {
                throw new IOException("not a GGUF file (magic 0x" + Integer.toHexString(magic) + ")");
            }
            version = readI32(raf);
            if (version < 2 || version > 3) {
                throw new IOException("unsupported GGUF version " + version + " (expected 2 or 3)");
            }
            long tensorCount = readU64(raf);
            long kvCount = readU64(raf);
            if (tensorCount < 0 || tensorCount > 10_000_000L || kvCount < 0 || kvCount > 10_000_000L) {
                throw new IOException("implausible GGUF header counts");
            }

            for (long i = 0; i < kvCount; i++) {
                String key = readString(raf);
                int type = readI32(raf);
                Object value = readValue(raf, type);
                meta.put(key, value);
            }

            for (long i = 0; i < tensorCount; i++) {
                String name = readString(raf);
                int nDims = readI32(raf);
                if (nDims <= 0 || nDims > 16) {
                    throw new IOException("bad tensor rank " + nDims + " for " + name);
                }
                long[] dims = new long[nDims];
                for (int d = 0; d < nDims; d++) {
                    dims[d] = readU64(raf);
                }
                int type = readI32(raf);
                long offset = readU64(raf);
                infos.put(name, new TensorInfo(name, dims, type, offset));
            }

            alignment = 32;
            Object alignObj = meta.get("general.alignment");
            if (alignObj instanceof Number n) {
                alignment = n.longValue();
            }
            if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
                throw new IOException("invalid GGUF alignment " + alignment);
            }
            dataOffset = Gguf.align(raf.getFilePointer(), alignment);
            fileSize = raf.length();
            if (dataOffset > fileSize) {
                throw new IOException("GGUF data offset " + dataOffset + " exceeds file size " + fileSize);
            }
        }

        FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
        return new GgufReader(version, meta, infos, channel, dataOffset, fileSize);
    }

    public int version() {
        return version;
    }

    public Map<String, Object> metadata() {
        return metadata;
    }

    public Map<String, TensorInfo> tensors() {
        return tensors;
    }

    public boolean has(String key) {
        return metadata.containsKey(key);
    }

    public Object raw(String key) {
        return metadata.get(key);
    }

    public String getString(String key) {
        Object v = metadata.get(key);
        return v instanceof String s ? s : null;
    }

    public String getString(String key, String def) {
        String s = getString(key);
        return s == null ? def : s;
    }

    public int getInt(String key, int def) {
        Object v = metadata.get(key);
        return v instanceof Number n ? n.intValue() : def;
    }

    public long getLong(String key, long def) {
        Object v = metadata.get(key);
        return v instanceof Number n ? n.longValue() : def;
    }

    public float getFloat(String key, float def) {
        Object v = metadata.get(key);
        return v instanceof Number n ? n.floatValue() : def;
    }

    public boolean getBoolean(String key, boolean def) {
        Object v = metadata.get(key);
        return v instanceof Boolean b ? b : def;
    }

    public String[] getStringArray(String key) {
        Object v = metadata.get(key);
        return v instanceof String[] a ? a : null;
    }

    public int[] getIntArray(String key) {
        Object v = metadata.get(key);
        if (v instanceof int[] a) {
            return a;
        }
        if (v instanceof long[] a) {
            int[] out = new int[a.length];
            for (int i = 0; i < a.length; i++) {
                out[i] = (int) a[i];
            }
            return out;
        }
        return null;
    }

    public float[] getFloatArray(String key) {
        Object v = metadata.get(key);
        return v instanceof float[] a ? a : null;
    }

    public boolean hasTensor(String name) {
        return tensors.containsKey(name);
    }

    /**
     * Returns null when absent. Each tensor is memory-mapped individually, so
     * GGUF files larger than 2 GiB work as long as no single tensor exceeds
     * the 2 GiB ByteBuffer limit (true for all common 7B/8B/13B models).
     */
    public GgufTensor tensor(String name) {
        GgufTensor cached = mappedTensors.get(name);
        if (cached != null) {
            return cached;
        }
        TensorInfo ti = tensors.get(name);
        if (ti == null) {
            return null;
        }
        int block = Quant.blockSize(ti.type);
        int typeSize = Quant.typeSize(ti.type);
        long count = ti.count();
        if (count % block != 0) {
            throw new IllegalArgumentException("tensor " + name + " has " + count
                    + " values, not a multiple of its block size " + block);
        }
        long bytes = (count / block) * typeSize;
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor " + name + " is larger than 2 GiB ("
                    + bytes + " bytes), which a single Java ByteBuffer cannot map");
        }
        long start = dataStart + ti.offset;
        if (start < 0 || start + bytes > fileSize) {
            throw new IllegalArgumentException("tensor " + name + " lies outside the file (offset "
                    + ti.offset + ", " + bytes + " bytes)");
        }
        ByteBuffer buf;
        try {
            buf = bytes == 0
                    ? ByteBuffer.allocate(0)
                    : channel.map(FileChannel.MapMode.READ_ONLY, start, bytes);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot map tensor " + name, e);
        }
        buf.order(ByteOrder.LITTLE_ENDIAN);
        GgufTensor t = new GgufTensor(ti.name, ti.type, ti.dims, 0, buf);
        mappedTensors.put(name, t);
        return t;
    }

    public GgufTensor requireTensor(String name) {
        GgufTensor t = tensor(name);
        if (t == null) {
            throw new IllegalArgumentException("GGUF is missing tensor " + name);
        }
        return t;
    }

    public long dataStart() {
        return dataStart;
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }

    // ------------------------------------------------------------ primitives

    private static int readU8(RandomAccessFile f) throws IOException {
        int v = f.read();
        if (v < 0) {
            throw new IOException("unexpected EOF");
        }
        return v;
    }

    private static int readI32(RandomAccessFile f) throws IOException {
        int b0 = readU8(f);
        int b1 = readU8(f);
        int b2 = readU8(f);
        int b3 = readU8(f);
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    private static long readU64(RandomAccessFile f) throws IOException {
        long v = 0;
        for (int i = 0; i < 8; i++) {
            v |= (long) readU8(f) << (8 * i);
        }
        return v;
    }

    private static int readU16(RandomAccessFile f) throws IOException {
        return readU8(f) | (readU8(f) << 8);
    }

    private static float readF32(RandomAccessFile f) throws IOException {
        return Float.intBitsToFloat(readI32(f));
    }

    private static double readF64(RandomAccessFile f) throws IOException {
        long bits = 0;
        for (int i = 0; i < 8; i++) {
            bits |= (long) readU8(f) << (8 * i);
        }
        return Double.longBitsToDouble(bits);
    }

    private static String readString(RandomAccessFile f) throws IOException {
        long len = readU64(f);
        if (len < 0 || len > Integer.MAX_VALUE || len > f.length()) {
            throw new IOException("invalid GGUF string length " + len);
        }
        byte[] b = new byte[(int) len];
        f.readFully(b);
        return new String(b, StandardCharsets.UTF_8);
    }

    private static Object readValue(RandomAccessFile f, int type) throws IOException {
        return switch (type) {
            case Gguf.META_UINT8 -> readU8(f);
            case Gguf.META_INT8 -> (int) (byte) readU8(f);
            case Gguf.META_UINT16 -> readU16(f);
            case Gguf.META_INT16 -> (int) (short) readU16(f);
            case Gguf.META_UINT32 -> readI32(f) & 0xFFFFFFFFL;
            case Gguf.META_INT32 -> readI32(f);
            case Gguf.META_FLOAT32 -> readF32(f);
            case Gguf.META_BOOL -> readU8(f) != 0;
            case Gguf.META_STRING -> readString(f);
            case Gguf.META_ARRAY -> readArray(f);
            case Gguf.META_UINT64, Gguf.META_INT64 -> readU64(f);
            case Gguf.META_FLOAT64 -> readF64(f);
            default -> throw new IOException("unknown GGUF metadata type " + type);
        };
    }

    private static Object readArray(RandomAccessFile f) throws IOException {
        int elemType = readI32(f);
        long count = readU64(f);
        if (count < 0 || count > 50_000_000L) {
            throw new IOException("invalid GGUF array length " + count);
        }
        int n = (int) count;
        switch (elemType) {
            case Gguf.META_STRING -> {
                String[] a = new String[n];
                for (int i = 0; i < n; i++) {
                    a[i] = readString(f);
                }
                return a;
            }
            case Gguf.META_INT32 -> {
                int[] a = new int[n];
                for (int i = 0; i < n; i++) {
                    a[i] = readI32(f);
                }
                return a;
            }
            case Gguf.META_UINT32 -> {
                long[] a = new long[n];
                for (int i = 0; i < n; i++) {
                    a[i] = readI32(f) & 0xFFFFFFFFL;
                }
                return a;
            }
            case Gguf.META_FLOAT32 -> {
                float[] a = new float[n];
                for (int i = 0; i < n; i++) {
                    a[i] = readF32(f);
                }
                return a;
            }
            case Gguf.META_FLOAT64 -> {
                double[] a = new double[n];
                for (int i = 0; i < n; i++) {
                    a[i] = readF64(f);
                }
                return a;
            }
            case Gguf.META_BOOL -> {
                boolean[] a = new boolean[n];
                for (int i = 0; i < n; i++) {
                    a[i] = readU8(f) != 0;
                }
                return a;
            }
            case Gguf.META_INT8 -> {
                int[] a = new int[n];
                for (int i = 0; i < n; i++) {
                    a[i] = (byte) readU8(f);
                }
                return a;
            }
            case Gguf.META_UINT8 -> {
                int[] a = new int[n];
                for (int i = 0; i < n; i++) {
                    a[i] = readU8(f);
                }
                return a;
            }
            default -> throw new IOException("unsupported GGUF array element type " + elemType);
        }
    }
}
