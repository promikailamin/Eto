package eto.gguf;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Minimal GGUF v3 writer. All tensors are written as F32, which keeps the
 * trainer simple and lossless. Third-party readers accept this format.
 */
public final class GgufWriter {
    private record Meta(String key, int type, Object value) {
    }

    private record Arr(int elemType, Object data) {
    }

    private static final class Tensor {
        final String name;
        final long[] dims;
        final float[] data;
        long offset;

        Tensor(String name, long[] dims, float[] data) {
            this.name = name;
            this.dims = dims;
            this.data = data;
        }
    }

    private final Path path;
    private final List<Meta> metas = new ArrayList<>();
    private final List<Tensor> tensors = new ArrayList<>();
    private long alignment = Gguf.DEFAULT_ALIGNMENT;
    private boolean written;

    public GgufWriter(Path path) {
        this.path = path;
    }

    public GgufWriter alignment(int alignment) {
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw new IllegalArgumentException("alignment must be a power of two");
        }
        this.alignment = alignment;
        return this;
    }

    public GgufWriter add(String key, String value) {
        metas.add(new Meta(key, Gguf.META_STRING, value));
        return this;
    }

    public GgufWriter add(String key, int value) {
        metas.add(new Meta(key, Gguf.META_UINT32, value));
        return this;
    }

    public GgufWriter add(String key, long value) {
        metas.add(new Meta(key, Gguf.META_UINT64, value));
        return this;
    }

    public GgufWriter add(String key, float value) {
        metas.add(new Meta(key, Gguf.META_FLOAT32, value));
        return this;
    }

    public GgufWriter add(String key, boolean value) {
        metas.add(new Meta(key, Gguf.META_BOOL, value));
        return this;
    }

    public GgufWriter add(String key, String[] values) {
        metas.add(new Meta(key, Gguf.META_ARRAY, new Arr(Gguf.META_STRING, values)));
        return this;
    }

    public GgufWriter add(String key, int[] values) {
        metas.add(new Meta(key, Gguf.META_ARRAY, new Arr(Gguf.META_INT32, values)));
        return this;
    }

    public GgufWriter add(String key, float[] values) {
        metas.add(new Meta(key, Gguf.META_ARRAY, new Arr(Gguf.META_FLOAT32, values)));
        return this;
    }

    public void addTensor(String name, long[] dims, float[] data) {
        long n = 1;
        for (long d : dims) {
            if (d <= 0) {
                throw new IllegalArgumentException("invalid dimension for " + name);
            }
            n = Math.multiplyExact(n, d);
        }
        if (n != data.length) {
            throw new IllegalArgumentException(
                    "tensor " + name + " has " + data.length + " values, dims require " + n);
        }
        tensors.add(new Tensor(name, dims.clone(), data));
    }

    public void write() throws IOException {
        if (written) {
            throw new IllegalStateException("already written");
        }
        written = true;

        long offset = 0;
        for (Tensor t : tensors) {
            offset = Gguf.align(offset, alignment);
            t.offset = offset;
            offset += 4L * t.data.length;
        }

        Buf buf = new Buf();
        buf.u32(Gguf.MAGIC);
        buf.u32(Gguf.VERSION);
        buf.u64(tensors.size());
        buf.u64(metas.size());
        for (Meta m : metas) {
            buf.str(m.key);
            buf.u32(m.type);
            writeValue(buf, m.type, m.value);
        }
        for (Tensor t : tensors) {
            buf.str(t.name);
            buf.u32(t.dims.length);
            for (long d : t.dims) {
                buf.u64(d);
            }
            buf.u32(Gguf.GGML_F32);
            buf.u64(t.offset);
        }

        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        long headerLen = buf.size();
        long dataStart = Gguf.align(headerLen, alignment);
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(path), 1 << 20)) {
            buf.writeTo(out);
            writeZeros(out, dataStart - headerLen);
            long pos = dataStart;
            for (Tensor t : tensors) {
                long target = dataStart + t.offset;
                if (target < pos) {
                    throw new IllegalStateException("tensor overlap");
                }
                writeZeros(out, target - pos);
                pos = target;
                writeFloats(out, t.data);
                pos += 4L * t.data.length;
            }
        }
    }

    private static void writeValue(Buf buf, int type, Object value) {
        switch (type) {
            case Gguf.META_UINT8 -> buf.u8(((Number) value).intValue());
            case Gguf.META_INT8 -> buf.u8(((Number) value).intValue());
            case Gguf.META_UINT16, Gguf.META_INT16 -> buf.u16(((Number) value).intValue());
            case Gguf.META_UINT32, Gguf.META_INT32 -> buf.u32(((Number) value).longValue());
            case Gguf.META_FLOAT32 -> buf.f32(((Number) value).floatValue());
            case Gguf.META_BOOL -> buf.u8(((Boolean) value) ? 1 : 0);
            case Gguf.META_STRING -> buf.str((String) value);
            case Gguf.META_ARRAY -> {
                Arr arr = (Arr) value;
                int n = Array.getLength(arr.data);
                buf.u32(arr.elemType);
                buf.u64(n);
                for (int i = 0; i < n; i++) {
                    writeValue(buf, arr.elemType, Array.get(arr.data, i));
                }
            }
            case Gguf.META_UINT64, Gguf.META_INT64 -> buf.u64(((Number) value).longValue());
            case Gguf.META_FLOAT64 -> buf.u64(Double.doubleToRawLongBits(((Number) value).doubleValue()));
            default -> throw new IllegalArgumentException("unsupported metadata type " + type);
        }
    }

    private static void writeZeros(OutputStream out, long count) throws IOException {
        if (count <= 0) {
            return;
        }
        byte[] zeros = new byte[4096];
        while (count > 0) {
            int n = (int) Math.min(count, zeros.length);
            out.write(zeros, 0, n);
            count -= n;
        }
    }

    private static void writeFloats(OutputStream out, float[] data) throws IOException {
        int chunk = 1 << 16;
        byte[] bytes = new byte[chunk * 4];
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int off = 0; off < data.length; off += chunk) {
            int n = Math.min(chunk, data.length - off);
            bb.clear();
            bb.asFloatBuffer().put(data, off, n);
            out.write(bytes, 0, n * 4);
        }
    }

    private static final class Buf {
        private byte[] a = new byte[4096];
        private int n;

        int size() {
            return n;
        }

        private void ensure(int extra) {
            if (n + extra <= a.length) {
                return;
            }
            int cap = a.length;
            while (cap < n + extra) {
                cap <<= 1;
            }
            byte[] b = new byte[cap];
            System.arraycopy(a, 0, b, 0, n);
            a = b;
        }

        void u8(int v) {
            ensure(1);
            a[n++] = (byte) v;
        }

        void u16(int v) {
            ensure(2);
            a[n++] = (byte) v;
            a[n++] = (byte) (v >>> 8);
        }

        void u32(long v) {
            ensure(4);
            a[n++] = (byte) v;
            a[n++] = (byte) (v >>> 8);
            a[n++] = (byte) (v >>> 16);
            a[n++] = (byte) (v >>> 24);
        }

        void u64(long v) {
            ensure(8);
            for (int i = 0; i < 8; i++) {
                a[n++] = (byte) (v >>> (8 * i));
            }
        }

        void f32(float v) {
            u32(Float.floatToRawIntBits(v));
        }

        void str(String s) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            u64(b.length);
            ensure(b.length);
            System.arraycopy(b, 0, a, n, b.length);
            n += b.length;
        }

        void writeTo(OutputStream out) throws IOException {
            out.write(a, 0, n);
        }
    }
}
