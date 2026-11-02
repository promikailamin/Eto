package eto.train;

import eto.nn.ModelConfig;
import eto.nn.TensorParam;
import eto.nn.TinyLlama;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

/** Stores model weights and AdamW state so a later training run can continue. */
final class TrainingCheckpoint {
    private static final int MAGIC = 0x45544F43;
    private static final int VERSION = 1;

    private TrainingCheckpoint() {
    }

    static int load(
            Path path,
            TinyLlama model,
            int batch,
            long seed,
            byte[] dataHash,
            List<TensorParam> params) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            if (in.readInt() != MAGIC || in.readInt() != VERSION) {
                throw new IOException("unsupported training checkpoint: " + path);
            }
            ModelConfig cfg = model.cfg;
            int[] actualConfig = {cfg.vocabSize, cfg.dim, cfg.nLayers, cfg.nHeads, cfg.ffnDim, cfg.ctxLen};
            for (int expected : actualConfig) {
                if (in.readInt() != expected) {
                    throw new IOException("checkpoint model configuration does not match this run");
                }
            }
            if (in.readInt() != batch || in.readLong() != seed) {
                throw new IOException("checkpoint batch size or seed does not match this run");
            }
            int hashLength = in.readInt();
            if (hashLength != dataHash.length) {
                throw new IOException("checkpoint data fingerprint does not match this run");
            }
            byte[] savedHash = in.readNBytes(hashLength);
            if (savedHash.length != hashLength || !java.security.MessageDigest.isEqual(savedHash, dataHash)) {
                throw new IOException("checkpoint was created with different training data");
            }
            int nextStep = in.readInt();
            if (nextStep < 0 || in.readInt() != params.size()) {
                throw new IOException("invalid checkpoint step or parameter count");
            }
            for (TensorParam param : params) {
                if (!param.name().equals(in.readUTF()) || param.length() != in.readInt()) {
                    throw new IOException("checkpoint parameters do not match model");
                }
                readArray(in, param.w());
                readArray(in, param.m());
                readArray(in, param.v());
            }
            if (in.read() != -1) {
                throw new IOException("unexpected trailing data in checkpoint");
            }
            return nextStep;
        }
    }

    static void save(
            Path path,
            TinyLlama model,
            int batch,
            long seed,
            byte[] dataHash,
            int nextStep,
            List<TensorParam> params) throws IOException {
        Path absolute = path.toAbsolutePath();
        Path parent = absolute.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path temporary = absolute.resolveSibling(absolute.getFileName() + ".tmp");
        try {
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(temporary)))) {
                out.writeInt(MAGIC);
                out.writeInt(VERSION);
                ModelConfig cfg = model.cfg;
                out.writeInt(cfg.vocabSize);
                out.writeInt(cfg.dim);
                out.writeInt(cfg.nLayers);
                out.writeInt(cfg.nHeads);
                out.writeInt(cfg.ffnDim);
                out.writeInt(cfg.ctxLen);
                out.writeInt(batch);
                out.writeLong(seed);
                out.writeInt(dataHash.length);
                out.write(dataHash);
                out.writeInt(nextStep);
                out.writeInt(params.size());
                for (TensorParam param : params) {
                    out.writeUTF(param.name());
                    out.writeInt(param.length());
                    writeArray(out, param.w());
                    writeArray(out, param.m());
                    writeArray(out, param.v());
                }
            }
            try {
                Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private static void readArray(DataInputStream in, float[] values) throws IOException {
        for (int i = 0; i < values.length; i++) {
            values[i] = in.readFloat();
        }
    }

    private static void writeArray(DataOutputStream out, float[] values) throws IOException {
        for (float value : values) {
            out.writeFloat(value);
        }
    }
}