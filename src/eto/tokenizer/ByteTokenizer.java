package eto.tokenizer;

import java.io.ByteArrayOutputStream;

/**
 * The trainer's tokenizer: 256 raw byte tokens plus explicit BOS/EOS tokens.
 * It produces the token vocabulary embedded into the GGUF file.
 */
public final class ByteTokenizer {
    public static final int BOS_ID = 256;
    public static final int EOS_ID = 257;

    private final String[] tokens;
    private final int[] types;

    private ByteTokenizer(String[] tokens, int[] types) {
        this.tokens = tokens;
        this.types = types;
    }

    public static ByteTokenizer standard() {
        String[] tokens = new String[258];
        int[] types = new int[258];
        for (int b = 0; b < 256; b++) {
            tokens[b] = String.valueOf(ByteUnicode.BYTE_TO_CHAR[b]);
            types[b] = 1; // normal
        }
        tokens[BOS_ID] = "<|bos|>";
        tokens[EOS_ID] = "<|eos|>";
        types[BOS_ID] = 3; // control
        types[EOS_ID] = 3;
        return new ByteTokenizer(tokens, types);
    }

    public String[] ggufTokens() {
        return tokens.clone();
    }

    public int[] ggufTokenTypes() {
        return types.clone();
    }

    public int vocabSize() {
        return tokens.length;
    }

    public int bosId() {
        return BOS_ID;
    }

    public int eosId() {
        return EOS_ID;
    }

    public int[] encodeBytes(byte[] bytes, boolean addBos) {
        int extra = addBos ? 1 : 0;
        int[] ids = new int[bytes.length + extra];
        int p = 0;
        if (addBos) {
            ids[p++] = BOS_ID;
        }
        for (byte b : bytes) {
            ids[p++] = b & 0xFF;
        }
        return ids;
    }

    public byte[] decodeBytes(int[] ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(ids.length);
        for (int id : ids) {
            if (id >= 0 && id < 256) {
                out.write(id);
            }
        }
        return out.toByteArray();
    }
}
