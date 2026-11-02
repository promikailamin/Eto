package eto.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** GPT-2's reversible byte-to-unicode mapping. */
public final class ByteUnicode {
    private ByteUnicode() {
    }

    public static final char[] BYTE_TO_CHAR = new char[256];
    public static final Map<Character, Integer> CHAR_TO_BYTE = new HashMap<>(512);

    static {
        List<Integer> bs = new ArrayList<>(256);
        List<Integer> cs = new ArrayList<>(256);
        for (int b = 33; b <= 126; b++) {
            bs.add(b);
            cs.add(b);
        }
        for (int b = 161; b <= 172; b++) {
            bs.add(b);
            cs.add(b);
        }
        for (int b = 174; b <= 255; b++) {
            bs.add(b);
            cs.add(b);
        }
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        for (int i = 0; i < bs.size(); i++) {
            int b = bs.get(i);
            char c = (char) (int) cs.get(i);
            BYTE_TO_CHAR[b] = c;
            CHAR_TO_BYTE.put(c, b);
        }
    }

    public static String fromBytes(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            sb.append(BYTE_TO_CHAR[b & 0xFF]);
        }
        return sb.toString();
    }
}
