package eto.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * GPT-2 style byte-level BPE tokenizer, as used by many GGUF models and by
 * this project's own byte vocabulary. Pure Java; merges are applied by rank.
 */
public final class Gpt2Tokenizer implements Tokenizer {
    private static final Pattern GPT2_PRETOKEN = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
    private static final Pattern QWEN2_PRETOKEN = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");
    private static final Pattern LLAMA3_PRETOKEN = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+");

    private record Special(String text, int id) {
    }

    private final String[] tokens;
    private final int[] types;
    private final Map<String, Integer> tokenToId;
    private final Map<String, Integer> mergeRank;
    private final int bosId;
    private final int eosId;
    private final boolean addBos;
    private final Pattern pretoken;
    private final Set<Integer> specials = new HashSet<>();
    private final List<Special> specialList = new ArrayList<>();

    public Gpt2Tokenizer(
            String[] tokens,
            int[] types,
            String[] merges,
            int bosId,
            int eosId,
            boolean addBos) {
        this(tokens, types, merges, bosId, eosId, addBos, null);
    }

    public Gpt2Tokenizer(
            String[] tokens,
            int[] types,
            String[] merges,
            int bosId,
            int eosId,
            boolean addBos,
            String pre) {
        this.tokens = tokens;
        this.types = types == null ? new int[tokens.length] : types;
        this.bosId = bosId;
        this.eosId = eosId;
        this.addBos = addBos;
        this.pretoken = pretokenPattern(pre);
        this.tokenToId = new HashMap<>(tokens.length * 2);
        for (int i = 0; i < tokens.length; i++) {
            tokenToId.putIfAbsent(tokens[i], i);
        }
        this.mergeRank = new HashMap<>();
        if (merges != null) {
            for (int i = 0; i < merges.length; i++) {
                String m = merges[i];
                int sp = m.indexOf(' ');
                if (sp <= 0 || sp >= m.length() - 1) {
                    continue;
                }
                mergeRank.putIfAbsent(m.substring(0, sp) + '\u0000' + m.substring(sp + 1), i);
            }
        }
        for (int i = 0; i < tokens.length; i++) {
            boolean special = types != null
                    ? types[i] == 3
                    : (tokens[i].startsWith("<|") || tokens[i].equals("<s>") || tokens[i].equals("</s>"));
            if (special) {
                specials.add(i);
                specialList.add(new Special(tokens[i], i));
            }
        }
        specialList.sort(Comparator.comparingInt((Special s) -> s.text.length()).reversed());
    }

    private static Pattern pretokenPattern(String pre) {
        if (pre == null) {
            return GPT2_PRETOKEN;
        }
        return switch (pre.toLowerCase()) {
            case "qwen2", "qwen3" -> QWEN2_PRETOKEN;
            case "llama3", "llama4" -> LLAMA3_PRETOKEN;
            default -> GPT2_PRETOKEN;
        };
    }

    @Override
    public int[] encode(String text, boolean addBos) {
        List<Integer> out = new ArrayList<>();
        if (addBos) {
            out.add(bosId);
        }
        int pos = 0;
        while (pos < text.length()) {
            int bestPos = -1;
            Special best = null;
            for (Special sp : specialList) {
                int idx = text.indexOf(sp.text, pos);
                if (idx < 0) {
                    continue;
                }
                if (bestPos < 0 || idx < bestPos || (idx == bestPos && sp.text.length() > best.text.length())) {
                    bestPos = idx;
                    best = sp;
                }
            }
            if (bestPos < 0) {
                encodeNormal(text.substring(pos), out);
                break;
            }
            if (bestPos > pos) {
                encodeNormal(text.substring(pos, bestPos), out);
            }
            out.add(best.id);
            pos = bestPos + best.text.length();
        }
        int[] ids = new int[out.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = out.get(i);
        }
        return ids;
    }

    private void encodeNormal(String text, List<Integer> out) {
        if (text.isEmpty()) {
            return;
        }
        Matcher m = pretoken.matcher(text);
        while (m.find()) {
            for (int id : encodePiece(m.group())) {
                out.add(id);
            }
        }
    }

    private int[] encodePiece(String piece) {
        byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        List<String> parts = new ArrayList<>(bytes.length);
        for (byte b : bytes) {
            parts.add(String.valueOf(ByteUnicode.BYTE_TO_CHAR[b & 0xFF]));
        }
        while (parts.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int best = -1;
            for (int i = 0; i + 1 < parts.size(); i++) {
                Integer rank = mergeRank.get(parts.get(i) + '\u0000' + parts.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    best = i;
                }
            }
            if (best < 0) {
                break;
            }
            String a = parts.get(best);
            String b = parts.get(best + 1);
            List<String> merged = new ArrayList<>(parts.size());
            for (int i = 0; i < parts.size(); ) {
                if (i + 1 < parts.size() && parts.get(i).equals(a) && parts.get(i + 1).equals(b)) {
                    merged.add(a + b);
                    i += 2;
                } else {
                    merged.add(parts.get(i));
                    i++;
                }
            }
            parts = merged;
        }
        int[] ids = new int[parts.size()];
        int n = 0;
        for (String part : parts) {
            Integer id = tokenToId.get(part);
            if (id != null) {
                ids[n++] = id;
                continue;
            }
            // Should only happen with an incomplete vocabulary: fall back to bytes.
            for (int i = 0; i < part.length(); i++) {
                Integer bid = tokenToId.get(String.valueOf(part.charAt(i)));
                if (bid == null) {
                    throw new IllegalStateException("tokenizer cannot encode piece: " + part);
                }
                if (n == ids.length) {
                    int[] bigger = new int[ids.length + 4];
                    System.arraycopy(ids, 0, bigger, 0, n);
                    ids = bigger;
                }
                ids[n++] = bid;
            }
        }
        if (n == ids.length) {
            return ids;
        }
        int[] trimmed = new int[n];
        System.arraycopy(ids, 0, trimmed, 0, n);
        return trimmed;
    }

    @Override
    public String decode(int[] ids) {
        byte[] bytes = decodeToBytes(ids);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    public byte[] decodeToBytes(int[] ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(ids.length * 2);
        for (int id : ids) {
            if (id < 0 || id >= tokens.length || specials.contains(id)) {
                continue;
            }
            String s = tokens[id];
            for (int i = 0; i < s.length(); i++) {
                Integer b = ByteUnicode.CHAR_TO_BYTE.get(s.charAt(i));
                if (b != null) {
                    out.write(b);
                } else {
                    byte[] ub = String.valueOf(s.charAt(i)).getBytes(StandardCharsets.UTF_8);
                    out.write(ub, 0, ub.length);
                }
            }
        }
        return out.toByteArray();
    }

    @Override
    public byte[] decodeToken(int id) {
        if (id < 0 || id >= tokens.length || specials.contains(id)) {
            return new byte[0];
        }
        String s = tokens[id];
        ByteArrayOutputStream out = new ByteArrayOutputStream(s.length());
        for (int i = 0; i < s.length(); i++) {
            Integer b = ByteUnicode.CHAR_TO_BYTE.get(s.charAt(i));
            if (b != null) {
                out.write(b);
            } else {
                byte[] ub = String.valueOf(s.charAt(i)).getBytes(StandardCharsets.UTF_8);
                out.write(ub, 0, ub.length);
            }
        }
        return out.toByteArray();
    }

    @Override
    public int bosId() {
        return bosId;
    }

    @Override
    public int eosId() {
        return eosId;
    }

    @Override
    public int vocabSize() {
        return tokens.length;
    }

    @Override
    public String family() {
        return "gpt2";
    }

    @Override
    public boolean defaultAddBos() {
        return addBos;
    }

    @Override
    public boolean isSpecial(int id) {
        return specials.contains(id);
    }

    @Override
    public int[] specialIds() {
        int[] ids = new int[specials.size()];
        int i = 0;
        for (int id : specials) {
            ids[i++] = id;
        }
        return ids;
    }

    @Override
    public String tokenText(int id) {
        return id >= 0 && id < tokens.length ? tokens[id] : "";
    }
}
