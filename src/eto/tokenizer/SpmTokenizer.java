package eto.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * SentencePiece BPE tokenizer used by LLaMA-family GGUF files. Mirrors
 * llama.cpp's score-ordered merge queue, ▁-normalization, and &lt;0xNN&gt;
 * byte fallback.
 */
public final class SpmTokenizer implements Tokenizer {
    private record Special(String text, int id) {
    }

    private record Bigram(int left, int right, float score, int size) {
    }

    private static final Comparator<Bigram> BIGRAM_ORDER = (a, b) -> {
        int c = Float.compare(b.score, a.score);
        return c != 0 ? c : Integer.compare(a.left, b.left);
    };

    private final String[] tokens;
    private final int[] types;
    private final float[] scores;
    private final Map<String, Integer> tokenToId;
    private final Map<String, Integer> mergeRank;
    private final boolean rankBpe;
    private final int[] byteTokenIds = new int[256];
    private final Set<Integer> specials = new HashSet<>();
    private final List<Special> specialList = new ArrayList<>();
    private final int bosId;
    private final int eosId;
    private final int unkId;
    private final boolean addBos;
    private final boolean addPrefixSpace;

    public SpmTokenizer(
            String[] tokens,
            int[] types,
            float[] scores,
            int bosId,
            int eosId,
            boolean addBos,
            boolean addPrefixSpace) {
        this(tokens, types, scores, null, bosId, eosId, addBos, addPrefixSpace);
    }

    public SpmTokenizer(
            String[] tokens,
            int[] types,
            float[] scores,
            String[] merges,
            int bosId,
            int eosId,
            boolean addBos,
            boolean addPrefixSpace) {
        this.tokens = tokens;
        this.types = types == null ? inferTypes(tokens) : types;
        this.scores = scores == null ? new float[tokens.length] : scores;
        this.bosId = bosId;
        this.eosId = eosId;
        this.addBos = addBos;
        this.addPrefixSpace = addPrefixSpace;
        this.tokenToId = new HashMap<>(tokens.length * 2);
        for (int i = 0; i < tokens.length; i++) {
            tokenToId.putIfAbsent(tokens[i], i);
        }
        this.mergeRank = new HashMap<>(merges == null ? 16 : merges.length * 2);
        boolean anyScore = false;
        for (float s : this.scores) {
            if (s != 0f) {
                anyScore = true;
                break;
            }
        }
        // Some GGUF exports (notably ones converted from a Hugging Face
        // tokenizer.json) carry real merge ranks but zero SentencePiece scores.
        // In that case use rank-ordered BPE, which is what the reference
        // tokenizer does. Score-based BPE is kept for native SPM files.
        this.rankBpe = !anyScore && merges != null && merges.length > 0;
        if (merges != null) {
            for (int i = 0; i < merges.length; i++) {
                int sp = merges[i].indexOf(' ');
                if (sp <= 0 || sp >= merges[i].length() - 1) {
                    continue;
                }
                mergeRank.putIfAbsent(
                        merges[i].substring(0, sp) + '\u0000' + merges[i].substring(sp + 1), i);
            }
        }
        java.util.Arrays.fill(byteTokenIds, -1);
        int unk = -1;
        for (int i = 0; i < tokens.length; i++) {
            String t = tokens[i];
            int b = parseByteToken(t);
            if (b >= 0) {
                byteTokenIds[b] = i;
            }
            if (t.equals("<unk>")) {
                unk = i;
            }
            if (this.types[i] == 2 || this.types[i] == 3) {
                specials.add(i);
                specialList.add(new Special(t, i));
            }
        }
        this.unkId = unk >= 0 ? unk : 0;
        specialList.sort(Comparator.comparingInt((Special s) -> s.text.length()).reversed());
    }

    private static int[] inferTypes(String[] tokens) {
        int[] t = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            String s = tokens[i];
            if (s.equals("<unk>")) {
                t[i] = 2;
            } else if (s.equals("<s>") || s.equals("</s>")) {
                t[i] = 3;
            } else if (parseByteToken(s) >= 0) {
                t[i] = 6;
            } else {
                t[i] = 1;
            }
        }
        return t;
    }

    private static int parseByteToken(String s) {
        if (s.length() != 6 || s.charAt(0) != '<' || s.charAt(1) != '0' || s.charAt(2) != 'x'
                || s.charAt(5) != '>') {
            return -1;
        }
        int hi = Character.digit(s.charAt(3), 16);
        int lo = Character.digit(s.charAt(4), 16);
        return hi < 0 || lo < 0 ? -1 : (hi << 4) | lo;
    }

    private float score(int id) {
        return id < scores.length ? scores[id] : 0f;
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
                encodeNormal(text.substring(pos), out, addPrefixSpace);
                break;
            }
            if (bestPos > pos) {
                encodeNormal(text.substring(pos, bestPos), out, addPrefixSpace);
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

    /**
     * SentencePiece BPE over already normalized text. This follows llama.cpp's
     * priority queue: the highest-scoring adjacent pair is merged first, and
     * only that occurrence, then the neighboring pairs are re-queued.
     */
    private void encodeNormal(String text, List<Integer> out, boolean prependSpace) {
        if (text.isEmpty()) {
            return;
        }
        String norm = text.replace(' ', '▁');
        if (prependSpace) {
            norm = "▁" + norm;
        }

        List<String> parts = new ArrayList<>();
        for (int i = 0; i < norm.length(); ) {
            int cp = norm.codePointAt(i);
            parts.add(new String(Character.toChars(cp)));
            i += Character.charCount(cp);
        }
        int n = parts.size();

        String[] sym = new String[n];
        int[] len = new int[n];
        int[] prev = new int[n];
        int[] next = new int[n];
        for (int i = 0; i < n; i++) {
            sym[i] = parts.get(i);
            len[i] = sym[i].length();
            prev[i] = i - 1;
            next[i] = i + 1;
        }
        next[n - 1] = -1;

        PriorityQueue<Bigram> queue = new PriorityQueue<>(BIGRAM_ORDER);
        for (int i = 1; i < n; i++) {
            tryAddBigram(i - 1, i, sym, len, queue);
        }
        while (!queue.isEmpty()) {
            Bigram b = queue.poll();
            int l = b.left;
            int r = b.right;
            if (len[l] == 0 || len[r] == 0 || len[l] + len[r] != b.size) {
                continue;
            }
            sym[l] = sym[l] + sym[r];
            len[l] += len[r];
            len[r] = 0;
            next[l] = next[r];
            if (next[r] >= 0) {
                prev[next[r]] = l;
            }
            tryAddBigram(prev[l], l, sym, len, queue);
            tryAddBigram(l, next[l], sym, len, queue);
        }

        for (int i = 0; i != -1; i = next[i]) {
            if (len[i] == 0) {
                continue;
            }
            String part = sym[i];
            Integer id = tokenToId.get(part);
            if (id != null) {
                out.add(id);
                continue;
            }
            byte[] bytes = part.getBytes(StandardCharsets.UTF_8);
            for (byte b : bytes) {
                int bid = byteTokenIds[b & 0xFF];
                out.add(bid >= 0 ? bid : unkId);
            }
        }
    }

    private void tryAddBigram(
            int left, int right, String[] sym, int[] len, PriorityQueue<Bigram> queue) {
        if (left < 0 || right < 0 || len[left] == 0 || len[right] == 0) {
            return;
        }
        String merged = sym[left] + sym[right];
        if (rankBpe) {
            Integer rank = mergeRank.get(sym[left] + '\u0000' + sym[right]);
            if (rank == null) {
                return;
            }
            // The queue prefers larger scores, so a lower merge rank becomes a
            // larger score. Ranks are unique, so ties only occur for the same
            // merge at different positions and are broken by leftmost first.
            queue.add(new Bigram(left, right, -rank, len[left] + len[right]));
            return;
        }
        Integer id = tokenToId.get(merged);
        if (id == null) {
            return;
        }
        queue.add(new Bigram(left, right, score(id), len[left] + len[right]));
    }

    @Override
    public String decode(int[] ids) {
        return new String(decodeToBytes(ids), StandardCharsets.UTF_8);
    }

    public byte[] decodeToBytes(int[] ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(ids.length * 2);
        for (int id : ids) {
            byte[] b = decodeToken(id);
            out.write(b, 0, b.length);
        }
        byte[] bytes = out.toByteArray();
        // SentencePiece's dummy prefix is not part of the text; the reference
        // decoder strips exactly one leading space.
        if (addPrefixSpace && bytes.length > 0 && bytes[0] == ' ') {
            byte[] trimmed = new byte[bytes.length - 1];
            System.arraycopy(bytes, 1, trimmed, 0, trimmed.length);
            return trimmed;
        }
        return bytes;
    }

    @Override
    public byte[] decodeToken(int id) {
        if (id < 0 || id >= tokens.length || specials.contains(id)) {
            return new byte[0];
        }
        int b = parseByteToken(tokens[id]);
        if (b >= 0) {
            return new byte[] {(byte) b};
        }
        String s = tokens[id].replace('▁', ' ');
        return s.getBytes(StandardCharsets.UTF_8);
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
        return "spm";
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
