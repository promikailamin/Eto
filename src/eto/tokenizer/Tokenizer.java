package eto.tokenizer;

/** Common interface for all pure-Java tokenizers. */
public interface Tokenizer {
    int[] encode(String text, boolean addBos);

    default int[] encode(String text) {
        return encode(text, defaultAddBos());
    }

    /** Decode to text. Invalid byte sequences are replaced by U+FFFD. */
    String decode(int[] ids);

    /** Raw bytes for one token, used for streaming output. Specials return empty. */
    byte[] decodeToken(int id);

    int bosId();

    int eosId();

    int vocabSize();

    String family();

    boolean defaultAddBos();

    boolean isSpecial(int id);

    int[] specialIds();

    String tokenText(int id);
}
