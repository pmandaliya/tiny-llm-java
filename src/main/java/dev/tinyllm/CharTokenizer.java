package dev.tinyllm;

import java.util.Map;
import java.util.TreeSet;
import java.util.HashMap;

/**
 * Character-level tokenizer: every distinct character becomes one token id.
 *
 * Production LLMs use sub-word tokenizers (BPE / SentencePiece) where a token is
 * usually a piece of a word, e.g. "token" + "izer". The idea is identical:
 * text -> list of integer ids -> model -> ids -> text.
 */
public final class CharTokenizer {
    private final char[] idToChar;
    private final Map<Character, Integer> charToId = new HashMap<>();

    public CharTokenizer(char[] vocab) {
        this.idToChar = vocab.clone();
        for (int i = 0; i < vocab.length; i++) charToId.put(vocab[i], i);
    }

    /** Builds the vocabulary from every character that appears in the corpus. */
    public static CharTokenizer fit(String corpus) {
        TreeSet<Character> set = new TreeSet<>();
        for (char c : corpus.toCharArray()) set.add(c);
        char[] vocab = new char[set.size()];
        int i = 0;
        for (char c : set) vocab[i++] = c;
        return new CharTokenizer(vocab);
    }

    public int vocabSize() { return idToChar.length; }

    /** Token id of one character, or -1 if it is not in the vocabulary. */
    public int id(char c) {
        Integer id = charToId.get(c);
        return id == null ? -1 : id;
    }

    public char[] vocab() { return idToChar.clone(); }

    public int[] encode(String text) {
        int[] ids = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            Integer id = charToId.get(text.charAt(i));
            if (id == null) throw new IllegalArgumentException(
                    "Character '" + text.charAt(i) + "' is not in the vocabulary");
            ids[i] = id;
        }
        return ids;
    }

    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder(ids.length);
        for (int id : ids) sb.append(idToChar[id]);
        return sb.toString();
    }

    public String decode(int id) { return String.valueOf(idToChar[id]); }
}
