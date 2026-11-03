package dev.tinyllm.data;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.Normalizer;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Normalises any text into a small, fixed character set (printable ASCII + newline).
 *
 * Why: our tokenizer is character-level. Raw web text contains thousands of rare
 * Unicode symbols (Cyrillic, CJK, emoji, math). Each one would become its own token
 * that the model sees only a few times and never learns. Real LLMs solve this with
 * byte-level BPE; we solve it by mapping text to ~96 common characters:
 *   "Café – 1944 “D-Day”"  ->  "Cafe - 1944 \"D-Day\""
 *
 * Usage:
 *   java -cp target/classes dev.tinyllm.data.TextClean --in raw.txt --out clean.txt
 *   java -cp target/classes dev.tinyllm.data.TextClean --in my_docs_folder --out clean.txt   (all .txt/.md files)
 */
public final class TextClean {
    private TextClean() {}

    private static final Pattern COMBINING = Pattern.compile("\\p{M}+");
    private static final Pattern SPACES = Pattern.compile("[ \\t]{2,}");
    private static final Pattern TRAILING = Pattern.compile("(?m)[ \\t]+$");
    private static final Pattern NEWLINES = Pattern.compile("\\n{3,}");

    public static String clean(String s) {
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '‘', '’', '‚', '′', '´', '`' -> sb.append('\'');
                case '“', '”', '„', '″', '«', '»' -> sb.append('"');
                case '‐', '‑', '‒', '–', '—', '―', '−' -> sb.append('-');
                case '…' -> sb.append("...");
                case '×' -> sb.append('x');
                case '°' -> sb.append(" degrees");
                case ' ', ' ', ' ', ' ', ' ' -> sb.append(' ');
                case 'ß' -> sb.append("ss");
                case 'Æ' -> sb.append("AE");
                case 'æ' -> sb.append("ae");
                case 'Œ' -> sb.append("OE");
                case 'œ' -> sb.append("oe");
                case 'Ø' -> sb.append('O');
                case 'ø' -> sb.append('o');
                case 'Ł' -> sb.append('L');
                case 'ł' -> sb.append('l');
                case '£' -> sb.append("GBP ");
                case '€' -> sb.append("EUR ");
                case '\r' -> { }
                default -> sb.append(c);
            }
        }
        // "é" -> "e" + combining accent -> "e"
        String t = COMBINING.matcher(Normalizer.normalize(sb, Normalizer.Form.NFKD)).replaceAll("");
        StringBuilder out = new StringBuilder(t.length());
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (c == '\n' || (c >= 32 && c < 127)) out.append(c);
            else if (c == '\t') out.append(' ');
            // everything else (CJK, Cyrillic, emoji, symbols) is dropped
        }
        String r = TRAILING.matcher(SPACES.matcher(out).replaceAll(" ")).replaceAll("");
        return NEWLINES.matcher(r).replaceAll("\n\n");
    }

    public static void main(String[] args) throws IOException {
        Map<String, String> a = dev.tinyllm.Args.parse(args);
        Path in = Path.of(a.getOrDefault("in", "raw.txt"));
        Path out = Path.of(a.getOrDefault("out", "clean.txt"));
        List<Path> files;
        if (Files.isDirectory(in)) {
            try (Stream<Path> s = Files.walk(in)) {
                files = s.filter(Files::isRegularFile)
                        .filter(p -> p.toString().endsWith(".txt") || p.toString().endsWith(".md"))
                        .sorted().toList();
            }
        } else files = List.of(in);

        long before = 0, after = 0;
        Map<Character, Long> counts = new TreeMap<>();
        if (out.getParent() != null) Files.createDirectories(out.getParent());
        try (Writer w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            for (Path f : files) {
                String raw = new String(Files.readAllBytes(f), StandardCharsets.UTF_8);
                String c = clean(raw);
                before += raw.length(); after += c.length();
                for (int i = 0; i < c.length(); i++) counts.merge(c.charAt(i), 1L, Long::sum);
                w.write(c);
                w.write("\n\n");
            }
        }
        System.out.printf("Cleaned %d file(s): %,d -> %,d chars, vocabulary %d characters%n",
                files.size(), before, after, counts.size());
        System.out.println("Wrote " + out);
    }
}
