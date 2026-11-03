package dev.tinyllm.data;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Streams an official Wikipedia XML dump and writes clean plain text for training.
 *
 * The dump (https://dumps.wikimedia.org/enwiki/latest/) is bzip2-compressed. The JDK
 * has no bzip2 decoder, so decompress with the standard `bzcat` tool and pipe it in:
 *
 *   # every article (the whole English Wikipedia, ~25 GB download)
 *   bzcat enwiki-latest-pages-articles.xml.bz2 | java -cp target/classes dev.tinyllm.data.WikiDump --out data/wiki.txt
 *
 *   # only articles tagged with a World War II category
 *   bzcat enwiki-latest-pages-articles.xml.bz2 | java -cp target/classes dev.tinyllm.data.WikiDump \
 *       --category-regex "World War II|Second World War" --out data/ww2.txt
 *
 * Nothing is loaded into memory except one article at a time, so it works on any laptop.
 */
public final class WikiDump {

    public static void main(String[] args) throws Exception {
        Map<String, String> a = dev.tinyllm.Args.parse(args);
        String input = a.getOrDefault("input", "-");
        Path out = Path.of(a.getOrDefault("out", "data/wiki.txt"));
        Pattern catRe = a.containsKey("category-regex") ? Pattern.compile(a.get("category-regex"), Pattern.CASE_INSENSITIVE) : null;
        Pattern titleRe = a.containsKey("title-regex") ? Pattern.compile(a.get("title-regex"), Pattern.CASE_INSENSITIVE) : null;
        int minChars = Integer.parseInt(a.getOrDefault("min-chars", "1500"));
        long maxArticles = Long.parseLong(a.getOrDefault("max-articles", "-1"));
        long maxMb = Long.parseLong(a.getOrDefault("max-mb", "-1"));
        boolean keepUnicode = a.containsKey("keep-unicode");

        // Wikipedia dumps are a single multi-GB XML document; lift the JDK's default size safety limits.
        System.setProperty("jdk.xml.totalEntitySizeLimit", "0");
        System.setProperty("jdk.xml.maxGeneralEntitySizeLimit", "0");
        System.setProperty("jdk.xml.entityExpansionLimit", "0");
        System.setProperty("jdk.xml.maxElementDepth", "0");

        InputStream in = input.equals("-") ? System.in : new FileInputStream(input);
        XMLInputFactory f = XMLInputFactory.newInstance();
        f.setProperty(XMLInputFactory.SUPPORT_DTD, false);
        f.setProperty(XMLInputFactory.IS_SUPPORTING_EXTERNAL_ENTITIES, false);
        XMLStreamReader r = f.createXMLStreamReader(new BufferedInputStream(in, 1 << 20), "UTF-8");

        if (out.getParent() != null) Files.createDirectories(out.getParent());
        long seen = 0, kept = 0, bytes = 0, t0 = System.currentTimeMillis();
        try (Writer w = new BufferedWriter(new OutputStreamWriter(Files.newOutputStream(out), StandardCharsets.UTF_8), 1 << 20)) {
            String title = null, ns = null, text = null;
            boolean redirect = false;
            while (r.hasNext()) {
                int ev = r.next();
                if (ev == XMLStreamConstants.START_ELEMENT) {
                    switch (r.getLocalName()) {
                        case "page" -> { title = null; ns = null; text = null; redirect = false; }
                        case "title" -> title = r.getElementText();
                        case "ns" -> ns = r.getElementText();
                        case "redirect" -> redirect = true;
                        case "text" -> text = r.getElementText();
                        default -> { }
                    }
                } else if (ev == XMLStreamConstants.END_ELEMENT && r.getLocalName().equals("page")) {
                    seen++;
                    if (seen % 20000 == 0) {
                        System.err.printf("  scanned %,d pages, kept %,d (%.1f MB) in %.0fs%n",
                                seen, kept, bytes / 1e6, (System.currentTimeMillis() - t0) / 1000.0);
                    }
                    // namespace 0 = real articles (not talk pages, templates, user pages...)
                    if (!"0".equals(ns) || redirect || text == null || title == null) continue;
                    if (titleRe != null && !titleRe.matcher(title).find() && catRe == null) continue;
                    if (catRe != null) {
                        boolean match = titleRe != null && titleRe.matcher(title).find();
                        if (!match) {
                            List<String> cats = WikiText.categories(text);
                            for (String c : cats) if (catRe.matcher(c).find()) { match = true; break; }
                        }
                        if (!match) continue;
                    }
                    String plain = WikiText.toPlainText(text);
                    if (!keepUnicode) plain = TextClean.clean(plain);
                    if (plain.length() < minChars) continue;          // skip stubs and lists
                    if (!keepUnicode) title = TextClean.clean(title);
                    String doc = title + "\n\n" + plain + "\n\n\n";
                    w.write(doc);
                    kept++;
                    bytes += doc.length();
                    if (maxArticles > 0 && kept >= maxArticles) break;
                    if (maxMb > 0 && bytes >= maxMb * 1_000_000L) break;
                }
            }
        }
        System.err.printf("Done: scanned %,d pages, kept %,d articles, %.1f MB -> %s (%.1f min)%n",
                seen, kept, bytes / 1e6, out, (System.currentTimeMillis() - t0) / 60000.0);
    }
}
