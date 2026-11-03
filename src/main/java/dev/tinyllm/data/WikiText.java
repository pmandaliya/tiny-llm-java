package dev.tinyllm.data;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Converts MediaWiki markup ("wikitext") into plain prose suitable for training.
 *
 * This is deliberately simple (a few regexes and a brace counter), not a full
 * MediaWiki parser. It removes infoboxes, tables, references, images and
 * citation templates, and keeps the readable text of links. Good enough for a
 * language model; tools like WikiExtractor do the same job more thoroughly.
 */
public final class WikiText {
    private WikiText() {}

    private static final Pattern CATEGORY = Pattern.compile("\\[\\[\\s*Category\\s*:\\s*([^\\]|]+)", Pattern.CASE_INSENSITIVE);
    private static final Pattern COMMENT = Pattern.compile("<!--.*?-->", Pattern.DOTALL);
    private static final Pattern REF_PAIR = Pattern.compile("<ref[^>/]*>.*?</ref\\s*>", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
    private static final Pattern REF_SELF = Pattern.compile("<ref[^>]*/\\s*>", Pattern.CASE_INSENSITIVE);
    private static final Pattern DROP_TAG_BLOCK = Pattern.compile(
            "<(gallery|math|chem|score|timeline|syntaxhighlight|source|pre|graph|mapframe|imagemap)[^>]*>.*?</\\1\\s*>",
            Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
    private static final Pattern HTML_TAG = Pattern.compile("</?[a-zA-Z][^>]*>");
    private static final Pattern EXT_LINK_LABEL = Pattern.compile("\\[(?:https?:)?//[^\\s\\]]+\\s+([^\\]]+)\\]");
    private static final Pattern EXT_LINK_BARE = Pattern.compile("\\[(?:https?:)?//[^\\]]*\\]");
    private static final Pattern BOLD_ITALIC = Pattern.compile("'{2,5}");
    private static final Pattern HEADING = Pattern.compile("(?m)^\\s*(={2,6})\\s*(.*?)\\s*\\1\\s*$");
    private static final Pattern LIST_MARK = Pattern.compile("(?m)^[*#:;]+\\s*");
    private static final Pattern MAGIC = Pattern.compile("__[A-Z]+__");
    private static final Pattern MULTI_SPACE = Pattern.compile("[ \\t\\u00A0]{2,}");
    private static final Pattern SPACE_PUNCT = Pattern.compile(" +([,.;:)])");
    private static final Pattern EMPTY_PARENS = Pattern.compile("\\(\\s*[,;]?\\s*\\)");
    private static final Pattern PAREN_LEADING_PUNCT = Pattern.compile("\\(\\s*[,;]\\s*");
    private static final Pattern MANY_NEWLINES = Pattern.compile("\\n{3,}");

    /** Sections after which an article is just references, so we stop there. */
    private static final Pattern TAIL_SECTION = Pattern.compile(
            "(?im)^\\s*==+\\s*(References|Notes|Citations|Footnotes|Sources|Bibliography|Further reading|"
            + "External links|See also|Notes and references|Works cited|Explanatory notes)\\s*==+\\s*$");

    /** Category names listed in the article, e.g. "Battles of World War II involving Japan". */
    public static List<String> categories(String wikitext) {
        List<String> out = new ArrayList<>();
        Matcher m = CATEGORY.matcher(wikitext);
        while (m.find()) out.add(m.group(1).trim());
        return out;
    }

    public static String toPlainText(String w) {
        Matcher tail = TAIL_SECTION.matcher(w);
        if (tail.find()) w = w.substring(0, tail.start());

        w = COMMENT.matcher(w).replaceAll("");
        w = REF_PAIR.matcher(w).replaceAll("");
        w = REF_SELF.matcher(w).replaceAll("");
        w = DROP_TAG_BLOCK.matcher(w).replaceAll("");
        w = removeNested(w, "{{", "}}");            // templates: infoboxes, citations, navboxes
        w = removeNested(w, "{|", "|}");            // tables
        w = links(w);                                // [[target|label]] -> label, drop File:/Category:
        w = EXT_LINK_LABEL.matcher(w).replaceAll("$1");
        w = EXT_LINK_BARE.matcher(w).replaceAll("");
        w = HTML_TAG.matcher(w).replaceAll("");
        w = decodeEntities(w);
        w = BOLD_ITALIC.matcher(w).replaceAll("");
        w = HEADING.matcher(w).replaceAll("\n$2\n");
        w = LIST_MARK.matcher(w).replaceAll("");
        w = MAGIC.matcher(w).replaceAll("");

        StringBuilder sb = new StringBuilder(w.length());
        for (String line : w.split("\n", -1)) {
            String t = line.strip();
            if (t.startsWith("|") || t.startsWith("!") || t.startsWith("{") || t.startsWith("}")) continue; // table debris
            t = MULTI_SPACE.matcher(t).replaceAll(" ");
            t = EMPTY_PARENS.matcher(t).replaceAll("");
            t = PAREN_LEADING_PUNCT.matcher(t).replaceAll("(");   // "(, 15 September 1890" -> "(15 September 1890"
            t = SPACE_PUNCT.matcher(t).replaceAll("$1");
            sb.append(t).append('\n');
        }
        return MANY_NEWLINES.matcher(sb.toString()).replaceAll("\n\n").strip();
    }

    /** Removes balanced blocks such as {{ ... {{ ... }} ... }} that regexes cannot handle. */
    static String removeNested(String s, String open, String close) {
        StringBuilder out = new StringBuilder(s.length());
        int depth = 0, i = 0;
        while (i < s.length()) {
            if (s.startsWith(open, i)) { depth++; i += open.length(); }
            else if (depth > 0 && s.startsWith(close, i)) { depth--; i += close.length(); }
            else { if (depth == 0) out.append(s.charAt(i)); i++; }
        }
        return out.toString();
    }

    /** Handles [[...]] links, including nested links inside image captions. */
    static String links(String s) {
        StringBuilder out = new StringBuilder(s.length());
        int i = 0;
        while (i < s.length()) {
            if (!s.startsWith("[[", i)) { out.append(s.charAt(i++)); continue; }
            int depth = 0, j = i;
            while (j < s.length()) {                  // find the matching ]]
                if (s.startsWith("[[", j)) { depth++; j += 2; }
                else if (s.startsWith("]]", j)) { depth--; j += 2; if (depth == 0) break; }
                else j++;
            }
            String inner = s.substring(i + 2, Math.max(i + 2, j - 2));
            String lower = inner.stripLeading().toLowerCase();
            boolean drop = lower.startsWith("file:") || lower.startsWith("image:") || lower.startsWith("category:")
                    || lower.startsWith("media:") || lower.matches("^[a-z]{2,3}(-[a-z]+)?:.*"); // interlanguage links
            if (!drop) {
                int bar = inner.lastIndexOf('|');
                String label = bar >= 0 ? inner.substring(bar + 1) : inner;
                if (label.isBlank() && bar >= 0) label = inner.substring(0, bar).replaceAll("\\s*\\(.*\\)$", "");
                out.append(label);
            }
            i = j;
        }
        return out.toString();
    }

    static String decodeEntities(String s) {
        return s.replace("&nbsp;", " ").replace("&ndash;", "-").replace("&mdash;", "-")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"")
                .replace("&apos;", "'").replace("&minus;", "-").replace("&thinsp;", " ")
                .replace("&amp;", "&");
    }
}
