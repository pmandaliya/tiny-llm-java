package dev.tinyllm.data;

import java.io.IOException;
import java.io.Writer;
import java.net.ProxySelector;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Downloads every article in a Wikipedia category tree through the official MediaWiki API.
 *
 *   java -cp target/classes dev.tinyllm.data.WikiFetch \
 *       --category "World War II" --depth 3 \
 *       --follow-regex "World War II|Second World War" \
 *       --user-agent "tiny-llm-java/1.0 (you@example.com)" \
 *       --out data/ww2.txt
 *
 * Steps:
 *  1. Breadth-first crawl of the category and its sub-categories (up to --depth levels),
 *     only descending into sub-categories whose name matches --follow-regex. Without that
 *     filter, category trees drift off-topic fast (WWII -> Aircraft -> Boeing -> Airports...).
 *  2. Fetch the wikitext of the collected articles, 50 per request.
 *  3. Convert to plain text (WikiText) and normalise characters (TextClean).
 *
 * Be polite: Wikimedia asks for a descriptive User-Agent with contact info and serial
 * requests. This tool sends one request at a time, honours Retry-After and maxlag.
 */
public final class WikiFetch {

    private final HttpClient http = HttpClient.newBuilder()
            .proxy(ProxySelector.getDefault())
            .connectTimeout(Duration.ofSeconds(20))
            .build();
    private final String api;
    private final String userAgent;
    private final long delayMs;

    WikiFetch(String api, String userAgent, long delayMs) {
        this.api = api; this.userAgent = userAgent; this.delayMs = delayMs;
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> a = dev.tinyllm.Args.parse(args);
        String root = a.getOrDefault("category", "World War II").replaceFirst("(?i)^Category:", "");
        int depth = Integer.parseInt(a.getOrDefault("depth", "2"));
        Pattern follow = Pattern.compile(a.getOrDefault("follow-regex", ".*"), Pattern.CASE_INSENSITIVE);
        String ua = a.getOrDefault("user-agent", "tiny-llm-java/1.0 (https://github.com/pmandaliya/tiny-llm-java)");
        String lang = a.getOrDefault("lang", "en");
        Path out = Path.of(a.getOrDefault("out", "data/wiki_category.txt"));
        int minChars = Integer.parseInt(a.getOrDefault("min-chars", "1500"));
        int maxArticles = Integer.parseInt(a.getOrDefault("max-articles", "-1"));
        long delay = Long.parseLong(a.getOrDefault("delay-ms", "200"));
        boolean keepUnicode = a.containsKey("keep-unicode");

        WikiFetch wf = new WikiFetch("https://" + lang + ".wikipedia.org/w/api.php", ua, delay);

        // ---- 1. crawl the category tree --------------------------------------
        Map<Long, String> pages = new LinkedHashMap<>();
        Set<String> visited = new HashSet<>();
        ArrayDeque<String[]> queue = new ArrayDeque<>();
        queue.add(new String[]{"Category:" + root, "0"});
        visited.add("Category:" + root);
        while (!queue.isEmpty()) {
            String[] item = queue.poll();
            String cat = item[0];
            int d = Integer.parseInt(item[1]);
            int before = pages.size(), subcats = 0;
            String cont = null;
            do {
                Map<String, String> q = new LinkedHashMap<>();
                q.put("action", "query"); q.put("list", "categorymembers"); q.put("cmtitle", cat);
                q.put("cmtype", "page|subcat"); q.put("cmlimit", "max");
                if (cont != null) q.put("cmcontinue", cont);
                Object resp = wf.call(q);
                for (Object m : Json.arr(Json.obj(resp, "query"), "categorymembers")) {
                    String ns = Json.str(m, "ns"), title = Json.str(m, "title");
                    if ("0".equals(ns)) pages.putIfAbsent(Long.parseLong(Json.str(m, "pageid")), title);
                    else if ("14".equals(ns) && d < depth && follow.matcher(title).find() && visited.add(title)) {
                        queue.add(new String[]{title, String.valueOf(d + 1)});
                        subcats++;
                    }
                }
                cont = Json.str(Json.obj(resp, "continue"), "cmcontinue");
            } while (cont != null);
            System.err.printf("  [depth %d] %-70s +%d articles, +%d subcategories (total %,d articles, %,d queued)%n",
                    d, cat, pages.size() - before, subcats, pages.size(), queue.size());
            if (maxArticles > 0 && pages.size() >= maxArticles) break;
        }
        System.err.printf("Found %,d articles in %,d categories. Downloading text...%n", pages.size(), visited.size());

        // ---- 2. download wikitext in batches of 50 and write plain text ------
        List<Long> ids = new ArrayList<>(pages.keySet());
        if (maxArticles > 0 && ids.size() > maxArticles) ids = ids.subList(0, maxArticles);
        if (out.getParent() != null) Files.createDirectories(out.getParent());
        long kept = 0, bytes = 0;
        try (Writer w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            for (int start = 0; start < ids.size(); start += 50) {
                List<Long> chunk = ids.subList(start, Math.min(ids.size(), start + 50));
                StringJoiner sj = new StringJoiner("|");
                for (long id : chunk) sj.add(Long.toString(id));
                Map<String, String> q = new LinkedHashMap<>();
                q.put("action", "query"); q.put("prop", "revisions"); q.put("rvprop", "content");
                q.put("rvslots", "main"); q.put("pageids", sj.toString());
                Object resp = wf.call(q);
                for (Object p : Json.arr(Json.obj(resp, "query"), "pages")) {
                    List<Object> revs = Json.arr(p, "revisions");
                    if (revs.isEmpty()) continue;
                    String wikitext = Json.str(Json.obj(Json.obj(revs.get(0), "slots"), "main"), "content");
                    if (wikitext == null || wikitext.regionMatches(true, 0, "#REDIRECT", 0, 9)) continue;
                    String title = Json.str(p, "title");
                    String plain = WikiText.toPlainText(wikitext);
                    if (!keepUnicode) { plain = TextClean.clean(plain); title = TextClean.clean(title); }
                    if (plain.length() < minChars) continue;
                    String doc = title + "\n\n" + plain + "\n\n\n";
                    w.write(doc);
                    kept++;
                    bytes += doc.length();
                }
                System.err.printf("  downloaded %,d / %,d articles, kept %,d (%.1f MB)%n",
                        Math.min(ids.size(), start + 50), ids.size(), kept, bytes / 1e6);
            }
        }
        System.err.printf("Done: %,d articles, %.1f MB -> %s%n", kept, bytes / 1e6, out);
    }

    /** One API request with retries for rate limiting (429), server lag (maxlag) and 5xx errors. */
    Object call(Map<String, String> params) throws IOException, InterruptedException {
        StringBuilder url = new StringBuilder(api).append("?format=json&formatversion=2&maxlag=5");
        for (var e : params.entrySet()) {
            url.append('&').append(e.getKey()).append('=').append(URLEncoder.encode(e.getValue(), StandardCharsets.UTF_8));
        }
        long backoff = 2000;
        for (int attempt = 1; ; attempt++) {
            Thread.sleep(delayMs);
            HttpRequest req = HttpRequest.newBuilder(URI.create(url.toString()))
                    .header("User-Agent", userAgent)
                    .timeout(Duration.ofSeconds(60))
                    .GET().build();
            HttpResponse<String> r = http.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            boolean retry = r.statusCode() == 429 || r.statusCode() >= 500;
            Object body = null;
            if (!retry) {
                body = Json.parse(r.body());
                Map<String, Object> err = Json.obj(body, "error");
                if (err != null) {
                    if ("maxlag".equals(Json.str(err, "code"))) retry = true;
                    else throw new IOException("API error: " + Json.str(err, "code") + " - " + Json.str(err, "info"));
                }
            }
            if (!retry) return body;
            if (attempt >= 8) throw new IOException("Giving up after " + attempt + " attempts, HTTP " + r.statusCode());
            long wait = backoff;
            Optional<String> ra = r.headers().firstValue("Retry-After");
            if (ra.isPresent() && ra.get().trim().matches("\\d+")) wait = Math.max(1000, Long.parseLong(ra.get().trim()) * 1000);
            System.err.printf("  server busy (HTTP %d), retrying in %ds...%n", r.statusCode(), wait / 1000);
            Thread.sleep(wait);
            backoff = Math.min(backoff * 2, 60_000);
        }
    }
}
