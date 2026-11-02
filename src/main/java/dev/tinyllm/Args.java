package dev.tinyllm;

import java.util.HashMap;
import java.util.Map;

/** Minimal "--key value" command-line parser. */
final class Args {
    static Map<String, String> parse(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--")) throw new IllegalArgumentException("Unexpected argument: " + args[i]);
            String key = args[i].substring(2);
            if (i + 1 >= args.length || args[i + 1].startsWith("--")) m.put(key, "true");
            else m.put(key, args[++i]);
        }
        return m;
    }
}
