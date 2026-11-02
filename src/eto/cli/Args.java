package eto.cli;

import java.util.LinkedHashMap;
import java.util.Map;

/** Tiny --key value parser. Flags may omit the value. */
public final class Args {
    private final Map<String, String> values = new LinkedHashMap<>();

    public static Args parse(String[] argv) {
        Args a = new Args();
        for (int i = 0; i < argv.length; i++) {
            String s = argv[i];
            if (!s.startsWith("--")) {
                throw new IllegalArgumentException("unexpected argument: " + s);
            }
            String key = s.substring(2);
            String value = "true";
            if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
                value = argv[++i];
            }
            a.values.put(key, value);
        }
        return a;
    }

    public boolean has(String key) {
        return values.containsKey(key);
    }

    public String get(String key) {
        return values.get(key);
    }

    public String get(String key, String def) {
        return values.getOrDefault(key, def);
    }

    public int getInt(String key, int def) {
        String s = values.get(key);
        return s == null ? def : Integer.parseInt(s);
    }

    public long getLong(String key, long def) {
        String s = values.get(key);
        return s == null ? def : Long.parseLong(s);
    }

    public float getFloat(String key, float def) {
        String s = values.get(key);
        return s == null ? def : Float.parseFloat(s);
    }

    public boolean flag(String key) {
        String s = values.get(key);
        return s != null && !"false".equalsIgnoreCase(s) && !"0".equals(s);
    }
}
