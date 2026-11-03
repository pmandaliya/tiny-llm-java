package dev.tinyllm;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Pre-training loop: teach the model to predict the next character of a text file.
 *
 *   java -cp target/classes dev.tinyllm.Train --data data/tinyshakespeare.txt --steps 2000
 *   java -cp target/classes dev.tinyllm.Train --data data/ww2.txt --resume checkpoints/ww2.bin --steps 5000
 */
public final class Train {

    public static void main(String[] args) throws Exception {
        Map<String, String> a = Args.parse(args);
        String dataPath = a.getOrDefault("data", "data/tinyshakespeare.txt");
        int steps = Integer.parseInt(a.getOrDefault("steps", "2000"));
        int batch = Integer.parseInt(a.getOrDefault("batch", "16"));
        float maxLr = Float.parseFloat(a.getOrDefault("lr", "3e-3"));
        int warmup = Integer.parseInt(a.getOrDefault("warmup", "100"));
        int evalEvery = Integer.parseInt(a.getOrDefault("eval-every", "250"));
        int evalWindows = Integer.parseInt(a.getOrDefault("eval-windows", "50"));
        int logEvery = Integer.parseInt(a.getOrDefault("log-every", "25"));
        long maxMb = Long.parseLong(a.getOrDefault("max-mb", "-1"));
        String samplePrompt = a.getOrDefault("sample-prompt", "\\n").replace("\\n", "\n");
        int threads = Integer.parseInt(a.getOrDefault("threads",
                String.valueOf(Runtime.getRuntime().availableProcessors())));
        long seed = Long.parseLong(a.getOrDefault("seed", "1337"));
        String outPath = a.getOrDefault("out", "checkpoints/model.bin");
        String resume = a.get("resume");

        // ---- 1. data -------------------------------------------------------
        String text = readCorpus(Path.of(dataPath), maxMb);

        // ---- 2. tokenizer + model (fresh, or continue from a checkpoint) ----
        GPT model;
        CharTokenizer tok;
        if (resume != null) {
            GPT.Loaded l = GPT.load(new File(resume));
            model = l.model();
            tok = l.tokenizer();
            System.out.println("Resumed weights from " + resume + " (optimizer state starts fresh)");
        } else {
            tok = CharTokenizer.fit(text);
            if (tok.vocabSize() > 256) {
                throw new IllegalArgumentException("Corpus has " + tok.vocabSize() + " distinct characters. "
                        + "Run dev.tinyllm.data.TextClean on it first to reduce the vocabulary.");
            }
            Config cfg = new Config();
            cfg.vocabSize = tok.vocabSize();
            cfg.contextLength = Integer.parseInt(a.getOrDefault("ctx", "64"));
            cfg.dModel = Integer.parseInt(a.getOrDefault("dmodel", "128"));
            cfg.nHeads = Integer.parseInt(a.getOrDefault("heads", "4"));
            cfg.nLayers = Integer.parseInt(a.getOrDefault("layers", "4"));
            model = new GPT(cfg, seed);
        }
        Config cfg = model.cfg;

        // Tokens are stored one byte each (vocab <= 256), so a 1 GB corpus needs ~1 GB of RAM, not 4.
        byte[] all = encodeToBytes(tok, text);
        text = null; // let the String be garbage-collected
        int split = (int) (all.length * 0.9);                 // 90% train / 10% validation
        byte[] train = Arrays.copyOfRange(all, 0, split), val = Arrays.copyOfRange(all, split, all.length);
        all = null;
        AdamW opt = new AdamW(model, 0.9f, 0.95f, 0.01f);

        int T = cfg.contextLength;
        System.out.printf("Dataset: %,d chars (%,d train / %,d val tokens)%n", train.length + (long) val.length,
                train.length, val.length);
        System.out.println("Model:   " + cfg);
        System.out.printf("Params:  %,d%n", model.numParams());
        System.out.printf("Training %d steps, batch %d x %d tokens = %,d tokens (%.2f epochs), %d threads%n%n",
                steps, batch, T, (long) steps * batch * T, (double) steps * batch * T / train.length, threads);

        // ---- 3. per-thread workers (data parallelism inside one batch) -----
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        GPT.Workspace[] ws = new GPT.Workspace[threads];
        float[][][] grads = new float[threads][][];
        for (int i = 0; i < threads; i++) { ws[i] = model.newWorkspace(); grads[i] = model.newGrads(); }
        Random rnd = new Random(seed);
        new File(outPath).getAbsoluteFile().getParentFile().mkdirs();
        long t0 = System.currentTimeMillis();
        float bestVal = Float.MAX_VALUE;

        for (int step = 1; step <= steps; step++) {
            // sample a random batch of windows: input = text[i..i+T), target = text[i+1..i+T+1)
            int[][] xs = new int[batch][], ys = new int[batch][];
            for (int b = 0; b < batch; b++) {
                int s = rnd.nextInt(train.length - T - 1);
                xs[b] = window(train, s, T);
                ys[b] = window(train, s + 1, T);
            }
            for (float[][] g : grads) for (float[] arr : g) Arrays.fill(arr, 0f);

            List<Future<Float>> fut = new ArrayList<>();
            for (int w = 0; w < threads; w++) {
                final int wid = w;
                fut.add(pool.submit(() -> {
                    float sum = 0f;
                    for (int b = wid; b < batch; b += threads) {
                        model.forward(xs[b], T, ws[wid]);
                        sum += model.loss(ys[b], T, ws[wid]);
                        model.backward(xs[b], ys[b], T, ws[wid], grads[wid]);
                    }
                    return sum;
                }));
            }
            float loss = 0f;
            for (Future<Float> f : fut) loss += f.get();
            loss /= batch;

            // all-reduce: sum per-thread gradients into grads[0] and average over the batch
            float[][] g = grads[0];
            for (int w = 1; w < threads; w++)
                for (int p = 0; p < g.length; p++) { float[] dst = g[p], src = grads[w][p]; for (int j = 0; j < dst.length; j++) dst[j] += src[j]; }
            for (float[] arr : g) for (int j = 0; j < arr.length; j++) arr[j] /= batch;

            float norm = AdamW.clipGradNorm(g, 1.0f);
            float lr = learningRate(step, steps, warmup, maxLr);
            opt.step(g, lr);

            if (step % logEvery == 0 || step == 1) {
                double sec = (System.currentTimeMillis() - t0) / 1000.0;
                double tps = step * (double) batch * T / sec;
                System.out.printf("step %5d | loss %.4f | lr %.2e | grad-norm %.2f | %.1f tok/s | ETA %.0f min%n",
                        step, loss, lr, norm, tps, (steps - step) * (double) batch * T / tps / 60);
            }
            if (step % evalEvery == 0 || step == steps) {
                float vl = evaluate(model, val, ws[0], evalWindows, new Random(step));
                System.out.printf("---- eval @ %d: val loss %.4f | perplexity %.2f%n", step, vl, Math.exp(vl));
                if (vl < bestVal) {
                    bestVal = vl;
                    model.save(new File(outPath), tok);
                    System.out.println("     saved checkpoint -> " + outPath);
                }
                Sampler s = new Sampler(model, step);
                int[] out = s.generate(tok.encode(samplePrompt), 160, 0.8f, 0, 1f, null);
                System.out.println("     sample: " + tok.decode(out).replace("\n", "\n             "));
            }
        }
        pool.shutdown();
        System.out.printf("%nDone in %.1f min. Best val loss %.4f. Checkpoint: %s%n",
                (System.currentTimeMillis() - t0) / 60000.0, bestVal, outPath);
    }

    /** Reads the training file, optionally only its first maxMb megabytes. */
    static String readCorpus(Path p, long maxMb) throws IOException {
        long size = Files.size(p);
        long limit = maxMb > 0 ? Math.min(size, maxMb * 1_000_000L) : size;
        if (limit > Integer.MAX_VALUE - 16) {
            throw new IllegalArgumentException(String.format(
                    "%s is %.1f GB. Java arrays top out at 2 GB; pass --max-mb 1500 (or less) to train on a slice.",
                    p, size / 1e9));
        }
        byte[] buf = new byte[(int) limit];
        try (InputStream in = Files.newInputStream(p)) {
            int off = 0;
            while (off < buf.length) {
                int n = in.read(buf, off, buf.length - off);
                if (n < 0) break;
                off += n;
            }
        }
        if (limit < size) System.out.printf("Using first %,d of %,d bytes (--max-mb)%n", limit, size);
        return new String(buf, StandardCharsets.UTF_8);
    }

    static byte[] encodeToBytes(CharTokenizer tok, String text) {
        if (tok.vocabSize() > 256) throw new IllegalStateException("vocab > 256 cannot be stored as bytes");
        int[] lut = new int[Character.MAX_VALUE + 1];          // char -> id lookup table (fast for big corpora)
        for (int c = 0; c <= Character.MAX_VALUE; c++) lut[c] = tok.id((char) c);
        byte[] out = new byte[text.length()];
        for (int i = 0; i < text.length(); i++) {
            int id = lut[text.charAt(i)];
            if (id < 0) {
                throw new IllegalArgumentException("Character '" + text.charAt(i) + "' at offset " + i
                        + " is not in the checkpoint's vocabulary. Clean the corpus with TextClean "
                        + "or train a fresh model instead of --resume.");
            }
            out[i] = (byte) id;
        }
        return out;
    }

    static int[] window(byte[] data, int start, int len) {
        int[] w = new int[len];
        for (int i = 0; i < len; i++) w[i] = data[start + i] & 0xFF;
        return w;
    }

    /** Linear warmup then cosine decay down to 10% of the peak learning rate. */
    static float learningRate(int step, int total, int warmup, float maxLr) {
        if (step <= warmup) return maxLr * step / warmup;
        double progress = (double) (step - warmup) / Math.max(1, total - warmup);
        double cos = 0.5 * (1 + Math.cos(Math.PI * progress));
        return (float) (maxLr * (0.1 + 0.9 * cos));
    }

    /** Average loss on held-out text the model never trains on (detects overfitting). */
    static float evaluate(GPT model, byte[] data, GPT.Workspace ws, int windows, Random rnd) {
        int T = model.cfg.contextLength;
        float sum = 0f;
        for (int i = 0; i < windows; i++) {
            int s = rnd.nextInt(data.length - T - 1);
            model.forward(window(data, s, T), T, ws);
            sum += model.loss(window(data, s + 1, T), T, ws);
        }
        return sum / windows;
    }
}
