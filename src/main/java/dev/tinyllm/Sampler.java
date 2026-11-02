package dev.tinyllm;

import java.util.Arrays;
import java.util.Random;
import java.util.function.IntConsumer;

/**
 * Turns the model's next-token probabilities into actual text ("inference" / "decoding").
 *
 * Autoregressive generation: predict one token, append it to the input, repeat.
 *  - temperature: < 1 makes the model more confident/boring, > 1 more random/creative
 *  - top-k:       only sample from the k most likely tokens (cuts off nonsense tail)
 *  - top-p:       only sample from the smallest set whose probabilities sum to p ("nucleus" sampling)
 */
public final class Sampler {
    private final GPT model;
    private final GPT.Workspace ws;
    private final Random rnd;

    public Sampler(GPT model, long seed) {
        this.model = model;
        this.ws = model.newWorkspace();
        this.rnd = new Random(seed);
    }

    public int[] generate(int[] prompt, int maxNewTokens, float temperature, int topK, float topP,
                          IntConsumer onToken) {
        int T = model.cfg.contextLength, V = model.cfg.vocabSize;
        int[] seq = Arrays.copyOf(prompt, prompt.length + maxNewTokens);
        int len = prompt.length;
        float[] probs = new float[V];
        for (int step = 0; step < maxNewTokens; step++) {
            // Sliding window: the model can only see the last T tokens (its context window).
            int start = Math.max(0, len - T), n = len - start;
            int[] window = Arrays.copyOfRange(seq, start, len);
            model.forward(window, n, ws);
            int next = sampleFrom(ws.logits, (n - 1) * V, V, temperature, topK, topP, probs);
            seq[len++] = next;
            if (onToken != null) onToken.accept(next);
        }
        return seq;
    }

    int sampleFrom(float[] logits, int off, int V, float temperature, int topK, float topP, float[] probs) {
        if (temperature <= 0f) {                       // greedy decoding
            int best = 0;
            for (int i = 1; i < V; i++) if (logits[off + i] > logits[off + best]) best = i;
            return best;
        }
        GPT.softmax(logits, off, probs, 0, V, temperature);
        Integer[] idx = new Integer[V];
        for (int i = 0; i < V; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> Float.compare(probs[b], probs[a]));
        int keep = topK > 0 ? Math.min(topK, V) : V;
        float cum = 0f;
        int cut = keep;
        for (int i = 0; i < keep; i++) {
            cum += probs[idx[i]];
            if (topP < 1f && cum >= topP) { cut = i + 1; break; }
        }
        float total = 0f;
        for (int i = 0; i < cut; i++) total += probs[idx[i]];
        float r = rnd.nextFloat() * total;
        for (int i = 0; i < cut; i++) {
            r -= probs[idx[i]];
            if (r <= 0f) return idx[i];
        }
        return idx[cut - 1];
    }
}
