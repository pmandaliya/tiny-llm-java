package dev.tinyllm;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A decoder-only Transformer (the "GPT" architecture) written in plain Java.
 *
 * Data flow for one sequence of n tokens:
 *
 *   token ids --> token embedding + positional embedding      (n x C)
 *            --> [ TransformerBlock ] x nLayers                (n x C)
 *                    x = x + Attention(LayerNorm(x))   // "residual connection"
 *                    x = x + MLP(LayerNorm(x))
 *            --> final LayerNorm --> LM head (n x C) @ (C x V) --> logits (n x V)
 *            --> softmax --> probability of every possible next token
 *
 * Everything (forward pass AND backpropagation) is hand-written so you can read
 * exactly what a deep-learning framework would otherwise hide.
 */
public final class GPT {

    public final Config cfg;
    /** All learnable weights. Index constants below say which array is which. */
    public final float[][] params;
    public final String[] names;
    /** true for 2-D weight matrices (these get weight decay), false for biases / norms. */
    public final boolean[] decay;

    // ---- parameter layout -------------------------------------------------
    static final int WTE = 0, WPE = 1;
    static final int PER_LAYER = 12;
    static final int LN1_G = 0, LN1_B = 1, W_QKV = 2, B_QKV = 3, W_O = 4, B_O = 5,
                     LN2_G = 6, LN2_B = 7, W_1 = 8, B_1 = 9, W_2 = 10, B_2 = 11;
    final int lnfG, lnfB, wLm;

    int p(int layer, int which) { return 2 + layer * PER_LAYER + which; }

    public GPT(Config cfg, long seed) {
        this.cfg = cfg;
        int V = cfg.vocabSize, T = cfg.contextLength, C = cfg.dModel, F = cfg.ffnDim(), L = cfg.nLayers;
        if (C % cfg.nHeads != 0) throw new IllegalArgumentException("dModel must be divisible by nHeads");
        int count = 2 + L * PER_LAYER + 3;
        params = new float[count][];
        names = new String[count];
        decay = new boolean[count];
        lnfG = count - 3; lnfB = count - 2; wLm = count - 1;

        Random rnd = new Random(seed);
        float std = 0.02f;
        // GPT-2 trick: scale init of layers that write into the residual stream.
        float resStd = (float) (0.02 / Math.sqrt(2.0 * L));

        set(WTE, "token_embedding", gaussian(V * C, std, rnd), true);
        set(WPE, "position_embedding", gaussian(T * C, std, rnd), true);
        for (int l = 0; l < L; l++) {
            String n = "block" + l + ".";
            set(p(l, LN1_G), n + "ln1.gamma", ones(C), false);
            set(p(l, LN1_B), n + "ln1.beta", new float[C], false);
            set(p(l, W_QKV), n + "attn.W_qkv", gaussian(C * 3 * C, std, rnd), true);
            set(p(l, B_QKV), n + "attn.b_qkv", new float[3 * C], false);
            set(p(l, W_O), n + "attn.W_out", gaussian(C * C, resStd, rnd), true);
            set(p(l, B_O), n + "attn.b_out", new float[C], false);
            set(p(l, LN2_G), n + "ln2.gamma", ones(C), false);
            set(p(l, LN2_B), n + "ln2.beta", new float[C], false);
            set(p(l, W_1), n + "mlp.W_up", gaussian(C * F, std, rnd), true);
            set(p(l, B_1), n + "mlp.b_up", new float[F], false);
            set(p(l, W_2), n + "mlp.W_down", gaussian(F * C, resStd, rnd), true);
            set(p(l, B_2), n + "mlp.b_down", new float[C], false);
        }
        set(lnfG, "ln_final.gamma", ones(C), false);
        set(lnfB, "ln_final.beta", new float[C], false);
        set(wLm, "lm_head", gaussian(C * V, std, rnd), true);
    }

    private void set(int i, String name, float[] w, boolean d) { params[i] = w; names[i] = name; decay[i] = d; }

    public long numParams() {
        long n = 0;
        for (float[] a : params) n += a.length;
        return n;
    }

    /** Fresh, zeroed gradient buffers with the same shapes as {@link #params}. */
    public float[][] newGrads() {
        float[][] g = new float[params.length][];
        for (int i = 0; i < params.length; i++) g[i] = new float[params[i].length];
        return g;
    }

    // =======================================================================
    //  Activation storage for one sequence (needed for backpropagation)
    // =======================================================================
    public final class Workspace {
        final int T = cfg.contextLength, C = cfg.dModel, F = cfg.ffnDim(), V = cfg.vocabSize,
                  H = cfg.nHeads, L = cfg.nLayers;
        final float[][] x = new float[L + 1][T * C];   // residual stream entering each block
        final float[][] xMid = new float[L][T * C];    // residual after attention
        final float[][] ln1 = new float[L][T * C], ln1Mean = new float[L][T], ln1Rstd = new float[L][T];
        final float[][] qkv = new float[L][T * 3 * C];
        final float[][] att = new float[L][H * T * T]; // attention weights (after softmax)
        final float[][] y = new float[L][T * C];       // attention output before projection
        final float[][] ln2 = new float[L][T * C], ln2Mean = new float[L][T], ln2Rstd = new float[L][T];
        final float[][] hPre = new float[L][T * F], hAct = new float[L][T * F];
        final float[] lnf = new float[T * C], lnfMean = new float[T], lnfRstd = new float[T];
        final float[] logits = new float[T * V];
        final float[] probs = new float[T * V];
        // backward scratch
        final float[] dx = new float[T * C], dTmp = new float[T * C], dLn = new float[T * C];
        final float[] dQkv = new float[T * 3 * C], dH = new float[T * F], dLogits = new float[T * V];
        final float[] dAttRow = new float[T];
    }

    public Workspace newWorkspace() { return new Workspace(); }

    // =======================================================================
    //  Forward pass
    // =======================================================================

    /** Runs the network on tokens[0..n). Afterwards ws.logits holds n x V scores. */
    public void forward(int[] tokens, int n, Workspace ws) {
        int C = cfg.dModel, F = cfg.ffnDim(), V = cfg.vocabSize;
        if (n > cfg.contextLength) throw new IllegalArgumentException("sequence longer than context window");

        // 1) Embedding lookup: meaning of the token + where it sits in the sequence.
        float[] wte = params[WTE], wpe = params[WPE], x0 = ws.x[0];
        for (int t = 0; t < n; t++) {
            int tok = tokens[t];
            for (int c = 0; c < C; c++) x0[t * C + c] = wte[tok * C + c] + wpe[t * C + c];
        }

        // 2) Transformer blocks.
        for (int l = 0; l < cfg.nLayers; l++) {
            float[] x = ws.x[l], xm = ws.xMid[l], xOut = ws.x[l + 1];

            // --- attention sub-layer (tokens talk to each other) ---
            layerNorm(ws.ln1[l], ws.ln1Mean[l], ws.ln1Rstd[l], x, params[p(l, LN1_G)], params[p(l, LN1_B)], n, C);
            matmul(ws.qkv[l], ws.ln1[l], params[p(l, W_QKV)], params[p(l, B_QKV)], n, C, 3 * C);
            attention(ws.y[l], ws.att[l], ws.qkv[l], n);
            matmul(xm, ws.y[l], params[p(l, W_O)], params[p(l, B_O)], n, C, C);
            for (int i = 0; i < n * C; i++) xm[i] += x[i];                     // residual

            // --- feed-forward sub-layer (each token "thinks" on its own) ---
            layerNorm(ws.ln2[l], ws.ln2Mean[l], ws.ln2Rstd[l], xm, params[p(l, LN2_G)], params[p(l, LN2_B)], n, C);
            matmul(ws.hPre[l], ws.ln2[l], params[p(l, W_1)], params[p(l, B_1)], n, C, F);
            float[] hp = ws.hPre[l], ha = ws.hAct[l];
            for (int i = 0; i < n * F; i++) ha[i] = gelu(hp[i]);
            matmul(xOut, ha, params[p(l, W_2)], params[p(l, B_2)], n, F, C);
            for (int i = 0; i < n * C; i++) xOut[i] += xm[i];                  // residual
        }

        // 3) Final norm + language-model head -> one score (logit) per vocabulary entry.
        layerNorm(ws.lnf, ws.lnfMean, ws.lnfRstd, ws.x[cfg.nLayers], params[lnfG], params[lnfB], n, C);
        matmul(ws.logits, ws.lnf, params[wLm], null, n, C, V);
    }

    /** Average cross-entropy loss of predicting targets[t] from position t. Also fills ws.probs. */
    public float loss(int[] targets, int n, Workspace ws) {
        int V = cfg.vocabSize;
        double total = 0;
        for (int t = 0; t < n; t++) {
            softmax(ws.logits, t * V, ws.probs, t * V, V, 1f);
            total += -Math.log(Math.max(ws.probs[t * V + targets[t]], 1e-12f));
        }
        return (float) (total / n);
    }

    // =======================================================================
    //  Backward pass (backpropagation) - accumulates into grads
    // =======================================================================

    /** Must be called right after forward()+loss() on the same workspace. */
    public void backward(int[] tokens, int[] targets, int n, Workspace ws, float[][] grads) {
        int C = cfg.dModel, F = cfg.ffnDim(), V = cfg.vocabSize;

        // dLoss/dLogits for softmax + cross-entropy is simply (probs - one_hot) / n.
        float[] dLogits = ws.dLogits;
        for (int t = 0; t < n; t++) {
            for (int v = 0; v < V; v++) dLogits[t * V + v] = ws.probs[t * V + v] / n;
            dLogits[t * V + targets[t]] -= 1f / n;
        }
        float[] dx = ws.dx;
        java.util.Arrays.fill(ws.dLn, 0, n * C, 0f);
        matmulBackward(ws.dLn, grads[wLm], null, dLogits, ws.lnf, params[wLm], n, C, V);
        java.util.Arrays.fill(dx, 0, n * C, 0f);
        layerNormBackward(dx, grads[lnfG], grads[lnfB], ws.dLn, ws.x[cfg.nLayers], ws.lnfMean, ws.lnfRstd,
                params[lnfG], n, C);

        for (int l = cfg.nLayers - 1; l >= 0; l--) {
            // ---- MLP sub-layer. dx currently = dLoss/d(x[l+1]); residual passes it straight to xMid.
            java.util.Arrays.fill(ws.dH, 0, n * F, 0f);
            matmulBackward(ws.dH, grads[p(l, W_2)], grads[p(l, B_2)], dx, ws.hAct[l], params[p(l, W_2)], n, F, C);
            float[] hp = ws.hPre[l];
            for (int i = 0; i < n * F; i++) ws.dH[i] *= geluGrad(hp[i]);
            java.util.Arrays.fill(ws.dLn, 0, n * C, 0f);
            matmulBackward(ws.dLn, grads[p(l, W_1)], grads[p(l, B_1)], ws.dH, ws.ln2[l], params[p(l, W_1)], n, C, F);
            layerNormBackward(dx, grads[p(l, LN2_G)], grads[p(l, LN2_B)], ws.dLn, ws.xMid[l], ws.ln2Mean[l],
                    ws.ln2Rstd[l], params[p(l, LN2_G)], n, C);   // adds into dx (residual)

            // ---- Attention sub-layer. dx = dLoss/d(xMid[l]).
            java.util.Arrays.fill(ws.dTmp, 0, n * C, 0f);
            matmulBackward(ws.dTmp, grads[p(l, W_O)], grads[p(l, B_O)], dx, ws.y[l], params[p(l, W_O)], n, C, C);
            java.util.Arrays.fill(ws.dQkv, 0, n * 3 * C, 0f);
            attentionBackward(ws.dQkv, ws.dTmp, ws.att[l], ws.qkv[l], n, ws.dAttRow);
            java.util.Arrays.fill(ws.dLn, 0, n * C, 0f);
            matmulBackward(ws.dLn, grads[p(l, W_QKV)], grads[p(l, B_QKV)], ws.dQkv, ws.ln1[l], params[p(l, W_QKV)],
                    n, C, 3 * C);
            layerNormBackward(dx, grads[p(l, LN1_G)], grads[p(l, LN1_B)], ws.dLn, ws.x[l], ws.ln1Mean[l],
                    ws.ln1Rstd[l], params[p(l, LN1_G)], n, C);
        }

        // Embeddings: scatter the gradient back to the table rows that were looked up.
        float[] gTe = grads[WTE], gPe = grads[WPE];
        for (int t = 0; t < n; t++) {
            int tok = tokens[t];
            for (int c = 0; c < C; c++) {
                gTe[tok * C + c] += dx[t * C + c];
                gPe[t * C + c] += dx[t * C + c];
            }
        }
    }

    // =======================================================================
    //  Building blocks
    // =======================================================================

    /** out[n x O] = in[n x I] @ W[I x O] + b */
    static void matmul(float[] out, float[] in, float[] W, float[] b, int n, int I, int O) {
        for (int t = 0; t < n; t++) {
            int ro = t * O, ri = t * I;
            if (b != null) System.arraycopy(b, 0, out, ro, O);
            else java.util.Arrays.fill(out, ro, ro + O, 0f);
            for (int i = 0; i < I; i++) {
                float a = in[ri + i];
                int rw = i * O;
                for (int o = 0; o < O; o++) out[ro + o] += a * W[rw + o];
            }
        }
    }

    /** Accumulates dIn += dOut @ W^T, dW += in^T @ dOut, dB += sum(dOut). */
    static void matmulBackward(float[] dIn, float[] dW, float[] dB, float[] dOut, float[] in, float[] W,
                               int n, int I, int O) {
        for (int t = 0; t < n; t++) {
            int ro = t * O, ri = t * I;
            for (int i = 0; i < I; i++) {
                float a = in[ri + i];
                int rw = i * O;
                float s = 0f;
                for (int o = 0; o < O; o++) {
                    float d = dOut[ro + o];
                    dW[rw + o] += a * d;
                    s += W[rw + o] * d;
                }
                dIn[ri + i] += s;
            }
            if (dB != null) for (int o = 0; o < O; o++) dB[o] += dOut[ro + o];
        }
    }

    /** LayerNorm: normalise each token vector to mean 0 / variance 1, then scale (gamma) and shift (beta). */
    static void layerNorm(float[] out, float[] mean, float[] rstd, float[] in, float[] g, float[] b, int n, int C) {
        for (int t = 0; t < n; t++) {
            int r = t * C;
            float m = 0f;
            for (int c = 0; c < C; c++) m += in[r + c];
            m /= C;
            float v = 0f;
            for (int c = 0; c < C; c++) { float d = in[r + c] - m; v += d * d; }
            v /= C;
            float s = (float) (1.0 / Math.sqrt(v + 1e-5));
            mean[t] = m; rstd[t] = s;
            for (int c = 0; c < C; c++) out[r + c] = (in[r + c] - m) * s * g[c] + b[c];
        }
    }

    static void layerNormBackward(float[] dIn, float[] dG, float[] dB, float[] dOut, float[] in, float[] mean,
                                  float[] rstd, float[] g, int n, int C) {
        for (int t = 0; t < n; t++) {
            int r = t * C;
            float m = mean[t], s = rstd[t];
            float meanD = 0f, meanDX = 0f;
            for (int c = 0; c < C; c++) {
                float xhat = (in[r + c] - m) * s;
                float d = dOut[r + c] * g[c];
                meanD += d; meanDX += d * xhat;
                dG[c] += dOut[r + c] * xhat;
                dB[c] += dOut[r + c];
            }
            meanD /= C; meanDX /= C;
            for (int c = 0; c < C; c++) {
                float xhat = (in[r + c] - m) * s;
                float d = dOut[r + c] * g[c];
                dIn[r + c] += s * (d - meanD - xhat * meanDX);
            }
        }
    }

    /**
     * Causal multi-head self-attention.
     * For every head: score(t, s) = q_t . k_s / sqrt(headDim) for s <= t (the causal mask hides the future),
     * weights = softmax(scores), output_t = sum_s weights(t, s) * v_s.
     */
    void attention(float[] out, float[] att, float[] qkv, int n) {
        int C = cfg.dModel, H = cfg.nHeads, D = cfg.headDim(), T = cfg.contextLength, C3 = 3 * C;
        float scale = (float) (1.0 / Math.sqrt(D));
        for (int h = 0; h < H; h++) {
            int ho = h * D;
            for (int t = 0; t < n; t++) {
                int ab = h * T * T + t * T;
                int qo = t * C3 + ho;
                float max = Float.NEGATIVE_INFINITY;
                for (int s = 0; s <= t; s++) {
                    int ko = s * C3 + C + ho;
                    float dot = 0f;
                    for (int d = 0; d < D; d++) dot += qkv[qo + d] * qkv[ko + d];
                    dot *= scale;
                    att[ab + s] = dot;
                    if (dot > max) max = dot;
                }
                float sum = 0f;
                for (int s = 0; s <= t; s++) { float e = (float) Math.exp(att[ab + s] - max); att[ab + s] = e; sum += e; }
                for (int s = 0; s <= t; s++) att[ab + s] /= sum;
                int oo = t * C + ho;
                for (int d = 0; d < D; d++) out[oo + d] = 0f;
                for (int s = 0; s <= t; s++) {
                    float w = att[ab + s];
                    int vo = s * C3 + 2 * C + ho;
                    for (int d = 0; d < D; d++) out[oo + d] += w * qkv[vo + d];
                }
            }
        }
    }

    void attentionBackward(float[] dQkv, float[] dOut, float[] att, float[] qkv, int n, float[] dAtt) {
        int C = cfg.dModel, H = cfg.nHeads, D = cfg.headDim(), T = cfg.contextLength, C3 = 3 * C;
        float scale = (float) (1.0 / Math.sqrt(D));
        for (int h = 0; h < H; h++) {
            int ho = h * D;
            for (int t = 0; t < n; t++) {
                int ab = h * T * T + t * T;
                int oo = t * C + ho;
                // d(weights) and d(values)
                float dot = 0f;
                for (int s = 0; s <= t; s++) {
                    int vo = s * C3 + 2 * C + ho;
                    float da = 0f, w = att[ab + s];
                    for (int d = 0; d < D; d++) {
                        da += dOut[oo + d] * qkv[vo + d];
                        dQkv[vo + d] += w * dOut[oo + d];
                    }
                    dAtt[s] = da;
                    dot += w * da;
                }
                // through softmax, then through the q.k dot product
                int qo = t * C3 + ho;
                for (int s = 0; s <= t; s++) {
                    float dScore = att[ab + s] * (dAtt[s] - dot) * scale;
                    int ko = s * C3 + C + ho;
                    for (int d = 0; d < D; d++) {
                        dQkv[qo + d] += dScore * qkv[ko + d];
                        dQkv[ko + d] += dScore * qkv[qo + d];
                    }
                }
            }
        }
    }

    private static final float GELU_C = (float) Math.sqrt(2.0 / Math.PI);

    /** GELU activation (tanh approximation), the non-linearity used by GPT-2/3. */
    static float gelu(float x) {
        return 0.5f * x * (1f + (float) Math.tanh(GELU_C * (x + 0.044715f * x * x * x)));
    }

    static float geluGrad(float x) {
        float u = GELU_C * (x + 0.044715f * x * x * x);
        float th = (float) Math.tanh(u);
        float du = GELU_C * (1f + 3f * 0.044715f * x * x);
        return 0.5f * (1f + th) + 0.5f * x * (1f - th * th) * du;
    }

    /** Numerically stable softmax with temperature. */
    static void softmax(float[] in, int inOff, float[] out, int outOff, int len, float temperature) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < len; i++) max = Math.max(max, in[inOff + i]);
        double sum = 0;
        for (int i = 0; i < len; i++) {
            float e = (float) Math.exp((in[inOff + i] - max) / temperature);
            out[outOff + i] = e; sum += e;
        }
        for (int i = 0; i < len; i++) out[outOff + i] /= (float) sum;
    }

    // =======================================================================
    //  Checkpointing
    // =======================================================================

    public void save(File file, CharTokenizer tok) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
            out.writeInt(0x47505431); // magic "GPT1"
            out.writeInt(cfg.vocabSize); out.writeInt(cfg.contextLength); out.writeInt(cfg.dModel);
            out.writeInt(cfg.nHeads); out.writeInt(cfg.nLayers);
            char[] vocab = tok.vocab();
            for (char c : vocab) out.writeChar(c);
            for (float[] a : params) for (float f : a) out.writeFloat(f);
        }
    }

    public record Loaded(GPT model, CharTokenizer tokenizer) {}

    public static Loaded load(File file) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            if (in.readInt() != 0x47505431) throw new IOException("Not a tiny-llm checkpoint: " + file);
            Config c = new Config();
            c.vocabSize = in.readInt(); c.contextLength = in.readInt(); c.dModel = in.readInt();
            c.nHeads = in.readInt(); c.nLayers = in.readInt();
            char[] vocab = new char[c.vocabSize];
            for (int i = 0; i < vocab.length; i++) vocab[i] = in.readChar();
            GPT m = new GPT(c, 0);
            for (float[] a : m.params) for (int i = 0; i < a.length; i++) a[i] = in.readFloat();
            return new Loaded(m, new CharTokenizer(vocab));
        }
    }

    // =======================================================================
    //  helpers
    // =======================================================================
    private static float[] gaussian(int n, float std, Random r) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) (r.nextGaussian() * std);
        return a;
    }

    private static float[] ones(int n) { float[] a = new float[n]; java.util.Arrays.fill(a, 1f); return a; }

    /** Parameter count grouped by name, handy for printing a model summary. */
    public List<String> summary() {
        List<String> lines = new ArrayList<>();
        for (int i = 0; i < params.length; i++) lines.add(String.format("  %-28s %,10d", names[i], params[i].length));
        return lines;
    }
}
