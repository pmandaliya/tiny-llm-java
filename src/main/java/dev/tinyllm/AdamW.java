package dev.tinyllm;

/**
 * AdamW optimizer - the default optimizer for virtually every LLM.
 *
 * Plain gradient descent does  w -= lr * grad.
 * Adam keeps a running average of the gradient (momentum, m) and of its square (v),
 * so every weight gets its own adaptive step size. The "W" means weight decay is
 * applied directly to the weights (decoupled), which acts as regularisation.
 */
public final class AdamW {
    private final float[][] params, m, v;
    private final boolean[] decay;
    private final float beta1, beta2, eps, weightDecay;
    private int step = 0;

    public AdamW(GPT model, float beta1, float beta2, float weightDecay) {
        this.params = model.params;
        this.decay = model.decay;
        this.m = model.newGrads();
        this.v = model.newGrads();
        this.beta1 = beta1; this.beta2 = beta2; this.eps = 1e-8f; this.weightDecay = weightDecay;
    }

    public void step(float[][] grads, float lr) {
        step++;
        float bc1 = 1f - (float) Math.pow(beta1, step);
        float bc2 = 1f - (float) Math.pow(beta2, step);
        for (int i = 0; i < params.length; i++) {
            float[] w = params[i], g = grads[i], mi = m[i], vi = v[i];
            float wd = decay[i] ? weightDecay : 0f;
            for (int j = 0; j < w.length; j++) {
                mi[j] = beta1 * mi[j] + (1 - beta1) * g[j];
                vi[j] = beta2 * vi[j] + (1 - beta2) * g[j] * g[j];
                float mHat = mi[j] / bc1, vHat = vi[j] / bc2;
                w[j] -= lr * (mHat / ((float) Math.sqrt(vHat) + eps) + wd * w[j]);
            }
        }
    }

    /** Gradient clipping: if the overall gradient vector is too long, shrink it. Prevents loss spikes. */
    public static float clipGradNorm(float[][] grads, float maxNorm) {
        double sq = 0;
        for (float[] g : grads) for (float x : g) sq += (double) x * x;
        float norm = (float) Math.sqrt(sq);
        if (norm > maxNorm) {
            float s = maxNorm / (norm + 1e-6f);
            for (float[] g : grads) for (int j = 0; j < g.length; j++) g[j] *= s;
        }
        return norm;
    }
}
