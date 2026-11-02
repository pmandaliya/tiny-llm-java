package dev.tinyllm;

import java.util.Random;

/**
 * Sanity test for the hand-written backpropagation.
 * Compares the analytic gradient with a numerical one: (loss(w+eps) - loss(w-eps)) / 2eps.
 */
public final class GradCheck {
    public static void main(String[] args) {
        Config cfg = new Config();
        cfg.vocabSize = 11; cfg.contextLength = 8; cfg.dModel = 16; cfg.nHeads = 2; cfg.nLayers = 2;
        GPT model = new GPT(cfg, 7);
        // bigger random weights so the check is not dominated by tiny numbers
        Random r = new Random(1);
        for (float[] p : model.params) for (int i = 0; i < p.length; i++) p[i] += (float) (r.nextGaussian() * 0.1);

        int n = 8;
        int[] x = new int[n], y = new int[n];
        for (int i = 0; i < n; i++) { x[i] = r.nextInt(cfg.vocabSize); y[i] = r.nextInt(cfg.vocabSize); }
        GPT.Workspace ws = model.newWorkspace();
        float[][] g = model.newGrads();
        model.forward(x, n, ws);
        model.loss(y, n, ws);
        model.backward(x, y, n, ws, g);

        double worst = 0;
        float eps = 1e-2f;
        for (int pi = 0; pi < model.params.length; pi++) {
            float[] p = model.params[pi];
            double maxRel = 0;
            for (int k = 0; k < 5; k++) {
                int j = r.nextInt(p.length);
                float old = p[j];
                p[j] = old + eps; model.forward(x, n, ws); float lp = model.loss(y, n, ws);
                p[j] = old - eps; model.forward(x, n, ws); float lm = model.loss(y, n, ws);
                p[j] = old;
                double num = (lp - lm) / (2.0 * eps), ana = g[pi][j];
                // float32 finite differences are noisy for tiny gradients, so ignore differences below 2e-4
                double diff = Math.abs(num - ana);
                double rel = diff < 2e-4 ? 0 : diff / (Math.abs(num) + Math.abs(ana));
                maxRel = Math.max(maxRel, rel);
            }
            worst = Math.max(worst, maxRel);
            System.out.printf("%-28s max rel error %.2e%n", model.names[pi], maxRel);
        }
        System.out.println(worst < 2e-2 ? "\nGRADIENT CHECK PASSED" : "\nGRADIENT CHECK FAILED");
        if (worst >= 2e-2) System.exit(1);
    }
}
