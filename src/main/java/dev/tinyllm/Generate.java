package dev.tinyllm;

import java.io.File;
import java.util.Map;

/**
 * Inference: load a trained checkpoint and generate text from a prompt.
 *
 *   java -cp target/classes dev.tinyllm.Generate --prompt "ROMEO:" --tokens 400 --temperature 0.8
 */
public final class Generate {
    public static void main(String[] args) throws Exception {
        Map<String, String> a = Args.parse(args);
        String ckpt = a.getOrDefault("model", "checkpoints/model.bin");
        String prompt = a.getOrDefault("prompt", "ROMEO:").replace("\\n", "\n");
        int tokens = Integer.parseInt(a.getOrDefault("tokens", "400"));
        float temperature = Float.parseFloat(a.getOrDefault("temperature", "0.8"));
        int topK = Integer.parseInt(a.getOrDefault("top-k", "0"));
        float topP = Float.parseFloat(a.getOrDefault("top-p", "0.95"));
        long seed = Long.parseLong(a.getOrDefault("seed", String.valueOf(System.nanoTime())));

        GPT.Loaded loaded = GPT.load(new File(ckpt));
        GPT model = loaded.model();
        CharTokenizer tok = loaded.tokenizer();
        System.err.printf("Loaded %s (%,d params) | temperature=%.2f top-k=%d top-p=%.2f%n%n",
                model.cfg, model.numParams(), temperature, topK, topP);

        System.out.print(prompt);
        new Sampler(model, seed).generate(tok.encode(prompt), tokens, temperature, topK, topP,
                id -> { System.out.print(tok.decode(id)); System.out.flush(); });   // stream tokens as they arrive
        System.out.println();
    }
}
