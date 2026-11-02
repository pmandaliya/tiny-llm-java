package dev.tinyllm;

/**
 * Hyperparameters of the model. These are the "knobs" people mean when they say
 * "a 7B model with 32 layers and a 4k context window".
 */
public final class Config {
    /** Number of distinct tokens the model knows (size of the vocabulary). */
    public int vocabSize;
    /** Context window / block size: max tokens the model can look at at once. */
    public int contextLength = 64;
    /** Embedding dimension (a.k.a. d_model / hidden size). */
    public int dModel = 128;
    /** Number of attention heads per layer. dModel must be divisible by this. */
    public int nHeads = 4;
    /** Number of stacked transformer blocks (depth). */
    public int nLayers = 4;

    public int headDim() { return dModel / nHeads; }
    public int ffnDim() { return 4 * dModel; }

    @Override
    public String toString() {
        return String.format("vocab=%d ctx=%d d_model=%d heads=%d layers=%d",
                vocabSize, contextLength, dModel, nHeads, nLayers);
    }
}
