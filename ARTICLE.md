# How LLMs Actually Work: Build a GPT in Pure Java and Train It on Your Laptop

*Tokens, embeddings, attention, logits, temperature… what all that jargon really means, explained with ~600 lines of plain Java you can run tonight.*

---

Every week someone in a meeting says "the context window is too small", "lower the temperature", or "it's just predicting the next token". Most of us nod along. This article removes the hand-waving.

We will build a real (tiny) GPT-style language model in **plain Java — no libraries, no GPU, no Python** — train it on your laptop in about 10–25 minutes, and get it to write Shakespeare-ish text. Along the way, every buzzword gets mapped to an actual line of code.

👉 **Full source code:** `github.com/<your-username>/tiny-llm-java`

Here is what our model produces after 22 minutes of training on a 2-core CPU:

```
ROMEO:
I'll not not me freshore: he and your gentlement me;
This cause say your blooding to cheeds of Edward wear
My lade strangth the lame: how that the fall.

Thoughts Are secul:
Good make all feart me more. She that what it me of
our daughted again. See is a whish matter'd more
```

Not Shakespeare. But remember: it started knowing *nothing* — not the alphabet, not what a word is, not that names are followed by colons. Everything you see, it learned from raw text. That is exactly what ChatGPT, Gemini and Claude do, just at a vastly larger scale.

---

## The one-sentence version

> **An LLM is a function that takes a sequence of tokens and returns a probability for every possible next token.**

That's it. Text generation is just calling that function in a loop:

```
"ROMEO"   → model → ":"  is very likely   → append ":"
"ROMEO:"  → model → "\n" is very likely  → append "\n"
...
```

This loop is called **autoregressive generation**. Everything else — attention, transformers, training — is about making that one prediction good.

The full pipeline:

```
text → tokenizer → token ids → embeddings → N × transformer blocks → logits → softmax → probabilities → sample → next token
```

Let's build each box.

---

## 1. Tokenization: text becomes numbers

Neural networks only do math, so text must become integers. A **tokenizer** splits text into **tokens** and maps each to an id. The set of all tokens is the **vocabulary**.

We use the simplest possible tokenizer — one token per character:

```java
public static CharTokenizer fit(String corpus) {
    TreeSet<Character> set = new TreeSet<>();
    for (char c : corpus.toCharArray()) set.add(c);   // every distinct character
    ...
}

int[] ids = tok.encode("Hi!");   // → [20, 47, 2]
String s  = tok.decode(ids);     // → "Hi!"
```

Shakespeare uses 65 distinct characters, so our **vocab size** is 65.

**In real LLMs:** they use sub-word tokenizers like **BPE** (Byte-Pair Encoding). "unbelievable" might become `un` + `believ` + `able`. Vocabularies are 50k–200k tokens. That's why API pricing is "per token" and why a token is roughly ¾ of an English word.

---

## 2. Embeddings: numbers become meaning

An id like `47` has no meaning by itself. So each token id looks up a row in a learnable table — its **embedding**, a vector of `d_model` numbers (128 in our model).

Attention (coming next) has no built-in sense of order, so we add a second learned vector per *position*: the **positional embedding**.

```java
// x[t] = tokenEmbedding[token at t] + positionEmbedding[t]
for (int t = 0; t < n; t++) {
    int tok = tokens[t];
    for (int c = 0; c < C; c++)
        x0[t * C + c] = wte[tok * C + c] + wpe[t * C + c];
}
```

These tables start as random noise. During training, tokens used in similar ways drift toward similar vectors. That is where the famous "king − man + woman ≈ queen" property comes from.

**Parameters** (or **weights**) are all these learnable numbers. "A 70B model" means 70 billion of them. Ours has about **818,000**.

---

## 3. Self-attention: tokens talking to each other

This is the idea that made modern LLMs possible (from the 2017 paper *Attention Is All You Need*).

To predict what comes after `"the king took his"`, the model must look back at `"king"`. **Self-attention** lets each token gather information from earlier tokens. Each token emits three vectors:

* **Query (Q)** — "what am I looking for?"
* **Key (K)** — "what do I contain?"
* **Value (V)** — "what will I share if you pick me?"

The attention score between token *t* and an earlier token *s* is the dot product `Q_t · K_s`. Then **softmax** turns the scores into weights that sum to 1, and the output is the weighted mix of the Values.

```java
for (int s = 0; s <= t; s++) {              // s <= t : the CAUSAL MASK
    float dot = 0f;
    for (int d = 0; d < D; d++) dot += q[t][d] * k[s][d];
    att[s] = dot / sqrt(D);                  // "scaled" dot-product attention
}
softmax(att);                                 // weights that sum to 1
for (int s = 0; s <= t; s++)
    for (int d = 0; d < D; d++) out[t][d] += att[s] * v[s][d];
```

Three pieces of jargon live in this snippet:

* **Causal mask** — `s <= t`. A token may only look at the past, never the future. Otherwise the model could cheat during training by peeking at the answer. That is what "decoder-only" means.
* **Scaled** — dividing by `√D` keeps the scores from getting huge, which would make softmax too peaky.
* **Multi-head attention** — we run this 4 times in parallel on different slices of the vector (4 **heads**). One head might track "who is speaking", another "what's the last punctuation". GPT-3 had 96 heads per layer.

The **context window** (`contextLength = 64` here) is how many tokens attention can see. Attention cost grows with the *square* of the window, which is why long context is expensive.

---

## 4. The Transformer block

A **transformer block** = attention (tokens share information) + an **MLP / feed-forward network** (each token processes what it gathered). We stack `N` of these blocks; that's the model's **depth** or **layers**.

```java
for (int l = 0; l < cfg.nLayers; l++) {
    // attention sub-layer
    a   = layerNorm(x);
    qkv = a @ W_qkv;            // project into Q, K, V
    y   = attention(qkv);
    x   = x + y @ W_out;        // RESIDUAL connection

    // feed-forward sub-layer
    b   = layerNorm(x);
    h   = gelu(b @ W_up);       // expand 4x, non-linearity
    x   = x + h @ W_down;       // RESIDUAL connection
}
```

The jargon:

* **Residual connection** (`x = x + ...`) — each sub-layer *adds* a correction instead of replacing the vector. The running vector is called the **residual stream**. This is what makes deep networks trainable.
* **LayerNorm** — rescales each vector to mean 0 / variance 1 so numbers don't explode as they pass through many layers. Applying it *before* each sub-layer is called **pre-norm** (used by GPT-2 onward).
* **GELU** — the **activation function**, a smooth non-linearity. Without it, stacking layers would collapse into one big linear function.
* The MLP widens the vector 4× (128 → 512) and back. Researchers believe much of a model's "factual memory" lives in these MLP weights.

---

## 5. Logits, softmax and loss

After the last block, an **LM head** projects each 128-dim vector to 65 scores — one per vocabulary token. Raw scores are called **logits**. **Softmax** turns logits into probabilities:

```java
p_i = exp(logit_i) / Σ_j exp(logit_j)
```

During training we know the true next character, so we measure how surprised the model was. That is the **cross-entropy loss**:

```java
loss = -log(probability the model gave to the correct next token)
```

* Perfect prediction (p = 1) → loss 0.
* Random guessing over 65 chars → loss = ln(65) ≈ **4.17**. Our first training step prints **4.1958**. It checks out!

**Perplexity** = `e^loss`, roughly "how many tokens the model is torn between". It starts at 65 and our model gets it to about 5.7.

---

## 6. Training: learning from mistakes

**Pre-training** means repeating this loop thousands of times on raw text:

1. Grab a **batch** of random 64-character windows. Inputs are `text[i..i+64)`, targets are the same text shifted by one, `text[i+1..i+65)`. The text *is* the label, which is why this is called **self-supervised learning**.
2. **Forward pass** — run the model, compute the loss.
3. **Backward pass (backpropagation)** — use the chain rule to compute the **gradient**: for each of the 818k parameters, "if I nudge you up, does the loss go up or down, and by how much?"
4. **Optimizer step** — nudge every parameter slightly in the direction that lowers the loss.

```java
for (int step = 1; step <= steps; step++) {
    sampleBatch(xs, ys);
    model.forward(x, T, ws);
    loss += model.loss(y, T, ws);
    model.backward(x, y, T, ws, grads);

    AdamW.clipGradNorm(grads, 1.0f);              // GRADIENT CLIPPING
    float lr = learningRate(step, steps, warmup); // WARMUP + COSINE DECAY
    optimizer.step(grads, lr);                    // AdamW update
}
```

In PyTorch, step 3 is a single `loss.backward()`. In this repo it is written by hand in `GPT.java`, so you can see exactly how the error signal flows back through softmax, attention, LayerNorm and GELU. A `GradCheck` program confirms it numerically.

More jargon you'll now recognise:

* **Learning rate** — the size of each nudge. Too big and training explodes; too small and it crawls.
* **Warmup + cosine decay** — start with tiny steps, ramp up, then slowly cool down. Nearly every LLM uses this schedule.
* **AdamW** — the standard optimizer. It keeps a running average of each weight's gradient (**momentum**) and gives every weight its own adaptive step size. The "W" is **weight decay**, a gentle pull toward zero that reduces overfitting.
* **Gradient clipping** — if the gradient is unusually large, scale it down so one bad batch can't wreck the model.
* **Validation loss** — loss on 10% of the text the model never trains on. If training loss falls but validation loss rises, the model is **overfitting** (memorising, not generalising).
* **Checkpoint** — saved weights. Our "model file" is just 818k floats, about 3 MB. A 70B model's checkpoint is the same idea, at ~140 GB.

Training uses all your CPU cores. Each thread processes part of the batch, then gradients are summed — a mini version of the **data parallelism** used across thousands of GPUs.

---

## 7. Inference: generating text

Once trained, **inference** is the autoregressive loop from the start. The only interesting choice is *how* to pick the next token from the probabilities:

```java
if (temperature == 0) return argmax(logits);       // GREEDY decoding
probs = softmax(logits / temperature);             // TEMPERATURE
keep only the top-k most likely tokens;            // TOP-K
keep the smallest set whose probs sum to top-p;    // TOP-P (nucleus sampling)
return randomSample(probs);
```

* **Temperature** < 1 sharpens the distribution (safe, repetitive). Temperature > 1 flattens it (creative, error-prone). This is the slider in every LLM API.
* **Top-k / top-p** cut off the long tail of unlikely tokens so the model rarely picks something absurd.
* Once the text exceeds the context window, we keep only the last 64 tokens. The model literally **forgets** anything older. That's why chatbots "lose track" in long conversations.

Try it:

```bash
./run.sh generate --prompt "JULIET:" --temperature 0.5   # conservative
./run.sh generate --prompt "JULIET:" --temperature 1.2   # chaotic
```

Seeing a model invent a plausible-looking character name that never appears in Shakespeare is also a nice, harmless demonstration of **hallucination**. The model produces what is *statistically plausible*, not what is *true*.

---

## Run it yourself

You need JDK 17+.

```bash
git clone https://github.com/<your-username>/tiny-llm-java.git
cd tiny-llm-java
./run.sh gradcheck        # verifies backprop (seconds)
./run.sh train            # ~818k params; 22 min on 2 CPU cores, faster with more cores
./run.sh generate --prompt "ROMEO:" --tokens 500
```

Watch the samples improve during training. Here is what the model printed at different points of a 2,000-step run:

```
step   50  (val loss 2.71)
FDCFDROF
Y
VUSEpAB! 3e ma&$ w:

step  500  (val loss 2.14)
Heren pore hing me fardmor thres,
The dest pres foll not you that out him helt wall and pall's lardeds

step 1000  (val loss 1.88)
Which the carrer, Lord. The biken us, and busin is him
To son vising in mine trible ret my of as a bearther your dead on

step 1500  (val loss 1.76)
Harge this have be else a man to-brother!

CAPULET:
They from accome, one heartor

step 2000  (val loss 1.73)
In odam cheer the ablood on and bettert the duken made
To might; and therefore upon the Lord By highter like
```

First noise, then letters with spaces, then word shapes, then names followed by colons and line breaks, then real words. Nobody programmed those rules. They emerged from minimising one number: the loss.

Train on your own text with `--data your_file.txt`.

---

## From this toy to ChatGPT

| | This repo | GPT-3 (2020) |
|---|---|---|
| Parameters | 0.8 M | 175 B |
| Layers | 4 | 96 |
| d_model | 128 | 12,288 |
| Attention heads | 4 | 96 |
| Context window | 64 | 2,048 |
| Vocabulary | 65 characters | 50,257 BPE tokens |
| Training data | 1 MB | ~300 B tokens |

The architecture is the same. What modern assistants add on top:

* **Scale** — more parameters, more data and more compute make the same next-token objective yield grammar, facts, reasoning and code. These relationships are described by **scaling laws**.
* **Instruction tuning / fine-tuning** — continue training on examples of questions and good answers so the model follows instructions instead of merely continuing text.
* **RLHF** (Reinforcement Learning from Human Feedback) — further tune the model toward answers people rate as helpful and safe.
* **KV cache** — at inference, cache each token's Keys and Values so they aren't recomputed for every new token. (Our `Sampler` recomputes everything, which is fine at this size.)
* **Quantization** — store weights in 8 or 4 bits instead of 32 so big models fit on smaller hardware.
* **RAG** (Retrieval-Augmented Generation) — paste relevant documents into the context window at question time, so the model can answer from information it was never trained on.
* **Embeddings APIs** — the same idea as our embedding table, used for semantic search.

---

## Experiments to try

1. **Scale up**: `--dmodel 192 --layers 6 --heads 6 --ctx 128`. Watch validation loss drop further (and training slow down).
2. **Overfit on purpose**: train on a 5 KB file. Training loss goes near zero, validation loss climbs.
3. **Break it**: set `--lr 0.05` and watch the loss explode. Then remove the clipping line and see it get worse.
4. **Look inside**: print `ws.att` for a prompt and see which earlier characters each head attends to.

---

## Wrap-up

Next time someone says "it's just predicting the next token", you'll know that's both completely true and wildly underselling it. The prediction runs through embeddings, stacks of attention and MLP layers, and millions (or billions) of weights tuned by gradient descent until the predictions are good.

You built the same thing, in Java, on your laptop.

⭐ Code: `github.com/<your-username>/tiny-llm-java` — PRs and experiments welcome.

*Inspired by Andrej Karpathy's nanoGPT and llm.c, and the paper "Attention Is All You Need" (Vaswani et al., 2017).*
