# Training tiny-llm-java on Wikipedia (or any other text)

The main [README](README.md) trains on 1 MB of Shakespeare. This guide shows how to build a real
**training corpus** from Wikipedia: every article about **World War II**, or all of Wikipedia. It
also covers how to clean any other text so the model can learn from it.

Everything is plain Java with no extra libraries. The only outside tool you need is `bzcat`, which
comes with macOS and Linux.

> **Results below are real.** A model trained for 38 minutes on 2 CPU cores on 12.9 MB of WWII articles,
> given the prompt `The Battle of`, wrote:
>
> ```
> The Battle of the Hunsbach of the capital beginning the time election all short of more of the south a can should the original strun of the 1940. One in the Allegar former to the Shoviet US Berler were agreement of sident in 1949 force of the Jews General Britain well the position, and a selections to convent fo
> ```
>
> It's fluent-looking nonsense: the model learned the *shape* of a Wikipedia history article (years like
> 1940 and 1949, "Soviet", "US", "General", "Britain", "the Allies") but not grammar or facts. That's exactly what to
> expect from a 0.8M-parameter model. The section on scale below explains why.

---

## Contents

1. [Can I train on *all* of Wikipedia?](#1-can-i-train-on-all-of-wikipedia-the-honest-math)
2. [The data pipeline](#2-the-data-pipeline)
3. [Route A: World War II articles through the Wikipedia API](#3-route-a-world-war-ii-articles-through-the-wikipedia-api-recommended)
4. [Route B: World War II articles from the official dump](#4-route-b-world-war-ii-articles-from-the-official-dump)
5. [Train the model on your WWII corpus](#5-train-the-model-on-your-wwii-corpus)
6. [Route C: All of Wikipedia](#6-route-c-all-of-wikipedia)
7. [Any other data source](#7-any-other-data-source)
8. [Troubleshooting](#8-troubleshooting)
9. [Data jargon cheat-sheet](#9-data-jargon-cheat-sheet)

---

## 1. Can I train on *all* of Wikipedia? The honest math

**Extracting** all of English Wikipedia works fine on a laptop. The tools in this repo stream it one
article at a time. **Training** this model on all of it does not, and the reason is compute, not memory.

| | Measured / estimated |
|---|---|
| English Wikipedia as clean plain text | ~25–30 GB (extrapolated from dump part 10: 674 MB from 2.3% of the dump) |
| WWII-related articles only | ~0.7–1 GB, roughly 100k articles (extrapolated from parts 1 and 10) |
| Training speed, default model (0.8M params), 2 CPU cores | ~1,450 tokens/s (measured) |
| Training speed, default model, 8-core laptop | ~4,000–6,000 tokens/s (rough estimate) |
| **One pass (epoch) over all of Wikipedia at 5,000 tok/s** | ~2 months (25 B chars ÷ 5,000/s); ~6 months on 2 cores |
| **One pass over the WWII subset at 5,000 tok/s** | ~2 days; but you don't need a full epoch (see below) |

There is a second reason. The **Chinchilla scaling law** (DeepMind, 2022) says a model learns
best from about **20 tokens of data per parameter**:

* Our 0.8M-parameter model is "full" after about **16M tokens** (16 MB of text). More data than that
  barely helps it. It's like pouring a lake into a glass.
* All of English Wikipedia (billions of tokens) suits a model of hundreds of millions to billions of
  parameters, trained on GPUs.

**So the plan is:** use a **topic subset** such as WWII (tens to hundreds of MB), and train on a
random sample of it. The trainer always picks random windows, so you never need to finish an epoch.
Section 6 still shows how to extract all of Wikipedia if you want it for a bigger project.

---

## 2. The data pipeline

Every LLM, from this toy to GPT-4, has the same front end: collect → extract → clean → (dedupe/filter) → tokenize → train.

```mermaid
flowchart LR
    A["Source<br/>Wikipedia API or dump,<br/>books, your notes"] --> B["Extract<br/>WikiFetch / WikiDump<br/>wikitext → plain text"]
    B --> C["Clean<br/>TextClean<br/>Unicode → ~95 chars"]
    C --> D["Corpus file<br/>data/ww2.txt"]
    D --> E["Train<br/>random 64-char windows<br/>90% train / 10% val"]
    E --> F["Checkpoint<br/>checkpoints/ww2.bin"]
    F --> G["Generate"]
```

New code for this guide (in `src/main/java/dev/tinyllm/data/`):

| Tool | What it does |
|---|---|
| `WikiFetch` | Crawls a Wikipedia **category tree** through the official API and downloads each article's text. |
| `WikiDump` | Streams the official multi-GB **XML dump** and keeps all articles, or only those whose categories match a regex. |
| `WikiText` | Turns wiki markup (`{{Infobox}}`, `[[links|text]]`, `<ref>`, tables) into plain prose. |
| `TextClean` | Maps any text to a small character set: `“D-Day” – Café` → `"D-Day" - Cafe`. |

The trainer (`Train`) also gained `--resume`, `--max-mb`, `--sample-prompt`, an ETA, and 1-byte token
storage, so a 1 GB corpus uses about 1 GB of RAM instead of 4.

Build once:

```bash
./run.sh build          # or: mvn package
```

---

## 3. Route A: World War II articles through the Wikipedia API (recommended)

Wikipedia files articles into **categories**, which nest. `Category:World War II` contains
`Category:Battles of World War II`, which contains `Category:Battles of World War II involving Japan`,
and so on. `WikiFetch` walks that tree breadth-first and downloads every article it finds.

```bash
java -cp target/classes dev.tinyllm.data.WikiFetch \
    --category "World War II" \
    --depth 3 \
    --follow-regex "World War II|Second World War" \
    --user-agent "tiny-llm-java/1.0 (your-email@example.com)" \
    --out data/ww2.txt
```

What you'll see (real output from a `--depth 1` test run):

```
  [depth 0] Category:World War II                    +19 articles, +17 subcategories (total 19 articles, 17 queued)
  [depth 1] Category:World War II-related lists      +40 articles, +0 subcategories (total 78 articles, 12 queued)
  [depth 1] Category:Chronology of World War II      +39 articles, +0 subcategories (total 124 articles, 10 queued)
  [depth 1] Category:Aftermath of World War II       +95 articles, +0 subcategories (total 219 articles, 9 queued)
  server busy (HTTP 429), retrying in 50s...
  ...
Found 305 articles in 18 categories. Downloading text...
  downloaded 300 / 305 articles, kept 262 (4.7 MB)
Done: 262 articles, 4.7 MB -> data/ww2.txt
```

**The knobs:**

| Flag | Meaning |
|---|---|
| `--category` | Root category, without the `Category:` prefix. Try `Roman Empire`, `Physics`, `Cricket`... |
| `--depth` | How many levels of sub-categories to follow. Depth 1 gave 305 articles in our test; each extra level multiplies that, so watch the running total it prints. |
| `--follow-regex` | Only descend into sub-categories whose name matches. **Important:** category trees drift off-topic fast (WWII → aircraft → Boeing → airports in Ohio). |
| `--max-articles` | Stop after N articles. Good for a quick first run. |
| `--min-chars` | Skip stubs shorter than this (default 1500). |
| `--lang` | Another Wikipedia, e.g. `de` or `fr`. TextClean strips accents, so stick to Latin-script languages. |
| `--user-agent` | **Put your contact info here.** Wikimedia's API policy asks for it. |
| `--delay-ms` | Pause between requests (default 200 ms). The tool also obeys `Retry-After` and `maxlag`. |

**How it's polite:** one request at a time, 50 articles per request, and it backs off automatically
when the server says it's busy (HTTP 429). Don't run several copies in parallel. If you need more
than about 100k articles, use Route B instead.

---

## 4. Route B: World War II articles from the official dump

Wikimedia publishes a full snapshot of every article twice a month at
<https://dumps.wikimedia.org/enwiki/latest/>. No rate limits, no API, and it's reproducible: the same
dump always gives the same corpus.

```bash
mkdir -p dumps && cd dumps
curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2   # ~26 GB
cd ..

bzcat dumps/enwiki-latest-pages-articles.xml.bz2 | \
  java -cp target/classes dev.tinyllm.data.WikiDump \
    --category-regex "World War II|Second World War" \
    --out data/ww2.txt
```

`WikiDump` reads the XML as a stream, so memory use stays flat. It skips redirects, talk pages and
templates, and keeps an article if any of its `[[Category:...]]` tags match the regex.

**Measured on this repo's test machine:**

| Input | Pages scanned | WWII articles kept | Clean text | Time (2 cores) |
|---|---|---|---|---|
| Dump part 1 (`pages-articles1.xml-p1p41242.bz2`, 300 MB) | 27,352 | 354 | 12.9 MB | 57 s |
| Dump part 10 (`pages-articles10.xml-p4045403p5399366.bz2`, 602 MB) | 441,717 | 2,209 | 17.0 MB | 2.3 min |
| Full dump (26 GB), estimated | ~20 M | ~100,000 | ~0.7–1 GB | ~1.5–2 h with `bzcat` (faster with `lbzip2`) |

**Tips**

* **Try it on one part first.** The dump is also published in about 70 smaller parts
  (`enwiki-latest-pages-articles1.xml-p1p41242.bz2`, ...). Download one (~300 MB) and run the same command
  to see what your filter keeps before spending hours on the full file.
* **Faster decompression:** `bzcat` uses one core and is the bottleneck. Install `lbzip2` and use
  `lbzip2 -dc file.bz2 | java ...` to decompress on all cores.
* **Tighter topic.** `"World War II|Second World War"` also catches people who merely served in the
  war (actors, writers, scientists). For battles and campaigns only, try
  `--category-regex "^(Battles|Campaigns|Military operations|Naval battles|Air battles|Conflicts) .*World War II"`.
* **By title instead:** `--title-regex "^Battle of|^Operation "`.
* `--max-articles 1000` or `--max-mb 50` stops early.

---

## 5. Train the model on your WWII corpus

```bash
java -cp target/classes dev.tinyllm.Train \
    --data data/ww2.txt \
    --steps 3000 \
    --sample-prompt "The Battle of" \
    --out checkpoints/ww2.bin

java -cp target/classes dev.tinyllm.Generate \
    --model checkpoints/ww2.bin --prompt "In 1944, the Allied forces" --tokens 500 --temperature 0.7
```

**Real training log** (12.9 MB of WWII articles, default model, 2 CPU cores):

```
Dataset: 12,936,310 chars (11,642,679 train / 1,293,631 val tokens)
Model:   vocab=92 ctx=64 d_model=128 heads=4 layers=4
Params:  825,088
Training 3000 steps, batch 16 x 64 tokens = 3,072,000 tokens (0.26 epochs), 2 threads

step     1 | loss 4.5309 | lr 3.00e-05 | grad-norm 6.35 | 633.7 tok/s | ETA 81 min
step   500 | loss 2.2234 | lr 2.88e-03 | grad-norm 0.72 | 1456.8 tok/s | ETA 29 min
---- eval @ 500: val loss 2.2225 | perplexity 9.23
step  1000 | loss 2.0257 | lr 2.41e-03 | grad-norm 0.65 | 1130.5 tok/s | ETA 30 min
---- eval @ 1000: val loss 1.9785 | perplexity 7.23
step  1500 | loss 1.7673 | lr 1.72e-03 | grad-norm 0.64 | 1242.7 tok/s | ETA 21 min
---- eval @ 1500: val loss 1.8311 | perplexity 6.24
step  2000 | loss 1.6669 | lr 1.02e-03 | grad-norm 0.70 | 1290.9 tok/s | ETA 13 min
---- eval @ 2000: val loss 1.7005 | perplexity 5.48
step  2500 | loss 1.7022 | lr 4.93e-04 | grad-norm 0.76 | 1324.7 tok/s | ETA 6 min
---- eval @ 2500: val loss 1.7159 | perplexity 5.56
step  3000 | loss 1.6986 | lr 3.00e-04 | grad-norm 0.80 | 1356.9 tok/s | ETA 0 min
---- eval @ 3000: val loss 1.6153 | perplexity 5.03
Done in 37.8 min. Best val loss 1.6153. Checkpoint: checkpoints/ww2.bin
```

Read it like this:

* Loss starts at ~4.5 = ln(92). The cleaned Wikipedia text has 92 distinct characters, so the
  untrained model is guessing among 92 options.
* **"0.26 epochs"**: in 3,000 steps the model saw only about a quarter of the corpus, once. With this
  much data it can't **overfit** (memorise), and train and val loss stay close together.
* Speed dips mid-run (1,130 tok/s) because another job shared the CPU. Training is compute-bound, so
  close other heavy apps.
* Wikipedia prose is harder to predict than Shakespeare (more names, numbers and facts), so loss ends higher.

### Bigger models for bigger data

More data means you can, and should, use a bigger model. Parameter counts are exact and
speeds were measured on 2 CPU cores. Scale speeds up roughly in proportion to your core count.

| Preset | Flags | Params | Speed (2 cores) | Notes |
|---|---|---|---|---|
| Small (default) | `--dmodel 128 --layers 4 --heads 4 --ctx 64` | 825,088 | ~1,450 tok/s | 3,000 steps in ~35 min |
| Medium | `--dmodel 192 --layers 6 --heads 6 --ctx 128` | 2,729,472 | ~490 tok/s | 3,000 steps in ~3.5 h; longer context sees whole sentences |
| Large | `--dmodel 256 --layers 8 --heads 8 --ctx 128` | 6,398,464 | ~190 tok/s | 3,000 steps in ~9 h: an overnight run |

For bigger models, lower the learning rate a little (`--lr 1e-3`) and train for more steps.

### Long runs: resume and slices

* **Resume:** `--resume checkpoints/ww2.bin` loads the weights and keeps training, for example on
  more data or for more steps. The optimizer's memory (Adam momentum) starts fresh, so use a smaller
  `--lr` and a short `--warmup` when resuming. Model-size flags are ignored, because the checkpoint fixes them.
* **Slices:** `--max-mb 200` trains on the first 200 MB of a huge file.
* **Memory:** the corpus takes about 1 byte per character of RAM. For a 1 GB file run
  `java -Xmx3g -cp target/classes dev.tinyllm.Train ...`.
* **Run overnight safely:** `nohup java -cp target/classes dev.tinyllm.Train ... > train.log 2>&1 &`,
  then `tail -f train.log`. The best checkpoint (lowest validation loss) is saved at every eval.

---

## 6. Route C: All of Wikipedia

Same command as Route B, without the filter:

```bash
lbzip2 -dc dumps/enwiki-latest-pages-articles.xml.bz2 | \
  java -cp target/classes dev.tinyllm.data.WikiDump --out data/wiki_all.txt
```

This produces roughly 25–30 GB of clean text (about 7 million articles), so check you have the disk space. Then train on
a slice, because Java arrays (and this model's appetite) top out long before that:

```bash
java -Xmx4g -cp target/classes dev.tinyllm.Train --data data/wiki_all.txt --max-mb 1000 \
    --dmodel 256 --layers 8 --heads 8 --ctx 128 --lr 1e-3 --steps 20000 --out checkpoints/wiki.bin
```

**Want a random slice instead of the first N MB?** The dump is ordered by page ID, roughly by when
articles were created, so the first GB leans toward old, long, popular articles. For a more
representative 5% sample, keep every 20th article (articles are separated by two blank lines):

```bash
awk 'BEGIN{RS="\n\n\n"; ORS="\n\n\n"} NR % 20 == 0' data/wiki_all.txt > data/wiki_5pct.txt
```

**If you really want to use all of it**, the next steps are what real labs do, and each is a good
learning project on top of this repo:

1. **Byte-level BPE tokenizer.** Tokens of about 4 characters instead of 1 means 4× fewer steps per article.
2. **Memory-mapped data loading** (`FileChannel.map`), to train on files bigger than 2 GB.
3. **A GPU and a framework** (PyTorch / JAX), or the Java route: the Vector API or a GPU binding.
   This gives a 100–1000× speed-up over our hand-written loops.

---

## 7. Any other data source

Anything that can become a `.txt` file works. The rule of thumb: **at least 1 MB of text with a
consistent style**, cleaned with `TextClean`.

```bash
# one file
java -cp target/classes dev.tinyllm.data.TextClean --in raw.txt --out data/clean.txt

# a whole folder: every .txt and .md file, recursively
java -cp target/classes dev.tinyllm.data.TextClean --in ~/Documents/notes --out data/notes.txt
# prints: number of files, characters before -> after, and the final vocabulary size

java -cp target/classes dev.tinyllm.Train --data data/notes.txt --out checkpoints/notes.bin
```

Ideas:

| Source | How to get text | Model learns |
|---|---|---|
| **Project Gutenberg** (public-domain books) | Download "Plain Text UTF-8" from <https://www.gutenberg.org>, and delete the license header and footer | Dickens, Austen, Sherlock Holmes... |
| **Any Wikipedia topic** | `WikiFetch --category "Roman Empire"`, `"Cricket"`, `"Physics"` ... | That domain's vocabulary and style |
| **Your own notes / blog** | Export as Markdown, then run `TextClean --in folder` | Your writing style |
| **Source code** | `find . -name '*.java' -exec cat {} + > code.txt` | Brace-matching, `public static void`... (a fun one) |
| **Hugging Face datasets** | Download the Parquet files from the dataset's "Files" tab, write the text column to a `.txt` (e.g. a few lines of pandas), then run `TextClean` | Anything: recipes, news, dialogue |
| **Subtitles / scripts / lyrics you own** | Plain-text export | Dialogue structure |

**Cleaning tips** (more corpus quality comes from these steps than from any model change):

* **Remove boilerplate:** navigation menus, license headers, "Click here to subscribe".
* **Deduplicate.** Repeated text gets memorised: `awk '!seen[$0]++' in.txt > out.txt` removes duplicate lines.
* **Keep one style per model.** A model trained on a mix of tweets and legal contracts learns neither well.
* **Respect licenses.** Wikipedia is CC BY-SA, Gutenberg is public domain in the US, and your own writing is yours.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Corpus has 412 distinct characters. Run TextClean...` | Raw text has rare Unicode. Run `TextClean` first. `WikiFetch` and `WikiDump` already clean unless you pass `--keep-unicode`. |
| `Character 'x' ... not in the checkpoint's vocabulary` when using `--resume` | The new data has a character the old model never saw. Clean it the same way, or train a fresh model. |
| `OutOfMemoryError` | Add `-Xmx4g` (or more) after `java`, or use `--max-mb`. |
| `server busy (HTTP 429), retrying...` | Normal. Wikipedia is rate-limiting you and the tool waits. If it gives up, raise `--delay-ms` or use the dump. |
| `JAXP00010004 ... accumulated size of entities` | Old JDK XML limits. WikiDump lifts them; make sure you use the jar or classes from this repo. |
| Loss stuck around 3+ | Learning rate too high or too low. Try `--lr 1e-3`. Also check that the corpus isn't mostly junk with `head -c 3000 data/ww2.txt`. |
| Val loss rising while train loss falls | Overfitting: too little data for the model. Get more data, use a smaller model, or train fewer steps. |
| Output repeats the same phrase | Raise `--temperature` (0.8–1.0) or lower `--top-p`. |

---

## 9. Data jargon cheat-sheet

| Term | Meaning, in this repo's terms |
|---|---|
| **Corpus** | The training text (`data/ww2.txt`). |
| **Dump** | A full snapshot of a website's content (Wikipedia's XML). |
| **Extraction** | Turning markup into text (`WikiText`). Real pipelines also strip HTML from web crawls such as Common Crawl. |
| **Normalisation** | Making text consistent: Unicode, whitespace, quotes (`TextClean`). |
| **Deduplication** | Removing repeated documents or lines so the model doesn't memorise them. |
| **Quality filtering** | Keeping "good" documents: here `--min-chars` and category filters; at scale, classifier models. |
| **Data mixture** | The recipe of sources. GPT-3's was about 60% filtered web crawl, 22% curated web pages, 16% books and 3% Wikipedia. |
| **Train/val split** | 90% of the corpus to learn from, 10% held out to measure honest progress. |
| **Epoch** | One full pass over the corpus. LLMs often train for about 1 epoch or less. |
| **Tokens per parameter** | Chinchilla's guideline is about 20. It tells you how big a model your data deserves. |
| **Data contamination** | Test questions leaking into training data, which makes a model look smarter than it is. |
| **Fine-tuning** | `--resume` a model trained on general text, then continue on a small specialised corpus. Try: Wikipedia → WWII letters. |

---

*Wikipedia content is available under CC BY-SA 4.0. If you publish a model or text generated from
it, credit Wikipedia. This repo's code is MIT-licensed.*
