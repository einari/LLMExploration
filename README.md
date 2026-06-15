# LLM Exploration

A complete, from-scratch implementation of a GPT-style language model in C# — no ML libraries, no GPU required.

This project is meant to be **read as much as run**. Every layer, every gradient, every matrix multiply is written out explicitly so you can trace exactly what happens when a language model thinks.

---

## Quick Start

### 1. Train the model

```bash
cd src/LLMExploration.Training
dotnet run
```

This reads `data/input.txt`, trains for 2,000 steps, and saves `model.bin` and `tokenizer.json` to the current directory. Training takes a few minutes on a modern CPU.

To train on your own text, replace `data/input.txt` with any `.txt` file before running.

### 2. Chat with it

```bash
cd src/LLMExploration.Chat
dotnet run
```

Type any text and the model will continue it, character by character. Because this is a tiny model trained on a small text, it will echo the *style* of whatever you trained on, not answer questions intelligently. That is expected — and educational.

**Chat commands**

| Command | What it does |
|---|---|
| `/temp 0.5` | Lower temperature → more predictable output |
| `/temp 1.5` | Higher temperature → more creative / random output |
| `/tokens 300` | Generate up to 300 characters |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Project Structure

```
LLMExploration/
├── data/
│   └── input.txt                       ← your training text goes here
└── src/
    ├── LLMExploration.Core/            ← the model, no dependencies on anything external
    │   ├── Tensor.cs                   ← how numbers are stored and passed around
    │   ├── Tokenizer.cs                ← converts text to numbers and back
    │   ├── GPTConfig.cs                ← model size settings
    │   └── Layers/
    │       ├── Module.cs               ← base class every layer inherits from
    │       ├── Embedding.cs            ← turns token IDs into vectors
    │       ├── Linear.cs               ← a single learned linear transformation
    │       ├── LayerNorm.cs            ← keeps numbers well-behaved during training
    │       ├── CausalSelfAttention.cs  ← the core of every LLM
    │       ├── MLP.cs                  ← the feed-forward part of each block
    │       ├── Block.cs                ← one full transformer block (attention + MLP)
    │       └── GPT.cs                  ← the complete model
    ├── LLMExploration.Training/
    │   ├── AdamW.cs                    ← the optimizer that updates weights
    │   ├── DataLoader.cs               ← feeds text to the model during training
    │   ├── Trainer.cs                  ← the training loop
    │   └── Program.cs                  ← entry point for training
    └── LLMExploration.Chat/
        └── Program.cs                  ← interactive chat console
```

---

## How It Works — From First Principles

This section explains every concept in the model from scratch, with no assumed background. Each explanation links to the file where that concept is implemented.

### The big picture

A language model has one job: **given some text, predict what character comes next**.

That's it. Do that well enough, on enough text, and the model learns grammar, facts, style, and reasoning as a side effect — because all of those things help predict the next character.

During **training**, we show the model billions of examples of "here is some text, what comes next?" and nudge its internal numbers every time it gets it wrong.

During **inference** (generation), we give it a starting prompt, ask it what comes next, append that character, ask again, and repeat. This is called *autoregressive* generation.

---

### Step 1 — Tokenization (`Tokenizer.cs`)

Computers work with numbers, not text. Before the model can do anything, every character must be converted to an integer.

This project uses **character-level tokenization**: every unique character in the training text gets its own number (called a *token ID*).

```
Training text contains: a b c d ... x y z A B ... space , . \n
                        ↓
Build a lookup table:   'a'→0  'b'→1  'c'→2  ...  '\n'→62

Encode "Hello":   [7, 4, 11, 11, 14]
Decode [7,4,...]: "Hello"
```

The full set of known tokens is the **vocabulary**. A model trained on Shakespeare might have ~65 unique characters, giving a vocabulary size of 65.

Real-world LLMs (GPT-4, Claude) use *sub-word* tokenizers that split text into chunks like "un", "believ", "able" — giving vocabularies of 50,000–100,000 tokens. The math is identical; only the granularity changes.

---

### Step 2 — Tensors (`Tensor.cs`)

A **tensor** is just a multi-dimensional array of floating-point numbers. This project implements its own from scratch.

```
A 1D tensor: [0.1, 0.5, -0.3]           ← shape [3]
A 2D tensor: [[0.1, 0.5],               ← shape [2, 3]
              [-0.3, 0.9],
              [0.2, 0.4]]
A 3D tensor: shape [batch, sequence, features]
```

Internally all tensors are stored as a flat `float[]` in **row-major order**: for shape `[2, 3, 4]`, element `[b, t, f]` lives at position `b*3*4 + t*4 + f`.

Every trainable tensor also carries a `Grad` array of the same size. This is where **gradients** accumulate during the backward pass (explained in the Training section).

---

### Step 3 — Embeddings (`Embedding.cs`)

Token IDs are integers. The model needs continuous vectors it can do math on. An **embedding** is a lookup table that maps each token ID to a learned vector.

```
Vocabulary size: 65 characters
Embedding dimension (NEmbedding): 64

Embedding table shape: [65, 64]
  Each row is a 64-dimensional vector for one character.

Token 'H' (ID=7)  →  [0.12, -0.43, 0.88, ..., 0.03]  (64 numbers)
Token 'e' (ID=4)  →  [-0.5,  0.17, 0.22, ..., 0.91]
```

Before training, these vectors are random noise. After training, similar characters end up with similar vectors — the model has learned that 'a', 'e', 'i', 'o', 'u' are all vowels, without being told.

The model also has a **positional embedding**: a second lookup table with one vector per position (0, 1, 2, ... up to `BlockSize`). The token embedding and position embedding are added together so the model knows both *what* a token is and *where* it appears in the sequence.

---

### Step 4 — The Transformer Block (`Block.cs`, `CausalSelfAttention.cs`, `MLP.cs`)

The model stacks several identical **transformer blocks**. Each block has two parts: an attention layer and a feed-forward network. Here is what each does.

#### Attention — how tokens talk to each other

Without attention, each position in the sequence would be processed independently. The model would not know that the word "bank" means something different in "river bank" versus "bank account".

**Self-attention** lets every token look at every other token and decide what information to borrow from it.

Here is the intuition:

> Imagine you are reading a sentence and trying to understand the word "it". You naturally scan back through the sentence looking for what "it" refers to. Attention teaches the model to do exactly that — and to do it for every word simultaneously.

The mechanism works as follows (see `CausalSelfAttention.cs`):

**1. Project to Query, Key, Value**

For each token, we compute three vectors by multiplying its embedding by three different learned weight matrices:

- **Query (Q)** — "What information am I looking for?"
- **Key (K)** — "What information do I contain?"
- **Value (V)** — "What will I actually share if someone attends to me?"

**2. Compute attention scores**

Each token's Query is compared against every other token's Key using a dot product. A high dot product means "these two tokens are relevant to each other".

```
score[i, j] = Q[i] · K[j] / sqrt(head_dim)
```

The division by `sqrt(head_dim)` prevents the dot products from getting so large that the subsequent softmax saturates.

**3. Causal mask**

This is a language model, not a translator — it can only see past tokens when predicting the next one. We force the score for every future token to `-infinity` so it gets zeroed out by softmax. This is the "causal" in "causal self-attention".

```
Token 0 can attend to: [0]
Token 1 can attend to: [0, 1]
Token 2 can attend to: [0, 1, 2]
...
```

**4. Softmax → attention weights**

Softmax converts the scores into a probability distribution over positions. These are the **attention weights**: how much each token should borrow from each other token.

**5. Weighted sum of Values**

Each token's output is a weighted average of all Value vectors, where the weights came from step 4. Tokens that were deemed relevant contribute more; masked-out future tokens contribute nothing.

**6. Multi-head**

Instead of running this once, we run it `NHead` times in parallel, each with its own Q/K/V weight matrices. Each "head" can specialize — one head might track grammatical agreement, another might track coreference, another might notice rhyme patterns.

The outputs of all heads are concatenated and projected through a final linear layer.

#### Feed-Forward Network (MLP)

After attention, each token is processed independently through a small two-layer network:

```
x → Linear(64→256) → GELU → Linear(256→64) → output
```

The intermediate size (4× the embedding dimension) gives the model room to do more complex per-token computation. Think of attention as the "communication" step where tokens share information, and the MLP as the "thinking" step where each token processes what it learned.

**GELU** (Gaussian Error Linear Unit) is the activation function that introduces non-linearity. Without it, stacking linear layers would collapse to a single linear transformation, no matter how deep.

#### Residual connections and Layer Normalization

Both the attention layer and the MLP use **residual connections**: the input is added back to the output.

```
x = x + attention(layernorm(x))
x = x + mlp(layernorm(x))
```

This gives gradients a "highway" to travel through during backpropagation, making it possible to train deep networks without the signal vanishing.

**Layer Normalization** (`LayerNorm.cs`) normalizes each token's embedding vector to have zero mean and unit variance, then applies learned scale (`gamma`) and shift (`beta`) parameters. This keeps activations in a stable range throughout training.

---

### Step 5 — The Full Model (`GPT.cs`)

The complete forward pass through the model:

```
Input: token IDs  [t0, t1, t2, ..., tT]
         ↓
Token Embedding    [T, 64]   ← look up each token's vector
  + Positional Embedding     ← add a vector for each position
         ↓
Transformer Block 0          ← attention + MLP + residuals
Transformer Block 1
Transformer Block 2
Transformer Block 3
         ↓
Final Layer Norm             ← stabilize before final projection
         ↓
Linear (64 → VocabSize)      ← project to one score per character
         ↓
Output: logits  [T, 65]      ← one vector of 65 scores per position
```

The output **logits** are the model's raw scores for every possible next character at every position. Higher score = model thinks that character is more likely to come next.

---

### Step 6 — Training (`AdamW.cs`, `Trainer.cs`, `DataLoader.cs`)

Training adjusts every weight in the model so that the logits get better at predicting the next character.

#### The loss function

We convert logits to probabilities using softmax, then measure how surprised the model was by the actual next character. This is **cross-entropy loss**:

```
loss = -log(probability assigned to the correct next character)
```

- If the model gave the correct character a probability of 1.0, loss = 0 (perfect).
- If the model gave it a probability of 0.01, loss = 4.6 (very surprised).
- At the start of training with random weights, loss ≈ log(VocabSize) ≈ 4.17 for 65 characters. The program prints this as a sanity check.

The loss is averaged over all positions in the sequence. Lower loss = better model.

#### Backpropagation

Once we have the loss, we need to know how to adjust each of the thousands of weights in the model to reduce it. This is done with the **chain rule from calculus**, applied backwards through the computation graph.

For the combined cross-entropy + softmax, the gradient has a beautiful form:

```
dLogits[t, v] = (softmax(logits[t])[v]  −  1 if v==target else 0)  /  T
```

In words: push the probability of the correct character up, and push everything else down, in proportion to how much probability each character currently has.

This gradient then flows backwards through every layer:

```
dLogits → lm_head.Backward() → layernorm.Backward() → block[3].Backward() → ... → embedding.Backward()
```

Each layer's `Backward()` method computes two things:
1. The gradient to pass to the layer below it (so the chain continues)
2. The gradient for its own parameters (so those can be updated)

Every `Backward()` method in this project is implemented by hand, with comments explaining the math. `CausalSelfAttention.Backward()` is the most instructive to study.

#### The optimizer (AdamW)

After backpropagation, every parameter has an accumulated gradient. The **AdamW optimizer** (`AdamW.cs`) translates those gradients into weight updates:

```
For each parameter θ with gradient g:
  m = 0.9 * m + 0.1 * g          ← running average of gradient (momentum)
  v = 0.999 * v + 0.001 * g²     ← running average of gradient squared
  θ = θ - lr * m̂ / (√v̂ + ε)    ← update (bias-corrected)
  θ = θ * (1 - lr * weight_decay) ← weight decay (regularization)
```

The momentum (`m`) smooths out noisy gradients. The variance estimate (`v`) scales the learning rate per-parameter — parameters with consistently large gradients get a smaller effective learning rate. Together they make training much faster and more stable than plain gradient descent.

---

### Step 7 — Generation (`GPT.Generate()`)

After training, generating text works like this:

1. Start with a prompt (e.g. `"Once upon"`)
2. Encode it to token IDs
3. Run the forward pass → get logits for the last position
4. Apply **temperature**: divide logits by `temperature` before softmax
   - Temperature 0.5 → sharper distribution → model picks safer, more predictable characters
   - Temperature 1.5 → flatter distribution → model takes more risks, more creative
5. Convert logits to probabilities with softmax
6. **Sample** from this probability distribution (randomly draw one character, weighted by probability)
7. Append the sampled character to the context
8. Go to step 3, repeat until done

This is called **autoregressive** generation: each generated token becomes part of the input for the next step.

---

## Default Model Configuration

| Setting | Value | What it means |
|---|---|---|
| `BlockSize` | 64 | The model sees up to 64 characters at once |
| `NEmbedding` | 64 | Each token is represented as a 64-dimensional vector |
| `NHead` | 4 | 4 parallel attention heads (each sees 16 dimensions) |
| `NLayer` | 4 | 4 transformer blocks stacked |
| `VocabSize` | ~65 | Number of unique characters in training data |

Total parameters: roughly **~400,000**. GPT-2 (small) has 117 million. GPT-4 has hundreds of billions. This model is tiny by design — it trains in minutes on a CPU and the entire thing fits in your head.

To make the model larger, increase `NEmbedding`, `NHead`, and `NLayer` in `Training/Program.cs`. Training will take longer but the model will handle longer-range patterns.

---

## What to Read Next

If you want to go deeper, read these files in this order:

1. **`Tensor.cs`** — understand the data structure everything is built on
2. **`Tokenizer.cs`** — how text becomes integers
3. **`Embedding.cs`** — how integers become vectors
4. **`Linear.cs`** — the fundamental building block, with forward and backward
5. **`LayerNorm.cs`** — normalisation, and its surprisingly involved backward pass
6. **`CausalSelfAttention.cs`** — the heart of every LLM; read the comments carefully
7. **`MLP.cs`** — GELU and the feed-forward network
8. **`Block.cs`** — how the pieces assemble into one transformer block
9. **`GPT.cs`** — the full model, loss computation, and generation
10. **`AdamW.cs`** — the optimizer
11. **`Trainer.cs`** — the training loop tying everything together

---

## Further Learning

- **Andrej Karpathy's "Neural Networks: Zero to Hero"** — the Python equivalent of this project, with video explanations. Highly recommended companion.
- **"Attention Is All You Need"** (Vaswani et al., 2017) — the original transformer paper.
- **The Annotated Transformer** — the paper with line-by-line code annotations.
