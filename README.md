# eto — a pure‑Java tiny LLM + GGUF toolchain

`eto` is a complete, dependency‑free Java project for **training your own tiny
GGUF language model** and **running it (or a third‑party GGUF)** from a CLI
chatbot. There is no Maven, Gradle, native code, or external jar. Only the JDK.

```
data/*.txt ──trainer.sh──> model.gguf ──runner.sh──> CLI chatbot
                               │
                               └── also loads llama / mistral / qwen2 GGUFs
```

## Requirements

- JDK 17 or newer (`javac`, `java`, `jar` on `PATH`)
- Linux/macOS shell (the convenience scripts are bash)

```bash
./build.sh
```

## Quick start

```bash
./build.sh                 # compile with javac only
./trainer.sh               # train the demo model (~25 s on a laptop)
./runner.sh                # chat with models/eto-tiny.gguf
```

Train with your own text and size:

```bash
./trainer.sh --data mybook.txt --out models/my.gguf \
             --steps 1500 --ctx 96 --dim 96 --layers 2 --heads 4 --ffn 192
```

One-shot generation without the chat wrapper:

```bash
./generate.sh --model models/eto-tiny.gguf \
              --prompt "The little machine" --tokens 120 --temp 0.7
```

Inspect any GGUF:

```bash
./info.sh --model models/eto-tiny.gguf
```

Prepare the balanced English corpus and train Eto with a resumable checkpoint:

```bash
python3 data/prepare_corpus.py --output .training/english_training.txt
./trainer.sh --data .training/english_training.txt --out models/eto.gguf --name Eto \
             --steps 50000 --ctx 256 --checkpoint .training/checkpoint.bin
./trainer.sh --data .training/english_training.txt --out models/eto.gguf --name Eto \
             --steps 50000 --ctx 256 --resume .training/checkpoint.bin
```

Each invocation runs the requested number of additional steps. Checkpoints
include model weights and AdamW state and are only accepted with matching model
dimensions, batch size, seed, and training data. The `Train Eto`
GitHub Actions workflow runs on relevant pushes or manual dispatch, restores its
previous checkpoint from the Actions cache, and uploads the trained GGUF as an
artifact. It uses balanced excerpts from the included Project Gutenberg books,
grammar and conversation corpora, Minecraft knowledge, and the Eto vocabulary.
The chat runner automatically uses the matching `user:` / `Eto:` prompt format
for models whose GGUF name is `Eto`.
The Minecraft sources are CC BY-NC-SA 3.0; treat the resulting corpus and
trained artifact as noncommercial share-alike material. See
`data/MINECRAFT-NOTICE.md` before redistributing or using the artifact. This is
not a complete English-language dataset. The byte-level trainer and tiny model
are educational and cannot learn broad English fluency from this corpus.

Verify the trainer's analytic backprop:

```bash
java -cp build eto.cli.GradCheck
# GRADCHECK: OK
```

## The model

The trainer implements a small **llama‑style decoder** from scratch:

- byte‑level tokenizer: 256 byte tokens + BOS/EOS (no vocabulary training needed)
- RMSNorm, rotary position embeddings (RoPE), causal self‑attention
- SwiGLU feed‑forward blocks
- AdamW, warmup + cosine learning‑rate decay, gradient clipping
- analytic backprop for every operation (checked numerically)

Default demo: 2 layers, 64 embedding width, 4 heads, 128 FFN width, 64-token
context, 115,264 parameters, 0.45 MiB GGUF. Training on the included 3 KB story
for 500 steps takes about 25 seconds on a CPU.

It is a real language model, but deliberately tiny. It learns the style and
local patterns of its training text; it will not reason like a 7B model.

## GGUF output

The trainer writes **GGUF v3** exactly as specified:

- little‑endian header, metadata key/value pairs, tensor infos
- 32‑byte alignment
- `general.architecture = llama`
- conventional llama tensor names (`token_embd.weight`, `blk.N.attn_q.weight`, …)
- F32 tensors (lossless for training output)
- q/k rows permuted to the interleaved layout used by llama.cpp‑style GGUF
  files, so the same inference engine runs the demo model and third‑party GGUFs
- GPT‑2 byte vocabulary in `tokenizer.ggml.tokens`, with BOS/EOS metadata

`info.sh` prints the full metadata and tensor table.

## Third‑party GGUF support

`runner.sh` / `LlamaModel` is a pure‑Java port of the pieces needed to run
common GGUF models. It does **not** shell out to `llama.cpp` or use a jar.

| Area | Supported |
|---|---|
| Architectures | `llama`, `mistral`, `qwen2` (SwiGLU + RMSNorm + RoPE) |
| Attention | MHA and GQA, optional q/k/v/output biases (Qwen2) |
| Tensor types | F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K |
| Tokenizers | GPT‑2 byte‑level BPE, SentencePiece/LLaMA BPE with `<0xNN>` fallback; Qwen2 and Llama‑3 pre‑tokenizer rules; rank‑ordered SPM BPE when a GGUF has merge ranks but zero scores |
| Metadata | BOS/EOS, add‑bos, rope base, linear rope scaling, head dimensions |
| RoPE | llama.cpp `NORM` (adjacent pairs, `llama`/`mistral`) and `NEOX` (half‑split, `qwen2`) layouts |
| Chat templates | auto, plain, ChatML, Llama‑2, Llama‑3, Zephyr/TinyLlama, Alpaca, raw |

Try a real model:

```bash
./runner.sh --model /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
            --ctx 1024 --tokens 256 --temp 0.7

./runner.sh --model /path/to/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
            --ctx 1024 --tokens 256 --temp 0.7
```

### What is intentionally not supported

This is a compact educational engine, not a full `llama.cpp` replacement.

- other architectures: Gemma, Phi, MoE models, state‑space models, etc.
- IQ*/TQ* and other exotic quantizations not listed above
- non‑linear rope scaling (`llama3`, `yarn`, `dynamic`) — a warning is printed
  and unscaled RoPE is used
- Jinja chat templates are not evaluated. `--template` selects a built‑in
  format; the GGUF's `tokenizer.chat_template` metadata is used only for
  auto‑detection.
- a *single tensor* larger than 2 GiB (the reader maps each tensor separately,
  so model files larger than 2 GiB are fine for normal 7B/8B/13B models)
- GPU acceleration (CPU only; the hot matrix‑vector loops do use all cores)

For a 0.5B Q4 model, expect a few tokens per second; for 7B, this pure‑Java
engine is slow. The tiny `eto` model is the intended fast demo.

### Performance knobs

```bash
ETO_HEAP=8g ETO_THREADS=8 ./runner.sh --model model.gguf
```

`ETO_HEAP` sets the JVM heap (default `4g`). `ETO_THREADS` overrides the number
of threads used for the matrix‑vector products (default: all available cores).
The reader memory‑maps each tensor separately, so the model itself is not
copied into the heap.

## Chat commands

```
/exit, /quit      quit
/reset            clear history
/system TEXT      set the system prompt
/temp F           temperature (0 = greedy)
/topk N           top-k
/topp F           top-p
/repeat F         repetition penalty
/tokens N         maximum new tokens
/ctx N            context window (capped by the model)
/model            show settings
/help             help
```

Options: `--model`, `--ctx`, `--tokens`, `--temp`, `--top-k`, `--top-p`,
`--repeat`, `--seed`, `--system`, `--template`, `--prompt`, `--greedy`,
`--raw`.

Templates: `auto` (default), `none`, `chatml`/`qwen`, `llama2`/`mistral`,
`llama3`, `zephyr`/`tinyllama`, `alpaca`, `raw`. Auto‑detection uses both the
tokenizer vocabulary and the GGUF's `tokenizer.chat_template` metadata.

## Project layout

```
build.sh            javac + jar only
trainer.sh          train and write GGUF
runner.sh           interactive chat (own + third-party GGUF)
generate.sh         one-shot completion
info.sh             dump GGUF metadata/tensors
src/eto/nn/         tensor math, blocks, TinyLlama, AdamW, GGUF export
src/eto/train/      training loop
src/eto/gguf/       GGUF v2/v3 reader/writer, lazy tensors, dequantizers
src/eto/tokenizer/  byte, GPT-2 BPE, SentencePiece BPE
src/eto/infer/      generic llama engine, KV cache, sampler
src/eto/cli/        train/chat/generate/info/gradcheck entry points
data/sample.txt     demo corpus
models/             generated GGUFs
```

## How training works

Each optimizer step samples `batch` random windows from the byte stream,
prepends BOS / appends EOS once at the file ends, and predicts the next byte at
every position. Loss is mean cross‑entropy. Gradients are computed analytically
through attention, RoPE, RMSNorm, SwiGLU and the output projection, averaged
over the batch, clipped to norm 1, and applied with AdamW. Parameters are saved
as F32 GGUF tensors.

## Attribution

The GGUF container layout and the quantization block formats follow the public
GGML/llama.cpp specification and reference dequantizers:

- llama.cpp / ggml — MIT License — <https://github.com/ggml-org/llama.cpp>

The implementations in `src/eto/gguf/` are independent Java ports; no C code,
jars, or native libraries are linked or invoked.
