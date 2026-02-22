# nanollama — Beginner's Guide

You know ChatGPT? nanollama lets you train your own version of that — from nothing, on rented GPUs, using your own data. You end up with a real language model that runs on your laptop, with zero internet connection required.

This guide assumes you have never trained an AI model before.

---

## What is nanollama?

nanollama is a training framework. It is not a model you download and run — it is a set of tools that lets you build a model yourself. You pick a size, rent a GPU for a few hours, run one command, and get a trained model file back. That file runs on any machine with the included Go inference engine.

The full pipeline, in order:

1. Download training data (text from the web)
2. Train the model (the GPU does the heavy work)
3. Export the trained model to a GGUF file (a portable format, like .mp3 but for AI models)
4. Copy the file to your machine
5. Run it with the Go engine — zero dependencies, works on any OS

---

## What you will need

- A **Lambda Cloud account** — this is where you rent the GPU. Sign up at https://lambdalabs.com. You will need a credit card. Costs for small models: $5-15 total.
- A **computer** to run the result on. Any modern laptop works. Linux, macOS, or Windows.
- Basic **terminal skills** — you need to know how to open a terminal, type commands, and use SSH. If you can `cd` into a folder and run a script, you are fine.
- **Go 1.21 or newer** installed locally, to build the inference engine. Download from https://go.dev/dl/.

You do NOT need to know Python. You do not need to understand machine learning. You do not need a GPU in your own machine.

---

## GPU glossary (read this once)

- **GPU** — Graphics Processing Unit. Originally made for video games, now the main hardware for training AI. Much faster than a regular CPU for matrix math.
- **LLM** — Large Language Model. The type of AI behind ChatGPT, Claude, etc. nanollama trains a smaller version of this.
- **GGUF** — A file format for storing trained language models. Think of it like .mp3 for audio: it is a container that holds the model's learned weights in a portable, compressed form.
- **H100** — NVIDIA's current high-end GPU. Lambda Cloud rents these by the hour. One H100 is enough for nano, micro, mini, and small.
- **Loss** — A number that measures how wrong the model is. Lower is better. During training you will see it decrease. That means the model is learning.

---

## Step 1: Rent a GPU on Lambda Cloud

1. Go to https://lambdalabs.com and create an account.
2. Add a payment method.
3. Go to **Instances** and click **Launch Instance**.
4. Pick a GPU type based on which model you want to train:

   | Model size | Params | Training time | GPU to pick |
   |------------|--------|---------------|-------------|
   | nano | 46M | ~30 min | 1x H100 SXM5 |
   | micro | 87M | ~1 hour | 1x H100 SXM5 |
   | mini | 175M | ~3 hours | 1x H100 SXM5 |
   | small | 338M | ~18 hours | 1x H100 SXM5 |
   | goldie | 1.1B | ~24 hours | 1-2x H100 |
   | medium | 1.6B | ~48 hours | 4x H100 |
   | large | 3.7B | ~96 hours | 8x H100 |
   | big | 7.0B | ~200 hours | 8x H100 |

   **Start with nano.** It costs around $3-5 and finishes in half an hour. Once you have the process down, scale up.

5. When the instance is ready, Lambda will show you an IP address and SSH instructions.

6. Add your SSH public key in the Lambda dashboard (under SSH Keys) before launching. If you do not have one:

   ```bash
   ssh-keygen -t ed25519 -C "your@email.com"
   cat ~/.ssh/id_ed25519.pub
   ```

   Copy the output and paste it into Lambda's SSH key field.

7. Connect to your instance:

   ```bash
   ssh ubuntu@<your-instance-ip>
   ```

   You should see a Linux shell prompt. You are now on the GPU machine.

---

## Step 2: Set up the environment

Run this one command on the Lambda instance. It installs everything: Python environment, PyTorch, and the nanollama code.

```bash
curl -sSf https://raw.githubusercontent.com/ariannamethod/nanollama/main/runs/lambda_setup.sh | bash
```

Or if you prefer to clone first and run locally:

```bash
git clone https://github.com/ariannamethod/nanollama.git
cd nanollama
bash runs/lambda_setup.sh
```

The setup script:
- Checks your GPU is visible
- Installs `uv` (a fast Python package manager)
- Creates a Python virtual environment
- Installs PyTorch with CUDA support
- Installs sentencepiece, numpy, and other minimal dependencies
- Configures multi-GPU networking (NCCL)

Expected output at the end:

```
PyTorch: 2.4.1+cu124
CUDA available: True
CUDA version: 12.4
SentencePiece: OK
NumPy: 1.26.4
========================================
  Setup complete!

  Full training run:
    bash runs/lambda_train.sh --name nano --base-only
========================================
```

If you see `CUDA available: True`, you are ready. If it says `False`, something is wrong with the GPU setup — stop and check that your instance type actually has a GPU.

---

## Step 3: Train your first model

Make sure you are inside the `nanollama` directory on the Lambda instance:

```bash
cd ~/nanollama
source .venv/bin/activate
```

Now run training. Start with nano — it is the smallest and cheapest:

```bash
bash runs/lambda_train.sh --name nano --base-only
```

The `--base-only` flag means: train the base model only, skip the personality pipeline. Good for your first run.

The script will:
1. Download ~380 million tokens of educational web text (FineWeb-Edu dataset)
2. Train the 46M parameter model
3. Export the result to a GGUF file

### What the output looks like

During data download:
```
>> Step 1: Downloading FineWeb-Edu (350000 samples)...
```

When training starts, you will see lines like this:
```
step     1 | loss 10.8234 | lr 6.00e-07 | tokens/s 1.02M | MFU 27.3%
step   100 | loss  7.3421 | lr 6.00e-05 | tokens/s 1.04M | MFU 28.1%
step   500 | loss  5.1209 | lr 3.00e-04 | tokens/s 1.03M | MFU 28.4%
step  1000 | loss  4.2817 | lr 3.00e-04 | tokens/s 1.03M | MFU 28.5%
step  2000 | loss  3.8134 | lr 2.40e-04 | tokens/s 1.03M | MFU 28.5%
step  5000 | loss  3.1200 | lr 6.00e-05 | tokens/s 1.03M | MFU 28.5%
```

**Loss going down = the model is learning.** Starting around 10, ending around 3. That is normal and correct.

- `loss` — lower is better. Below 4 means the model has learned real language patterns. Below 3.5 is a solid result for this size.
- `tokens/s` — training speed. For nano on a single H100, expect around 1 million tokens per second.
- `MFU` — Model FLOPs Utilization. How efficiently the GPU is being used. 28-35% is typical.

When done:
```
========================================================
  DONE — nano base (46M)
========================================================

  weights/nano-base-f16.gguf

  scp ubuntu@<ip>:~/nanollama/weights/nano-* .
  ./nanollama --model nano-base-f16.gguf --interactive
```

Training is done. Your model is sitting in `weights/nano-base-f16.gguf` on the Lambda instance.

---

## Step 4: Copy the model to your machine

The training output shows you the exact scp command. Run it on your local machine (not on the Lambda instance):

```bash
scp ubuntu@<your-instance-ip>:~/nanollama/weights/nano-* .
```

This copies all the weight files (the GGUF model) to your current directory.

Check that the file arrived and is not empty:

```bash
ls -lh nano-base-f16.gguf
```

You should see something like:
```
-rw-r--r--  1 you  staff    88M  nano-base-f16.gguf
```

A nano model in F16 format is about 88 megabytes. If the file is 0 bytes, something went wrong with the transfer — try the scp command again.

**What is GGUF?** It is a file format designed for storing language models efficiently. All the learned knowledge of your model — billions of numbers representing patterns in language — is packed into this one file. You can share it, load it into llama.cpp, or run it with the nanollama Go engine.

**Now you can terminate your Lambda instance.** Once you have the .gguf file, you no longer need the GPU. Lambda charges by the hour, so shut it down to stop being billed.

---

## Step 5: Run your model locally

The nanollama Go inference engine runs your model with zero external dependencies. No Python, no PyTorch, no CUDA needed. It is a single ~9MB binary.

### Build the engine

You need Go 1.21+ installed. Then:

```bash
cd go && go build -o nanollama .
```

This produces a binary called `nanollama` in the `go/` directory. Move it wherever you like, or run it from there.

### Run it

One-shot generation:

```bash
./nanollama --model nano-base-f16.gguf --prompt "Once upon a time"
```

Interactive mode (type prompts, get responses, type `/quit` to exit):

```bash
./nanollama --model nano-base-f16.gguf --interactive
```

Web chat UI in your browser:

```bash
./nanollama --model nano-base-f16.gguf --serve --port 8080
```

Then open http://localhost:8080 in your browser.

### Example interaction

```
> ./nanollama --model nano-base-f16.gguf --interactive

[nanollama] loading nano-base-f16.gguf
[nanollama] ready — 46M params, 12 layers, 384 dim
nanollama interactive mode. Type /quit to exit.

> The most important thing about learning is
that it is a very personal experience. The whole concept of education is to be
rooted in the inner connection of the individual to the world and to the world.
Education is the process of the individual being taught to the world through the
interaction of the individual and the world.
[52 tokens, 47.3 tok/s]

> /quit
```

### Useful flags

```
--max-tokens 512     Generate up to 512 tokens (default: 256)
--temp 0.7           Lower temperature = more predictable output (default: 0.8)
--temp 1.2           Higher temperature = more creative, more random
--top-p 0.9          Nucleus sampling threshold (default: 0.9)
--rep-penalty 1.15   Penalize repeated phrases (default: 1.15)
```

---

## Honest expectations by model size

Before you invest time and money training larger models, here is what to actually expect:

| Size | Params | What to expect |
|------|--------|----------------|
| **nano** (46M) | 46M | Grammatical sentences, mostly coherent paragraphs. Will repeat itself, make stuff up, lose the thread. Good for understanding the pipeline. |
| **micro** (87M) | 87M | Noticeably better coherence. Still makes things up constantly. Fun to play with, not useful for real tasks. |
| **mini** (175M) | 175M | Starts to be interesting. Can maintain a topic for several sentences. Outputs look like real text. Still not reliable. |
| **small** (338M) | 338M | Actually useful for text generation tasks. Decent coherence, reasonable factual grounding within its training data. Worth the 18 hours. |
| **goldie** (1.1B) | 1.1B | Real quality. Multilingual (English + Russian + French + German). This is where things get genuinely impressive. |
| **medium+** | 1.6B-7B | Serious models. Results comparable to early GPT-2 era models at the top end. Requires multiple GPUs and significant budget. |

The honest truth: nano and micro are learning exercises. mini starts to show you what training from scratch can do. small is the minimum for practical use. goldie is where you would actually show people results.

---

## Step 6: Level up — train bigger and add personality

### Train a bigger model

Same command, different name:

```bash
bash runs/lambda_train.sh --name mini
bash runs/lambda_train.sh --name small
bash runs/lambda_train.sh --name goldie
```

mini and small use a richer training corpus (FineWeb-Edu + DCLM + code + math) instead of just FineWeb-Edu. The script handles this automatically.

For goldie and above, use 4x H100 or more. The training script detects how many GPUs are available and uses distributed training automatically.

### Add personality

This is one of nanollama's unique features. You can train a model that has a specific personality — a distinct voice, way of thinking, set of interests.

**How it works, in plain English:** You train two identical models on the same data. One gets your regular training data. The other gets the same data, but with 20% of each training batch replaced by your personality text (conversations, writing, whatever you want the model to sound like). Then you subtract the weights of the base model from the personality model. What is left is the "personality vector" — called gamma. You can add this gamma to any compatible base model.

**What to prepare:** A JSONL file where each line is a JSON object with a `"text"` field. These are the texts that define the personality. Could be your own writing, conversation logs, a fictional character's dialogue — anything.

Example JSONL format:
```
{"text": "The question isn't whether we can, but whether we should."}
{"text": "I find that most problems have simpler solutions than people assume."}
{"text": "Let me think about this differently..."}
```

**Train with personality:**

```bash
# Upload your personality file to the Lambda instance first
scp my_personality.jsonl ubuntu@<ip>:~/

# Then train
bash runs/lambda_train.sh --name nano --personality my_personality.jsonl
```

This runs the full pipeline: base model, then personality model, then extracts gamma, then exports both models and the gamma file.

Output files:
```
weights/nano-f16.gguf         — personality model
weights/nano-base-f16.gguf    — base model (for comparison)
weights/gamma_nano.npz        — personality vector (~17MB)
```

**Run with personality:**

```bash
# With the personality baked in (personality model)
./nanollama --model nano-f16.gguf --interactive

# With gamma injected at runtime into the base model (same result, different method)
./nanollama --model nano-base-f16.gguf --gamma gamma_nano.npz --interactive
```

The gamma injection approach is useful because you can reuse one small gamma file across multiple base model sizes, as long as the architecture matches.

---

## FAQ

**How much does this cost?**

Rough estimates for Lambda Cloud H100 at current rates (check Lambda for current pricing):

| Model | Training time | Approximate cost |
|-------|---------------|-----------------|
| nano | 30 min | ~$3-5 |
| micro | 1 hour | ~$6-10 |
| mini | 3 hours | ~$18-30 |
| small | 18 hours | ~$100-160 |
| goldie (1x H100) | 24 hours | ~$140-220 |
| medium (4x H100) | 48 hours | ~$1000-1600 |

Note: these are rough estimates. Personality training doubles the training time (you train the model twice). Always terminate your instance when done.

**Can I use my own data instead of the web text?**

Yes. The `--corpus` flag lets you override the data source. For custom data, you would prepare it in the tokenized binary format that nanollama expects. The relevant script is `data/prepare_fineweb.py` — you can adapt it for your own text files.

For personality data specifically, just provide a JSONL file with `"text"` fields.

**How good will my model be compared to ChatGPT?**

Honest answer: nowhere close, for small sizes. ChatGPT is based on models trained on vastly more data, with instruction tuning, RLHF, and much larger compute budgets. nano is 10,000x smaller than GPT-4 and trained on a fraction of the data.

What nanollama gives you is a model you built yourself, that you understand, that you can inspect and modify. That is the point. The quality at goldie and above starts to be genuinely impressive for what a single person can produce.

**Do I need to know Python?**

For basic use (train a model, run it): no. The training pipeline is one bash command and the inference engine is Go. You need Python installed on the Lambda instance, but the setup script handles that.

If you want to understand or modify the training code, it is written in Python/PyTorch. But you can use the whole system without touching the Python.

**Do I need a GPU in my own machine?**

No. Training happens on the rented Lambda Cloud GPU. The final .gguf model runs on CPU — any modern laptop can do inference, just slower. Expect 20-100 tokens per second on a laptop CPU depending on model size.

**Can I use llama.cpp with my model?**

Yes. The GGUF files nanollama produces are fully llama.cpp compatible. You can load them with `llama-cli -m your-model.gguf -p "your prompt"`.

**What is the difference between training from scratch and fine-tuning?**

Fine-tuning takes an existing model (like Meta's Llama 3) and adjusts its weights on new data. This is faster and cheaper but you are building on someone else's foundation.

Training from scratch means starting with random weights and learning everything from your data. More expensive, more control, and your model belongs entirely to you — no licensing concerns, no restrictions.

**The training script printed a scp command but I do not know what to do with it.**

The scp command copies files from the Lambda instance to your local machine. Run it in a terminal on your laptop (not on the Lambda instance). Example:

```bash
scp ubuntu@1.2.3.4:~/nanollama/weights/nano-* .
```

This copies all files matching `nano-*` from the remote machine's `~/nanollama/weights/` directory to your current local directory (the `.` at the end means "here").

---

## Troubleshooting

**SSH key permissions error when connecting to Lambda:**

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
```

Fix: restrict permissions on your key file.

```bash
chmod 600 ~/.ssh/id_ed25519
```

**CUDA out of memory during training:**

This usually means the batch size is too large for your GPU. The script picks batch sizes automatically per model size, but if you have multiple models using the same GPU (you should not), or if you overrode something, reduce the batch size:

```bash
bash runs/lambda_train.sh --name nano --base-only
# If OOM: reduce steps or use a larger GPU instance
```

If you are getting OOM on a single nano run with a single H100, something unusual is wrong — check that no other processes are using the GPU (`nvidia-smi`).

**Training seems stuck — loss is not going down:**

A small amount of loss-not-moving is normal in the first 50-100 steps (warmup phase). If loss stays flat for 500+ steps, check:

- Is the GPU actually being used? Run `nvidia-smi` in another terminal and watch GPU utilization. Should be 90%+.
- Did data preparation actually complete? Look for lines like `Step 1: FineWeb-Edu data exists (N shards) — skip`. If N is 0 or 1, the data did not download properly. Delete the data directory and re-run.

**"CUDA available: False" after setup:**

This means PyTorch was installed without CUDA support, or the GPU driver is not visible. On Lambda instances this should not happen. Try:

```bash
nvidia-smi
```

If this fails, your instance may not have a GPU attached (rare, but happens). Terminate and launch a new one.

**The scp copy failed or the .gguf file is 0 bytes:**

The export step may not have completed. Check that training finished without errors by looking at the log file:

```bash
tail -50 train_nano_base.log
```

If training succeeded, try re-running just the export:

```bash
cd ~/nanollama
source .venv/bin/activate
python -m scripts.export_gguf \
  --checkpoint ~/.cache/nanollama/checkpoints/nano_base/checkpoint_step5000.pt \
  --tokenizer ~/.cache/nanollama/tokenizer/tokenizer.model \
  --output weights/nano-base-f16.gguf \
  --dtype f16
```

Then retry the scp command.

**The Go binary crashes with "model not found" or similar:**

Make sure you are pointing to the right file path:

```bash
ls -lh nano-base-f16.gguf
./nanollama --model nano-base-f16.gguf --prompt "test"
```

The model file must be in your current directory, or you must provide the full path to it.

---

## Quick reference

```bash
# Lambda Cloud: setup
bash runs/lambda_setup.sh

# Lambda Cloud: train
bash runs/lambda_train.sh --name nano --base-only
bash runs/lambda_train.sh --name mini
bash runs/lambda_train.sh --name nano --personality my_data.jsonl

# Local: build inference engine
cd go && go build -o nanollama .

# Local: run model
./nanollama --model nano-base-f16.gguf --prompt "Hello"
./nanollama --model nano-base-f16.gguf --interactive
./nanollama --model nano-base-f16.gguf --serve --port 8080
./nanollama --model nano-base-f16.gguf --gamma gamma_nano.npz --interactive

# Local: copy weights from Lambda (run on your machine, not Lambda)
scp ubuntu@<lambda-ip>:~/nanollama/weights/nano-* .
```

---

*nanollama is part of the [Arianna Method](https://github.com/ariannamethod) ecosystem.*
