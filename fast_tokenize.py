#!/usr/bin/env python3
"""Fast parallel tokenization for goldie corpus. All CPU cores."""
import os, sys, time
import multiprocessing as mp
import numpy as np

TOKENIZER = os.path.expanduser("~/.cache/nanollama/tokenizer_tier1/tokenizer.model")
OUTDIR = os.path.expanduser("~/.cache/nanollama/data/goldie_corpus")
SHARD_SIZE = 10_000_000

TARGETS = {
    "fineweb": 9_900_000_000,
    "fw2hq_ru": 3_740_000_000,
    "fw2hq_fr": 2_640_000_000,
    "fw2hq_de": 2_640_000_000,
    "stack": 1_980_000_000,
    "megamath": 1_100_000_000,
}

LOCAL = {
    "fw2hq_ru": os.path.expanduser("~/.cache/nanollama/data/fineweb2_hq/rus_Cyrl.txt"),
    "fw2hq_fr": os.path.expanduser("~/.cache/nanollama/data/fineweb2_hq/fra_Latn.txt"),
    "fw2hq_de": os.path.expanduser("~/.cache/nanollama/data/fineweb2_hq/deu_Latn.txt"),
}

def tokenize_chunk(args):
    import sentencepiece as spm
    lines, model = args
    sp = spm.SentencePieceProcessor(model_file=model)
    toks = []
    for l in lines:
        l = l.strip()
        if len(l) >= 50:
            toks.extend(sp.encode(l))
    return toks

def do_local(name, filepath, target, nworkers):
    comp_dir = os.path.join(OUTDIR, name)
    os.makedirs(comp_dir, exist_ok=True)
    existing = sorted([f for f in os.listdir(comp_dir) if f.endswith(".bin")])
    done = sum(os.path.getsize(os.path.join(comp_dir, f)) // 2 for f in existing)
    if done >= target * 0.95:
        print(f"[{name}] SKIP: {done:,} tokens already")
        return
    si = len(existing)
    print(f"[{name}] resume={done:,} target={target:,} workers={nworkers}")
    CHUNK = 10000
    buf = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        if done > 0:
            skip = done // 150
            print(f"[{name}] skipping ~{skip:,} lines...")
            for _ in range(skip):
                try: next(f)
                except StopIteration: break
        chunks = []
        cur = []
        for line in f:
            cur.append(line)
            if len(cur) >= CHUNK:
                chunks.append((cur, TOKENIZER))
                cur = []
                if len(chunks) >= nworkers:
                    with mp.Pool(nworkers) as pool:
                        results = pool.map(tokenize_chunk, chunks)
                    for t in results:
                        buf.extend(t)
                    chunks = []
                    while len(buf) >= SHARD_SIZE:
                        np.array(buf[:SHARD_SIZE], dtype=np.uint16).tofile(
                            os.path.join(comp_dir, f"train_{si:04d}.bin"))
                        buf = buf[SHARD_SIZE:]
                        done += SHARD_SIZE
                        si += 1
                    if si % 20 == 0:
                        print(f"[{name}] {done:,}/{target:,} ({100*done/target:.1f}%)")
                    if done >= target:
                        break
        if cur:
            chunks.append((cur, TOKENIZER))
        if chunks:
            with mp.Pool(min(nworkers, len(chunks))) as pool:
                results = pool.map(tokenize_chunk, chunks)
            for t in results:
                buf.extend(t)
    if buf and len(buf) > 1000:
        np.array(buf[:SHARD_SIZE] if len(buf)>SHARD_SIZE else buf, dtype=np.uint16).tofile(
            os.path.join(comp_dir, f"train_{si:04d}.bin"))
        done += min(len(buf), SHARD_SIZE)
        si += 1
    print(f"[{name}] DONE: {done:,} tokens, {si} shards")

def do_hf(name, target):
    import sentencepiece as spm
    from datasets import load_dataset
    comp_dir = os.path.join(OUTDIR, name)
    os.makedirs(comp_dir, exist_ok=True)
    existing = sorted([f for f in os.listdir(comp_dir) if f.endswith(".bin")])
    done = sum(os.path.getsize(os.path.join(comp_dir, f)) // 2 for f in existing)
    if done >= target * 0.95:
        print(f"[{name}] SKIP: {done:,} tokens already")
        return
    si = len(existing)
    CFG = {
        "fineweb": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
        "stack": ("bigcode/the-stack-v2-dedup", None, "content"),
        "megamath": ("MHHMM/MegaMath", None, "text"),
    }
    hf_id, subset, field = CFG[name]
    print(f"[{name}] HF stream {hf_id}, resume={done:,}, target={target:,}")
    kw = {"path": hf_id, "split": "train", "streaming": True}
    if subset: kw["name"] = subset
    ds = load_dataset(**kw)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
    buf = []
    batch = []
    if done > 0:
        skip = done // 200
        print(f"[{name}] skipping ~{skip:,} docs...")
        for i, _ in enumerate(ds):
            if i >= skip: break
    for ex in ds:
        if done >= target: break
        txt = ex.get(field, "")
        if not txt or len(txt) < 50: continue
        batch.append(txt)
        if len(batch) >= 5000:
            for toks in sp.encode(batch):
                buf.extend(toks)
            batch = []
            while len(buf) >= SHARD_SIZE:
                np.array(buf[:SHARD_SIZE], dtype=np.uint16).tofile(
                    os.path.join(comp_dir, f"train_{si:04d}.bin"))
                buf = buf[SHARD_SIZE:]
                done += SHARD_SIZE
                si += 1
            if si % 10 == 0 and si > 0:
                print(f"[{name}] {done:,}/{target:,} ({100*done/target:.1f}%)")
    if batch:
        for toks in sp.encode(batch):
            buf.extend(toks)
    if buf and len(buf) > 1000:
        np.array(buf, dtype=np.uint16).tofile(
            os.path.join(comp_dir, f"train_{si:04d}.bin"))
        done += len(buf)
        si += 1
    print(f"[{name}] DONE: {done:,} tokens, {si} shards")

if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    ncpu = mp.cpu_count()
    print(f"Fast tokenize: {ncpu} CPUs, output={OUTDIR}")
    workers = {"fw2hq_ru": 30, "fw2hq_fr": 25, "fw2hq_de": 25}
    procs = []
    for name, path in LOCAL.items():
        p = mp.Process(target=do_local, args=(name, path, TARGETS[name], workers[name]))
        p.start()
        procs.append((name, p))
        print(f"Started {name} ({workers[name]} workers)")
    for name in ["fineweb", "stack", "megamath"]:
        p = mp.Process(target=do_hf, args=(name, TARGETS[name]))
        p.start()
        procs.append((name, p))
        print(f"Started {name} (HF stream)")
    for name, p in procs:
        p.join()
        print(f"[{name}] finished in {(time.time()-t0)/60:.1f}m")
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} minutes")
