import argparse
import json
import logging
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
os.umask(0)

IGNORE = -100


# ======================================================================================
# data
# ======================================================================================
class SFTDataset(Dataset):
    """Pre-tokenised chat-formatted rows with the prompt masked out of the loss."""

    def __init__(self, path, tokenizer, max_seq_len, cache_dir=None, accelerator=None, tag=""):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sources = sorted({r.get("source", "unknown") for r in self.rows})
        self.source_to_id = {s: i for i, s in enumerate(self.sources)}

        cache = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = f"{tag}_{os.path.basename(path)}_{max_seq_len}_{len(self.rows)}.pt"
            cache = os.path.join(cache_dir, key)

        if cache and os.path.exists(cache):
            self.examples = torch.load(cache, weights_only=False)
        else:
            self.examples = [self._encode(r) for r in
                             tqdm(self.rows, desc=f"tokenising {tag}", disable=not _is_main(accelerator))]
            if cache and _is_main(accelerator):
                torch.save(self.examples, cache)
        self.lengths = [len(e["input_ids"]) for e in self.examples]

    def _encode(self, row):
        msgs = row["messages"]
        assert msgs[-1]["role"] == "assistant", "last message must be the target"
        full = self.tokenizer.apply_chat_template(msgs, tokenize=False)
        prompt = self.tokenizer.apply_chat_template(msgs[:-1], tokenize=False,
                                                    add_generation_prompt=True)
        full_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # apply_chat_template is not guaranteed to be a strict prefix; fall back to a
        # token-level common prefix rather than silently mislabelling the mask.
        n_prompt = len(prompt_ids)
        if full_ids[:n_prompt] != prompt_ids:
            n_prompt = 0
            for a, b in zip(full_ids, prompt_ids):
                if a != b:
                    break
                n_prompt += 1

        # right-truncate: keep the question, drop the tail of an over-long answer
        full_ids = full_ids[: self.max_seq_len]
        n_prompt = min(n_prompt, len(full_ids))
        labels = [IGNORE] * n_prompt + full_ids[n_prompt:]
        return {"input_ids": full_ids, "labels": labels,
                "source_id": self.source_to_id[row.get("source", "unknown")]}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def collate(self, batch):
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        n = max(len(b["input_ids"]) for b in batch)
        return {
            "input_ids": torch.LongTensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "labels": torch.LongTensor([b["labels"] + [IGNORE] * (n - len(b["labels"])) for b in batch]),
            "attention_mask": torch.LongTensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
            "source_id": torch.LongTensor([b["source_id"] for b in batch]),
        }


class LengthGroupedSampler(Sampler):
    """Shuffle, then sort by length inside mega-batches: keeps randomness, cuts padding."""

    def __init__(self, lengths, batch_size, mega=50, seed=0):
        self.lengths, self.batch_size, self.mega, self.seed = lengths, batch_size, mega, seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.randperm(len(self.lengths), generator=g).tolist()
        chunk = self.batch_size * self.mega
        out = []
        for i in range(0, len(idx), chunk):
            block = idx[i:i + chunk]
            block.sort(key=lambda j: self.lengths[j], reverse=True)
            out.extend(block)
        return iter(out)


def _is_main(acc):
    return acc is None or acc.is_main_process


# ======================================================================================
# eval
# ======================================================================================
@torch.no_grad()
def evaluate(model, dataloader, accelerator, n_sources):
    """Mean token-level CE on the held-out split, globally and per source."""
    model.eval()
    dev = accelerator.device
    tot_loss = torch.zeros(1, device=dev)
    tot_tok = torch.zeros(1, device=dev)
    tot_correct = torch.zeros(1, device=dev)
    src_loss = torch.zeros(n_sources, device=dev)
    src_tok = torch.zeros(n_sources, device=dev)

    for batch in dataloader:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    use_cache=False)
        logits = out.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  labels.reshape(-1), ignore_index=IGNORE,
                                  reduction="none").view(labels.shape)
        mask = (labels != IGNORE).float()
        tot_loss += (per_tok * mask).sum()
        tot_tok += mask.sum()
        tot_correct += ((logits.argmax(-1) == labels).float() * mask).sum()
        for i, s in enumerate(batch["source_id"].tolist()):
            src_loss[s] += (per_tok[i] * mask[i]).sum()
            src_tok[s] += mask[i].sum()

    for t in (tot_loss, tot_tok, tot_correct, src_loss, src_tok):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    model.train()
    n = max(tot_tok.item(), 1.0)
    per_source = {i: (src_loss[i] / src_tok[i]).item() for i in range(n_sources) if src_tok[i] > 0}
    return tot_loss.item() / n, tot_correct.item() / n, per_source


# ======================================================================================
# train
# ======================================================================================
def train(args):
    accelerator = Accelerator(mixed_precision="bf16",
                              gradient_accumulation_steps=args.gradient_accumulation_steps)
    world = accelerator.num_processes
    eff_bsz = args.train_bsz_per_gpu * world * args.gradient_accumulation_steps

    ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
    ds_cfg["train_micro_batch_size_per_gpu"] = args.train_bsz_per_gpu
    ds_cfg["train_batch_size"] = eff_bsz

    accelerator.print(json.dumps(vars(args), indent=2))
    accelerator.print(f"world={world} micro_bsz={args.train_bsz_per_gpu} "
                      f"accum={args.gradient_accumulation_steps} -> effective batch {eff_bsz}")
    if eff_bsz != 128:
        accelerator.print(f"WARNING: effective batch {eff_bsz} != 128 used by the paper")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    with accelerator.main_process_first():
        train_ds = SFTDataset(args.data_path, tokenizer, args.max_seq_len,
                              args.cache_dir, accelerator, "train")
        val_ds = (SFTDataset(args.val_path, tokenizer, args.max_seq_len,
                             args.cache_dir, accelerator, "val") if args.val_path else None)

    tl = train_ds.lengths
    accelerator.print(f"train rows={len(train_ds)} tokens={sum(tl):,} "
                      f"mean={sum(tl)/len(tl):.0f} p95={sorted(tl)[int(.95*len(tl))]} max={max(tl)}")

    sampler = LengthGroupedSampler(tl, args.train_bsz_per_gpu * world, seed=args.seed) \
        if args.length_grouping else None
    train_dl = DataLoader(train_ds, batch_size=args.train_bsz_per_gpu, sampler=sampler,
                          shuffle=sampler is None, drop_last=True, collate_fn=train_ds.collate,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = (DataLoader(val_ds, batch_size=args.eval_bsz_per_gpu, shuffle=False,
                         drop_last=False, collate_fn=val_ds.collate,
                         num_workers=args.num_workers) if val_ds else None)

    no_decay = ["bias", "LayerNorm.weight", "layernorm", "norm"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(d in n.lower() for d in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(d in n.lower() for d in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    # prepare() must not be handed a None: the DeepSpeed path inspects every argument
    if val_dl is None:
        model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    else:
        model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    # The schedule has to be built AFTER prepare(): before it, len(train_dl) counts the whole
    # dataset, and only prepare() shards it across processes. Sizing the cosine schedule on
    # the unsharded length overstates the step count by world_size, so the learning rate
    # never finishes decaying - a silent, unlogged way to end up with a worse model.
    steps_per_epoch = max(1, len(train_dl) // args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.n_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.warmup_rates * total_steps), total_steps)
    accelerator.print(f"optimizer steps: {steps_per_epoch}/epoch x {args.n_epochs} = {total_steps} "
                      f"(micro-batches per process per epoch: {len(train_dl)})")

    if accelerator.is_main_process and args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.experiment_name,
                   id=args.wandb_run_id or None, resume="allow" if args.wandb_run_id else None,
                   dir=args.log_dir, mode=args.wandb_mode,
                   config={**vars(args), "effective_batch_size": eff_bsz,
                           "world_size": world, "total_optimizer_steps": total_steps,
                           "train_rows": len(train_ds), "train_tokens": sum(tl)})

    def save(tag, meta):
        """Write a full-weight checkpoint from rank 0 only.

        `get_state_dict` under ZeRO-3 is a collective: every rank must enter it, but only
        rank 0 comes out with the consolidated tensors (the others get None). Only rank 0
        may then write. Calling `save_pretrained` on every rank corrupts the checkpoint:
        with safe_serialization=True transformers calls `safe_save_file` directly instead
        of going through `save_function`, so accelerate's is-main-process guard never
        applies, and the non-zero ranks fall back to `model.state_dict()` - the partitioned
        (empty) tensors plus the ~266k non-partitioned norm weights. That lands next to the
        real shards as a 0.6MB `model.safetensors`, which `from_pretrained` then prefers
        over the shard index, silently loading an empty model.
        """
        out = os.path.join(args.output_dir, tag)
        accelerator.wait_for_everyone()
        state = accelerator.get_state_dict(model)          # collective - all ranks
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(out, is_main_process=True,
                                      save_function=accelerator.save,
                                      state_dict=state, safe_serialization=True)
            tokenizer.save_pretrained(out)
            json.dump(meta, open(os.path.join(out, "training_meta.json"), "w"), indent=2)
            stray = os.path.join(out, "model.safetensors")
            if os.path.exists(stray) and os.path.exists(
                    os.path.join(out, "model.safetensors.index.json")):
                os.remove(stray)
                accelerator.print(f"removed stray {stray}")
        accelerator.wait_for_everyone()
        accelerator.print(f"saved {out} :: {meta}")

    # local mirror of every logged metric: survives a wandb outage and feeds make_figures.py
    metrics_path = os.path.join(args.log_dir, "metrics.jsonl")
    metrics_fh = open(metrics_path, "a", encoding="utf-8") if accelerator.is_main_process else None

    def log(payload, step):
        if not accelerator.is_main_process:
            return
        metrics_fh.write(json.dumps({"step": step, "wall": time.time(), **payload}) + "\n")
        metrics_fh.flush()
        if args.wandb:
            import wandb
            wandb.log(payload, step=step)

    def run_eval(step, epoch):
        """Validation pass -> same step axis as train/loss, so W&B overlays the curves."""
        vl, vacc, per_src = evaluate(model, val_dl, accelerator, len(train_ds.sources))
        named = {train_ds.sources[i] if i < len(train_ds.sources) else str(i): v
                 for i, v in per_src.items()}
        accelerator.print(f"[step {step}] val_loss={vl:.4f} val_token_acc={vacc:.4f} {named}")
        log({"val/loss": vl, "val/token_acc": vacc, "val/epoch": epoch,
             **{f"val/loss_{k}": v for k, v in named.items()}}, step)
        return vl, vacc

    best = {"val_loss": math.inf, "tag": None}

    def maybe_save_best(vl, vacc, step, epoch):
        """Persist a new best only on a real improvement.

        Gathering and writing 16GB of ZeRO-3 sharded weights takes on the order of a minute.
        Doing it on every microscopic dip would spend a large fraction of the run on I/O, so
        an improvement has to clear --save_best_min_delta to be worth a checkpoint. `best`
        still tracks the true minimum, and the epoch-end save is unconditional.
        """
        nonlocal best
        if vl >= best["val_loss"] - args.save_best_min_delta:
            return False
        best = {"val_loss": vl, "val_token_acc": vacc, "tag": "best",
                "opt_step": step, "epoch": epoch}
        save("best", best)
        log({"val/best_loss": vl}, step)
        return True

    global_step, opt_step, t0, tok_seen = 0, 0, time.time(), 0
    run_loss, run_tok, run_correct, run_ctok = 0.0, 0, 0.0, 0

    model.train()
    if val_dl is not None and args.eval_at_start:
        # step 0 = the backbone before any update; every later point is read against it
        v0, _ = run_eval(0, 0.0)
        accelerator.print(f"baseline val_loss before training: {v0:.4f}")
        t0 = time.time()
    for epoch in range(args.n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        bar = tqdm(train_dl, disable=not accelerator.is_main_process, desc=f"epoch {epoch}")
        for batch in bar:
            with accelerator.accumulate(model):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            labels=batch["labels"], use_cache=False)
                loss = out.loss
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            with torch.no_grad():
                mask = batch["labels"][:, 1:] != IGNORE
                ntok = mask.sum()
                run_loss += loss.detach().float().item() * ntok.item()
                run_tok += ntok.item()
                tok_seen += batch["input_ids"].numel()
                if global_step % args.metric_every == 0:
                    # sampled, not every step: the argmax over a [B,T,128k] logit tensor
                    # is not worth paying for on every micro-batch
                    pred = out.logits[:, :-1, :].argmax(-1)
                    run_correct += ((pred == batch["labels"][:, 1:]) & mask).sum().item()
                    run_ctok += ntok.item()
            global_step += 1

            if accelerator.sync_gradients:
                opt_step += 1
                if opt_step % args.log_every == 0 and accelerator.is_main_process:
                    stats = {
                        "train/loss": run_loss / max(run_tok, 1),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (opt_step % steps_per_epoch) / max(steps_per_epoch, 1),
                        "train/tokens_per_s": tok_seen * world / max(time.time() - t0, 1e-6),
                        "perf/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                    }
                    if run_ctok:
                        stats["train/token_acc"] = run_correct / run_ctok
                    if grad_norm is not None:
                        stats["train/grad_norm"] = float(grad_norm)
                    bar.set_postfix(loss=round(stats["train/loss"], 4),
                                    lr=f"{stats['train/lr']:.2e}",
                                    tps=int(stats["train/tokens_per_s"]))
                    log(stats, opt_step)
                    run_loss, run_tok, run_correct, run_ctok = 0.0, 0, 0.0, 0
                    tok_seen, t0 = 0, time.time()

                if val_dl is not None and args.eval_every > 0 and opt_step % args.eval_every == 0:
                    frac = epoch + (opt_step % max(steps_per_epoch, 1)) / max(steps_per_epoch, 1)
                    vl, vacc = run_eval(opt_step, frac)
                    maybe_save_best(vl, vacc, opt_step, epoch)
                    t0 = time.time()

        # end of epoch
        meta = {"epoch": epoch, "opt_step": opt_step}
        if val_dl is not None:
            vl, vacc = run_eval(opt_step, float(epoch + 1))
            meta.update(val_loss=vl, val_token_acc=vacc)
            if vl < best["val_loss"]:      # epoch boundary: no min-delta, always worth it
                best = {**meta, "tag": "best", "val_loss": vl, "val_token_acc": vacc}
                save("best", best)
                log({"val/best_loss": vl}, opt_step)
        if args.save_each_epoch:
            save(f"epoch-{epoch + 1}", meta)
        t0 = time.time()

    save("last", {"epoch": args.n_epochs - 1, "opt_step": opt_step, "final": True})
    if accelerator.is_main_process:
        json.dump({"best": best, "total_optimizer_steps": opt_step,
                   "effective_batch_size": eff_bsz, "args": vars(args)},
                  open(os.path.join(args.output_dir, "run_summary.json"), "w"), indent=2)
        accelerator.print(f"BEST CHECKPOINT: {best}")
        metrics_fh.close()
        if args.wandb:
            import wandb
            wandb.summary.update({"best_val_loss": best["val_loss"],
                                  "best_opt_step": best.get("opt_step"),
                                  "best_epoch": best.get("epoch")})
            wandb.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--val_path", default=None)
    p.add_argument("--experiment_name", default="brainmed-sft")
    p.add_argument("--output_dir", default="./ckpts")
    p.add_argument("--log_dir", default="./train_logs")
    p.add_argument("--cache_dir", default="./.tokcache")

    # paper recipe
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--n_epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_rates", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--train_bsz_per_gpu", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=4096)

    # throughput / infra
    # 2, not 4: the validation pass materialises a float32 [B, T, 128k] logit tensor to get
    # per-token losses, which is the single largest allocation in the run.
    p.add_argument("--eval_bsz_per_gpu", type=int, default=2)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--flash_attn", action="store_true", default=True)
    p.add_argument("--no_flash_attn", dest="flash_attn", action="store_false")
    p.add_argument("--length_grouping", action="store_true", default=True)
    p.add_argument("--no_length_grouping", dest="length_grouping", action="store_false")
    p.add_argument("--num_workers", type=int, default=4)

    # observability
    p.add_argument("--eval_every", type=int, default=50, help="optimizer steps; 0 disables")
    p.add_argument("--eval_at_start", action="store_true", default=True,
                   help="validation pass at step 0, before any update")
    p.add_argument("--no_eval_at_start", dest="eval_at_start", action="store_false")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--metric_every", type=int, default=20)
    p.add_argument("--save_best_min_delta", type=float, default=0.005,
                   help="val-loss improvement required to rewrite the best checkpoint mid-epoch")
    p.add_argument("--save_each_epoch", action="store_true", default=True)
    p.add_argument("--no_save_each_epoch", dest="save_each_epoch", action="store_false")
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", default="brainmed-sft")
    p.add_argument("--wandb_mode", default="online")
    p.add_argument("--wandb_run_id", default=None)
    p.add_argument("--seed", type=int, default=2002)
    args = p.parse_args()

    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
