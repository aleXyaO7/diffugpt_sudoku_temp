""" DiffuLLaMA-style adaptation training for Llama-3 with bidirectional attention.
# move # attn_mask # opt # args.eval_every # iterator # acc.accumulate

**2025-08-05 update** – adds the two training tweaks you requested:
1. Cosine-with-restarts LR schedule (1 k-step warm-up → cosine).
2. Noise-curriculum: during the first --curriculum_steps we linearly grow the upper bound of t from t_min to t_max.
"""
from __future__ import annotations

import argparse, os, random, itertools, json, time
from pathlib import Path
from collections import deque

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, get_scheduler, set_seed as hf_set_seed,
)
import transformers.models.llama.modeling_llama as modeling_llama  # noqa: F401

from packages.packed_dataset import PackedDataset
from tqdm import tqdm
from packages.custom_tokenizer import CustomTokenizer


# --------------------------- masking helpers ---------------------------
def _encode_variants(tok, text):
    return [tok.encode(v, add_special_tokens=False) for v in (text, "\n"+text, " "+text, "\n\n"+text)]

def _rfind_subseq(hay, needle):
    if not needle: return -1
    n, m = len(hay), len(needle)
    for i in range(n - m, -1, -1):
        if hay[i:i+m] == needle: return i
    return -1

def build_maskable_after_assistant(x, tokenizer, assistant_text="Assistant:"):
    pats_ass = _encode_variants(tokenizer, assistant_text)
    pats_usr = _encode_variants(tokenizer, "User:")

    x_cpu = x.to("cpu")
    bs, T = x.shape
    maskable = torch.zeros_like(x, dtype=torch.bool)

    for b in range(bs):
        seq = x_cpu[b].tolist()
        last_end = -1
        for p in pats_ass:
            if not p: continue
            j = _rfind_subseq(seq, p)
            if j != -1:
                last_end = max(last_end, j + len(p) - 1)
        if last_end != -1 and last_end + 1 < T:
            maskable[b, last_end+1:] = True
        else:
            has_user = any(_rfind_subseq(seq, p) != -1 for p in pats_usr if p)
            if not has_user:
                maskable[b, 1:] = True
    maskable[:, 0] = False
    return maskable.to(x.device)


# ------------------ distributed-safe dataset guard --------------------
class SafePackedDataset(IterableDataset):
    """
    Wraps PackedDataset and *immediately restarts* the iterator on ValueError.
    This prevents any rank from stalling on a corrupt/bad-offset sample.
    """
    def __init__(self, files, n_chunks, block_size, shuffle, seed):
        self.files = files
        self.n_chunks = n_chunks
        self.block_size = block_size
        self.shuffle = shuffle
        self.seed = seed

    def _make_iter(self):
        base = PackedDataset(
            self.files, n_chunks=self.n_chunks, block_size=self.block_size,
            shuffle=self.shuffle, seed=self.seed,
        )
        return iter(base)

    def __iter__(self):
        it = self._make_iter()
        while True:
            try:
                yield next(it)
            except ValueError:
                # Bad offset encountered: rebuild iterator and continue immediately.
                it = self._make_iter()
                continue
            except StopIteration:
                return


# ------------------------------ helpers -------------------------------
def make_loader(pattern: str, bs: int, block: int, shuffle: bool, seed: int, *, drop_last: bool):
    files = sorted(str(p) for p in Path(pattern).glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"No .bin files found in {pattern}")
    if shuffle:
        random.shuffle(files)
    ds = SafePackedDataset(files, n_chunks=8, block_size=block, shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_size=bs, pin_memory=True, num_workers=0, drop_last=drop_last)


def transition(x_0: torch.Tensor, sigma: torch.Tensor, maskable_mask: torch.Tensor, mask_token_id: int):
    # Avoid hidden full-size float copy
    rand = torch.rand_like(x_0, dtype=torch.float32)
    move = (rand < sigma) & maskable_mask
    return torch.where(move, mask_token_id, x_0)


def run_eval(acc, model, eval_loader, tokenizer, crit, mask_id, t_lo, t_hi, args, gstep):
    model.eval()
    bins = torch.linspace(t_lo, t_hi, steps=max(args.eval_t_bins, 2) + 1, device=acc.device)
    totals = torch.zeros(args.eval_t_bins, device=acc.device)
    counts = torch.zeros(args.eval_t_bins, device=acc.device)
    with torch.no_grad():
        it = itertools.islice(eval_loader, 0, max(args.eval_batches, 1))
        for batch in it:
            x = batch.to(acc.device)
            attn_mask = x.ne(tokenizer.pad_token_id)  # bool
            for b in range(args.eval_t_bins):
                lo, hi = bins[b].item(), bins[b+1].item()
                t = torch.empty(x.size(0), device=acc.device).uniform_(lo, hi)
                maskable = build_maskable_after_assistant(x, tokenizer, "Assistant:")
                noisy = transition(x, t[:, None], maskable, mask_id)
                loss_mask = noisy.eq(mask_id)
                logits = model(input_ids=noisy, attention_mask=attn_mask).logits[:, :-1]
                tgt = x[:, 1:]
                loss_tok = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).view_as(tgt)
                loss_tok = loss_tok * loss_mask[:, 1:]
                ce = (loss_tok.sum() / (loss_mask[:, 1:].sum() + 1e-8))
                totals[b] += ce
                counts[b] += 1
    ce_bins = totals / counts.clamp_min(1.0)
    acc.log({f"eval/ce_tbin_{i}": float(ce_bins[i]) for i in range(args.eval_t_bins)}, step=gstep)
    model.train()


def patch_model_for_bidirectional_training(model):
    for layer in model.transformer.h:
        if hasattr(layer.attn, "is_causal"):
            layer.attn.is_causal = False
    model.config.use_cache = False
 
 
 
 
    print("✅ Causal mask cleared – RoPE intact")


# ------------------------------- CLI ----------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    #p.add_argument("--load_checkpoint_or_model", default="diffu-save-final/checkpoint-base-08-26")
    p.add_argument("--load_checkpoint_or_model", default="gpt2-model-bs1024-lr1e-3-ep100-20250910-035030")
    p.add_argument("--train_dataset", default="data/train")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--wandb_project", default="diffu-gpt3")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.45e-4)
    p.add_argument("--seq_length", type=int, default=2048)
    p.add_argument("--max_steps", type=int, default=200_000_000)
    p.add_argument("--seed", type=int, default=105)
    p.add_argument("--t_min", type=float, default=0.01)
    p.add_argument("--t_max", type=float, default=0.99)
    p.add_argument("--curriculum_steps", type=int, default=1,
                   help="Steps over which to ramp t_max from t_min to t_max.")
    p.add_argument("--eval_dataset", default="data/test")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_batches", type=int, default=2)
    p.add_argument("--eval_t_bins", type=int, default=10)
    p.add_argument("--inverse_t_weight", action="store_true")
    p.set_defaults(inverse_t_weight=True)
    p.add_argument("--weight_warmup_steps", type=int, default=50_000_000)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--force_world_size_mismatch", action="store_true", default=True)
    return p.parse_args()


# ------------------------------ training ------------------------------
def main():
    args = parse_args()
    hf_set_seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)

    acc = Accelerator(gradient_accumulation_steps=args.grad_accum_steps,
                      mixed_precision="bf16", log_with="wandb")

    if not args.force_world_size_mismatch and acc.num_processes != 8:
        raise RuntimeError("Expected 8 GPUs; override with --force_world_size_mismatch if intentional.")
    acc.init_trackers(args.wandb_project, config=vars(args))

    if acc.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print("▶ Launching diffusion training run…")

    # model & tok
    model = AutoModelForCausalLM.from_pretrained(args.load_checkpoint_or_model, torch_dtype=torch.bfloat16)
    patch_model_for_bidirectional_training(model)
    tokenizer = CustomTokenizer.from_pretrained(args.load_checkpoint_or_model)

    # special [MASK]
    mask_token = "[MASK]"
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    need_add = (mask_id is None) or (mask_id == tokenizer.unk_token_id)
    if need_add:
        tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            std = emb.std().item()
            emb[mask_id].normal_(0.0, std)
            out_emb = model.get_output_embeddings()
            if out_emb is not None and out_emb.weight.data_ptr() != emb.data_ptr():
                out_emb.weight[mask_id].copy_(emb[mask_id])
        if acc.is_main_process:
            print(f"✅ Added {mask_token} id={mask_id}")
    else:
        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = mask_id

    # data
    train_loader = make_loader(args.train_dataset, args.batch_size, args.seq_length + 1,
                               True, args.seed, drop_last=True)
    eval_loader = None
    if args.eval_dataset:
        eval_loader = make_loader(args.eval_dataset, args.batch_size, args.seq_length + 1,
                                  False, args.seed, drop_last=True)

    # opt & sched
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=False)
    sched = get_scheduler("cosine_with_restarts", optimizer=opt,
                          num_warmup_steps=1000, num_training_steps=200000000)
    crit = nn.CrossEntropyLoss(reduction="none")

    if eval_loader is not None:
        model, opt, train_loader, eval_loader, sched = acc.prepare(model, opt, train_loader, eval_loader, sched)
    else:
        model, opt, train_loader, sched = acc.prepare(model, opt, train_loader, sched)

    # loop
    model.train()
    gstep = 0
    pbar = tqdm(range(args.max_steps), disable=not acc.is_main_process)

    t_lo, t_hi = min(args.t_min, args.t_max), max(args.t_min, args.t_max)
    corr_window = 1000
    loss_window, mask_window = deque(maxlen=corr_window), deque(maxlen=corr_window)
    t_last = time.time()

    while gstep < args.max_steps:
        for x in train_loader:
            with acc.accumulate(model):
                bs = x.size(0)
                # curriculum on t
                progress = min(1.0, gstep / max(1, args.curriculum_steps))
                curr_t_hi = t_lo + (t_hi - t_lo) * progress
                t = torch.empty(bs, device=x.device).uniform_(t_lo, curr_t_hi)

                maskable = build_maskable_after_assistant(x, tokenizer, "Assistant:")

                noisy = transition(x, t[:, None], maskable, mask_id)
                loss_mask = noisy.eq(mask_id)

                attn_mask = x.ne(tokenizer.pad_token_id)  # bool
                logits = model(input_ids=noisy, attention_mask=attn_mask).logits[:, :-1]
                tgt = x[:, 1:]

                loss_tok = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).view_as(tgt)
                loss_tok = loss_tok * loss_mask[:, 1:]

                scale = 1.0
                if args.inverse_t_weight and gstep >= args.weight_warmup_steps:
                    scale = (1.0 / t)[:, None]
                loss = (loss_tok * scale).sum() / (loss_mask[:, 1:].sum() + 1e-8)

                acc.backward(loss)

                if acc.sync_gradients:
                    grad_norm = acc.clip_grad_norm_(model.parameters(), 1.5)
                    ce_raw = (loss_tok.sum() / (loss_mask[:, 1:].sum() + 1e-8)).item()
                    mask_ratio_val = loss_mask.float().mean().item()

                    loss_window.append(loss.item()); mask_window.append(mask_ratio_val)
                    if len(loss_window) >= 30:
                        lw = torch.tensor(list(loss_window)); mw = torch.tensor(list(mask_window))
                        corr = float(((lw - lw.mean()) * (mw - mw.mean())).mean() / (lw.std()*mw.std() + 1e-8))
                    else:
                        corr = float('nan')

                    if gstep % 100 == 0:
                        with torch.no_grad():
                            emb = acc.unwrap_model(model).get_input_embeddings().weight
                            std_ratio = float(emb[mask_id].std().item() / (emb.std().item() + 1e-12))
                        acc.log({"train/mask_std_ratio": std_ratio}, step=gstep)

                    acc.log({
                        "train/loss": loss.item(),
                        "train/ce_raw": ce_raw,
                        "train/grad_norm": float(grad_norm),
                        "train/mask_ratio": mask_ratio_val,
                        "train/t_mean": float(t.mean().item()),
                        "train/loss_mask_corr": corr,
                        "lr": sched.get_last_lr()[0],
                    }, step=gstep)

                    if acc.is_main_process:
                        acc.print(
                            f"step={gstep} loss={loss.item():.4f} ce_raw={ce_raw:.4f} grad_norm={grad_norm:.2f} "
                            f"mask_ratio={mask_ratio_val:.3f} corr={corr:.2f} t_hi={curr_t_hi:.2f} dt={time.time()-t_last:.1f}s"
                        )

                    mem_alloc = torch.cuda.memory_allocated() / 1e9
                    mem_rsrv = torch.cuda.memory_reserved() / 1e9
                    acc.log({"mem/alloc_gb": float(mem_alloc), "mem/reserved_gb": float(mem_rsrv)}, step=gstep)
                    t_last = time.time()

                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

            if acc.sync_gradients:
                gstep += 1
                pbar.update(1)

                if args.eval_every > 0 and eval_loader is not None and (gstep % args.eval_every == 0):
                    try:
                        run_eval(acc, model, eval_loader, tokenizer, crit, mask_id, t_lo, t_hi, args, gstep)
                    except Exception as e:
                        if acc.is_main_process:
                            acc.print(f"[eval] skipped: {e}")
                    torch.cuda.empty_cache()
                    acc.wait_for_everyone()

                if gstep % args.save_every == 0 and acc.is_main_process:
                    ck = Path(args.output_dir) / f"checkpoint-{gstep}"
                    ck.mkdir(parents=True, exist_ok=True)
                    acc.unwrap_model(model).save_pretrained(ck)
                    tokenizer.save_pretrained(ck)
                    torch.save(opt.state_dict(), ck / "optimizer.pt")
                    torch.save(sched.state_dict(), ck / "scheduler.pt")
                    with open(ck / "trainer_state.json", "w") as f:
                        json.dump({"global_step": gstep}, f)
                    acc.print(f"💾 saved checkpoint-{gstep}")

            if gstep >= args.max_steps:
                break

    acc.end_training()
    if acc.is_main_process:
        print("🏁 Training complete.")


if __name__ == "__main__":
    main()

