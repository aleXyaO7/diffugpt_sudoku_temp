import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from pathlib import Path
import random, time
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, get_scheduler, set_seed as hf_set_seed,
)
from collections import deque

from packages.custom_tokenizer import CustomTokenizer
from packages.packed_dataset import PackedDataset

model_name = 'gpt2-model-bs1024-lr1e-3-ep100-20250910-035030'
train_dataset = 'data/train'
batch_size = 128
seq_length = 164
max_steps = 200_000_000
seed = 105
t_min = 0.01
t_max = 0.99
learning_rate = 0.45e-4

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

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = CustomTokenizer.from_pretrained(model_name)
train_loader = make_loader(train_dataset, batch_size, seq_length + 1, True, seed, drop_last=True)

mask_token = "[MASK]"
mask_id = tokenizer.convert_tokens_to_ids(mask_token)
tokenizer.pad_token_id = mask_id

opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=False)
sched = get_scheduler("cosine_with_restarts", optimizer=opt, num_warmup_steps=1000, num_training_steps=200000000)
crit = nn.CrossEntropyLoss(reduction="none")

model.train()
gstep = 0
pbar = tqdm(range(max_steps), disable=not acc.is_main_process)

t_lo, t_hi = t_min, t_max
corr_window = 1000
loss_window, mask_window = deque(maxlen=corr_window), deque(maxlen=corr_window)
t_last = time.time()