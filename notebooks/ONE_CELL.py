!pip install datasets tokenizers transformers torch_xla -q

import os, sys, json, time, math, gc, tempfile, shutil, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.serialization as xser
    DEVICE = xm.xla_device()
    print(f"TPU: {DEVICE}")
except:
    DEVICE = torch.device("cpu")
    print(f"CPU fallback")

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF")
WRITE_TOKEN = user_secrets.get_secret("WRITE_TK")
print("Secrets loaded.")

CONFIG = {
    'vocab_size': 50257, 'n_layers': 40, 'n_heads': 32, 'n_kv_heads': 8,
    'd_model': 4096, 'd_ff': 11008, 'max_seq_len': 2048, 'dropout': 0.0,
    'norm_eps': 1e-6, 'rope_theta': 10000.0, 'batch_size': 1, 'grad_accum': 128,
    'lr': 3e-5, 'weight_decay': 0.1, 'warmup_steps': 300, 'max_steps': 5000,
    'min_lr_ratio': 0.1, 'max_grad_norm': 1.0, 'checkpoint_every': 500,
    'eval_every': 250, 'eval_steps': 20, 'log_every': 10, 'max_time_hours': 8.5,
    'save_top_k': 3, 'gradient_checkpointing': True,
}

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x.float() * x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()).type_as(x) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        emb = torch.cat((torch.outer(t, self.inv_freq),) * 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    def forward(self, x, seq_len):
        if seq_len > self.cos_cached.shape[0]: self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
    def repeat_kv(self, x, n_rep):
        B, T, N, D = x.shape
        if n_rep == 1: return x
        return x[:, :, None, :, :].expand(B, T, n_rep, N, D).reshape(B, T, N*n_rep, D)
    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = self.repeat_kv(k, self.n_rep), self.repeat_kv(v, self.n_rep)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None: attn = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_drop(self.o_proj(out))

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout=0.1, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps)
        self.attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(d_model, eps)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        return x + self.ffn(self.norm2(x))

class KortexModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        self.drop = nn.Dropout(cfg['dropout'])
        self.rotary = RotaryEmbedding(cfg['d_model']//cfg['n_heads'], cfg['max_seq_len'], cfg['rope_theta'])
        self.layers = nn.ModuleList([TransformerBlock(cfg['d_model'], cfg['n_heads'], cfg['n_kv_heads'], cfg['d_ff'], cfg['dropout'], cfg['norm_eps']) for _ in range(cfg['n_layers'])])
        self.norm = RMSNorm(cfg['d_model'], cfg['norm_eps'])
        self.lm_head = nn.Linear(cfg['d_model'], cfg['vocab_size'], bias=False)
        self.apply(self._init_weights)
        nparams = sum(p.numel() for p in self.parameters())
        print(f"KortexModel: {nparams/1e9:.2f}B params ({nparams/1e6:.0f}M)")
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding): torch.nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rotary(x, T)
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers: x = layer(x, cos, sin, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100) if targets is not None else None
        return logits, loss
    def generate(self, idx, max_new_tokens=256, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg['max_seq_len']:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx

class TokenStream:
    def __init__(self, max_tokens=50000000):
        self.max_tokens = max_tokens
        self.tokens = []
        self.pos = 0
    def add(self, new_tokens):
        self.tokens.extend(new_tokens)
        if len(self.tokens) > self.max_tokens:
            self.tokens = self.tokens[-self.max_tokens:]
    def sample(self, seq_len):
        if len(self.tokens) < seq_len + 1: return None
        i = torch.randint(0, len(self.tokens) - seq_len - 1, (1,)).item()
        chunk = self.tokens[i:i + seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
    def __len__(self): return len(self.tokens)

class CosineScheduler:
    def __init__(self, opt, warmup, max_steps, min_lr=0.1):
        self.opt, self.warmup, self.max_steps, self.min_lr, self.n = opt, warmup, max_steps, min_lr, 0
        self.base_lrs = [pg['lr'] for pg in opt.param_groups]
    def step(self):
        self.n += 1
        s = self.n / max(1, self.warmup) if self.n < self.warmup else self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * (self.n - self.warmup) / max(1, self.max_steps - self.warmup)))
        for pg, bl in zip(self.opt.param_groups, self.base_lrs): pg['lr'] = bl * s
    def get_lr(self): return [pg['lr'] for pg in self.opt.param_groups]

print("All classes defined.")

def push_github(token, msg="Update checkpoint"):
    try:
        subprocess.run(["git", "add", "-A"], capture_output=True, timeout=30)
        if not subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=30).stdout.strip(): return
        subprocess.run(["git", "commit", "-m", msg], capture_output=True, timeout=30)
        subprocess.run(["git", "push"], capture_output=True, text=True, timeout=60)
        print("  Pushed to GitHub.")
    except Exception as e: print(f"  GH err: {e}")

def push_hf(path, token, name="kortex"):
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)
        rid = f"{name}-model"
        try: create_repo(rid, token=token, exist_ok=True)
        except: pass
        api.upload_folder(folder_path=path, repo_id=rid, token=token)
        print(f"  Uploaded to HF: https://huggingface.co/{rid}")
    except Exception as e: print(f"  HF err: {e}")

print("Initializing Kortex...")
model = KortexModel(CONFIG).to(DEVICE)
if CONFIG.get('gradient_checkpointing'):
    print("Enabling gradient checkpointing...")
    for i, layer in enumerate(model.layers):
        orig_fwd = layer.forward
        def make_ckpt_fwd(bl):
            def ckpt_fwd(x, cos, sin, mask=None):
                def inner(x2, cos2, sin2, mask2):
                    return bl.norm2(x2 + bl.attn(bl.norm1(x2), cos2, sin2, mask2))
                x = x + torch.utils.checkpoint.checkpoint(
                    lambda x2: bl.attn(bl.norm1(x2), cos, sin, mask), x, use_reentrant=False
                )
                x = x + torch.utils.checkpoint.checkpoint(
                    lambda x2: bl.ffn(bl.norm2(x2)), x, use_reentrant=False
                )
                return x
            return ckpt_fwd
        layer.forward = make_ckpt_fwd(layer)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], betas=(0.9, 0.95))
scheduler = CosineScheduler(optimizer, CONFIG['warmup_steps'], CONFIG['max_steps'], CONFIG['min_lr_ratio'])
global_step = 0
best_val_loss = float('inf')
checkpoints = []
start_time = time.time()
MAX_SECONDS = CONFIG['max_time_hours'] * 3600
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/kortex-output', exist_ok=True)

def save_ckpt(step, val_loss=None, is_best=False):
    global best_val_loss, checkpoints
    ckpt = {'step': step, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'global_step': global_step, 'best_val_loss': best_val_loss}
    if is_best:
        p = '/kaggle/working/checkpoints/best.pt'
        xser.save(ckpt, p) if str(DEVICE).startswith('xla') else torch.save(ckpt, p)
        print(f"  [BEST] step={step} loss={val_loss:.4f}")
    p = f'/kaggle/working/checkpoints/step_{step}.pt'
    xser.save(ckpt, p) if str(DEVICE).startswith('xla') else torch.save(ckpt, p)
    checkpoints.append((step, p, val_loss or float('inf')))
    if len(checkpoints) > CONFIG['save_top_k'] + 1:
        checkpoints.sort(key=lambda x: x[2])
        while len(checkpoints) > CONFIG['save_top_k']:
            s, path, _ = checkpoints.pop(0)
            if os.path.exists(path): os.remove(path)
    push_github(WRITE_TOKEN, f"Checkpoint step {step}")

def load_ckpt():
    global global_step, best_val_loss
    p = '/kaggle/working/checkpoints/best.pt'
    if os.path.exists(p):
        ckpt = torch.load(p, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state']); optimizer.load_state_dict(ckpt['optim_state'])
        global_step = ckpt.get('global_step', 0); best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"  Resumed step {global_step}"); return True
    print("  Fresh start."); return False

print("Training BPE tokenizer...")
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(vocab_size=CONFIG['vocab_size'], min_frequency=2, special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"], show_progress=True)
ds = load_dataset('angie-chen55/python-github-code', split='train', streaming=True)
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    c = 0
    for item in ds:
        code = item.get('code', '') or item.get('text', '')
        if len(code) > 50: f.write(code[:8000] + '\n'); c += 1
        if c >= 200000: break
    tf = f.name
tok.train([tf], trainer)
TOKENIZER_PATH = '/kaggle/working/kortex-output/tokenizer.json'
tok.save(TOKENIZER_PATH)
print(f"Tokenizer: vocab={tok.get_vocab_size()}")
del ds; gc.collect()

def encode(text): return tok.encode(text).ids

print("Streaming dataset into token buffer (50M tokens max, ~200MB RAM)...")
ds = load_dataset('angie-chen55/python-github-code', split='train', streaming=True)
stream = TokenStream(max_tokens=50000000)
c = 0
for item in ds:
    code = item.get('code', '') or item.get('text', '')
    if len(code) > 50:
        stream.add(encode(code[:8000]))
        c += 1
    if c >= 1500000: break
    if c % 100000 == 0 and c > 0:
        print(f"  {c} samples streamed, {len(stream)} tokens ({len(stream)*4/1e6:.0f}MB RAM)")
        xm.mark_step() if str(DEVICE).startswith('xla') else None
print(f"  Done: {c} samples, {len(stream)} tokens ({len(stream)*4/1e6:.0f}MB RAM)")
del ds; gc.collect()

load_ckpt()
nparams = sum(p.numel() for p in model.parameters())
print(f"\n{'='*60}")
print(f"KORTEX v4 — MAXIMUM MODEL")
print(f"Model:   {nparams/1e9:.2f}B params | {CONFIG['n_layers']} layers | d={CONFIG['d_model']} | ff={CONFIG['d_ff']}")
print(f"Heads:   {CONFIG['n_heads']}Q / {CONFIG['n_kv_heads']}KV | head_dim={CONFIG['d_model']//CONFIG['n_heads']}")
print(f"Train:   {CONFIG['max_time_hours']}h budget | {CONFIG['max_steps']} max steps")
print(f"Batch:   {CONFIG['batch_size']} x {CONFIG['grad_accum']} = {CONFIG['batch_size']*CONFIG['grad_accum']} effective")
print(f"LR:      {CONFIG['lr']} | warmup={CONFIG['warmup_steps']} | cosine decay")
print(f"Buffer:  {len(stream)} tokens | seq_len={CONFIG['max_seq_len']}")
print(f"Ckpt:    grad_checkpointing={'ON' if CONFIG.get('gradient_checkpointing') else 'OFF'}")
print(f"{'='*60}")

grad_accum_count = 0
train_losses = []

while global_step < CONFIG['max_steps']:
    elapsed = time.time() - start_time
    if elapsed >= MAX_SECONDS: print(f"\nTIME LIMIT ({CONFIG['max_time_hours']}h). Stopping."); break
    if (MAX_SECONDS - elapsed) / 3600 < 0.5: print("\n<30min left. Final save."); break

    batch = stream.sample(CONFIG['max_seq_len'])
    if batch is None:
        print("Buffer empty, refilling...")
        try:
            ds = load_dataset('angie-chen55/python-github-code', split='train', streaming=True)
            c2 = 0
            for item in ds:
                code = item.get('code', '') or item.get('text', '')
                if len(code) > 50:
                    stream.add(encode(code[:8000]))
                    c2 += 1
                if c2 >= 500000: break
            del ds; gc.collect()
            print(f"  Refilled: +{c2} samples, now {len(stream)} tokens")
        except Exception as e: print(f"  Refill err: {e}"); time.sleep(10)
        continue

    x, y = batch[0].unsqueeze(0).to(DEVICE), batch[1].unsqueeze(0).to(DEVICE)
    with torch.amp.autocast(device_type='xla', dtype=torch.bfloat16, enabled=True):
        _, loss = model(x, targets=y)
    loss.backward()
    grad_accum_count += 1

    if grad_accum_count % CONFIG['grad_accum'] == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step(); optimizer.zero_grad(); scheduler.step()
        global_step += 1
        xm.mark_step() if str(DEVICE).startswith('xla') else None
        train_losses.append(loss.item())

        if global_step % CONFIG['log_every'] == 0:
            avg = sum(train_losses[-CONFIG['log_every']:]) / len(train_losses[-CONFIG['log_every']:])
            rem = (MAX_SECONDS - (time.time() - start_time)) / 3600
            print(f"Step {global_step:>6d} | loss={avg:.4f} | lr={scheduler.get_lr()[0]:.2e} | {rem:.1f}h left")

        if global_step % CONFIG['eval_every'] == 0:
            model.eval(); el, ec = 0, 0
            with torch.no_grad():
                for _ in range(CONFIG['eval_steps']):
                    eb = stream.sample(CONFIG['max_seq_len'])
                    if eb is None: break
                    ex, ey = eb[0].unsqueeze(0).to(DEVICE), eb[1].unsqueeze(0).to(DEVICE)
                    with torch.amp.autocast(device_type='xla', dtype=torch.bfloat16, enabled=True):
                        _, l = model(ex, targets=ey)
                    el += l.item(); ec += 1
            if ec > 0:
                avg_eval = el / ec
                is_best = avg_eval < best_val_loss
                if is_best: best_val_loss = avg_eval
                print(f"  EVAL step={global_step} loss={avg_eval:.4f} {'[BEST]' if is_best else ''}")
                save_ckpt(global_step, avg_eval, is_best)
            model.train()

        if global_step % CONFIG['checkpoint_every'] == 0 and global_step % CONFIG['eval_every'] != 0:
            save_ckpt(global_step)

print(f"\n{'='*60}\nDone. Steps: {global_step} | Best: {best_val_loss:.4f} | Time: {(time.time()-start_time)/3600:.2f}h\n{'='*60}")

print("Saving final model...")
fp = '/kaggle/working/kortex-output'
os.makedirs(fp, exist_ok=True)
fc = {'model_state': model.state_dict(), 'config': CONFIG, 'global_step': global_step, 'best_val_loss': best_val_loss}
xser.save(fc, f'{fp}/kortex_final.pt') if str(DEVICE).startswith('xla') else torch.save(fc, f'{fp}/kortex_final.pt')
shutil.copy(TOKENIZER_PATH, f'{fp}/tokenizer.json')
with open(f'{fp}/config.json', 'w') as f: json.dump(CONFIG, f, indent=2)
push_github(WRITE_TOKEN, "Final 7B model save")
push_hf(fp, HF_TOKEN)

print("\nTesting generation...")
model.eval()
for prompt in ["def fibonacci(n):\n", "class BinaryTree:\n    def __init__(self, val):\n", "import numpy as np\n\ndef matrix_multiply(a, b):\n", "def merge_sort(arr):\n    if len(arr) <= 1:\n"]:
    print(f"\nPrompt: {prompt.strip()}")
    print("-" * 40)
    x = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.no_grad(): out = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=50)
    print(tok.decode(out[0].cpu().tolist()))
