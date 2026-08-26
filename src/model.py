import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def repeat_kv(self, x, n_rep):
        batch, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, None, :, :]
            .expand(batch, seq_len, n_rep, n_kv_heads, head_dim)
            .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
        )

    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        scale = 1.0 / math.sqrt(self.head_dim)

        if mask is not None:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.o_proj(out))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout=0.1, norm_eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class KortexConfig:
    def __init__(self, **kwargs):
        defaults = {
            "vocab_size": 50257,
            "n_layers": 24,
            "n_heads": 16,
            "n_kv_heads": 4,
            "d_model": 1024,
            "d_ff": 2816,
            "max_seq_len": 1024,
            "dropout": 0.1,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
        for k, v in defaults.items():
            setattr(self, k, kwargs.get(k, v))

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


class KortexModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(
            config.d_model // config.n_heads,
            config.max_seq_len,
            config.rope_theta,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.n_kv_heads,
                config.d_ff,
                config.dropout,
                config.norm_eps,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} > max {self.config.max_seq_len}"

        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rotary(x, T)

        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        x = self.norm(x)

        logits = self.lm_head(x) if self.lm_head is not None else F.linear(x, self.tok_emb.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class KortexForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = KortexModel(config)

    def forward(self, idx, targets=None):
        return self.model(idx, targets)

    def generate(self, idx, max_new_tokens=256, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.max_seq_len else idx[:, -self.model.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def num_parameters(self):
        return self.model.num_parameters()
