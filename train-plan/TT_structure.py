import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        if attn_mask is not None:
            # attn_mask: [B, T], True 表示有效 token
            key_mask = attn_mask[:, None, None, :].bool()
            att = att.masked_fill(~key_mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    统一后的输入形式：

    输入序列:
        [goal_x, goal_y, start_x, start_y, a1_x, a1_y, r1, R1, s2_x, s2_y, ...]

    type 序列:
        [GOAL, GOAL, STATE, STATE, ACTION, ACTION, REWARD, RETURN, STATE, STATE, ...]

    训练时：
        targets != None, loss_mask != None
    推理时：
        targets = None, loss_mask = None
        只取 logits[:, -1, :] 作为 next-token 分布
    """
    def __init__(self, config):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.type_size = config.type_size
        self.block_size = config.block_size

        # +1 for PAD token (pad_token_id == vocab_size)
        self.token_emb = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.type_emb = nn.Embedding(config.type_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 输出同样是 vocab_size + 1，最后一类保留给 PAD
        self.head = nn.Linear(config.n_embd, config.vocab_size + 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, type_idx, targets=None, attn_mask=None, loss_mask=None):
        """
        idx      : [B, T]
        type_idx : [B, T]
        targets  : [B, T] or None
        attn_mask: [B, T] bool or None
        loss_mask: [B, T] float/bool or None
        """
        assert idx.shape == type_idx.shape, "idx and type_idx must have the same shape"

        B, T = idx.size()
        assert T <= self.block_size, "sequence length exceeds block size"

        if attn_mask is None:
            attn_mask = torch.ones_like(idx, dtype=torch.bool)

        tok_emb = self.token_emb(idx)
        typ_emb = self.type_emb(type_idx)
        pos_emb = self.pos_emb[:, :T, :]

        x = self.drop(tok_emb + typ_emb + pos_emb)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none"
            )

            if loss_mask is not None:
                loss_mask = loss_mask.reshape(-1).float()
                ce = ce * loss_mask
                loss = ce.sum() / loss_mask.sum().clamp_min(1.0)
            else:
                loss = ce.mean()

        return logits, loss