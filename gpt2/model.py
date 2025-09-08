from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
#--------------------------------------------

class MultiheadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # key, query and value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C  = x.size()
        # calculate q, k, v for all heads in batch -> move head to be batch
        # nh is "num heads", hs is "head dimention", and C is "embedding dimension" = nh * hs
        # GPT2: 124M, n_head = 12, hs = 64, so nh * hs = 12 * 64 = 768 embedding dimention

        qkv = self.c_attn(x) # shape of (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2) # shape of (B, T, C) for each q,k,v

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # shape of (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # shape of (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # shape of (B, nh, T, hs)

        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) shape of (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # shape of (B, T, C)
        y = self.c_proj(y)
        return y
    
       

class Block(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.n_embd)
        self.layer_norm2 = nn.LayerNorm(config.n_embd)
        self.attention = MultiheadAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max seq length
    vocab_size: int = 50257 # # num of tokens
    n_layer:int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                hidden = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                layer_norm = nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
    
    def forward(self, idx, target=None):
        device = idx.device
        B, T = idx.shape
        
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)

        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)

        for block in self.transformer.hidden:
            x = block(x)
        
        x = self.transformer.layer_norm(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load models pretrained weights"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained gpt: %s' % model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        # state dict for both HF and our model
        sd = model.state_dict() 
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a Huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # provide diagnostics when keys/shapes don't match so it's easier to debug
        if len(sd_keys_hf) != len(sd_keys):
            missing_in_hf = set(sd_keys) - set(sd_keys_hf)
            missing_in_local = set(sd_keys_hf) - set(sd_keys)
            print(f"WARNING: number of keys differ: hf={len(sd_keys_hf)} local={len(sd_keys)}")
            if len(missing_in_hf) > 0:
                print("keys present in local model but missing in HF (sample up to 10):")
                for kk in list(missing_in_hf)[:10]:
                    print("  ", kk)
            if len(missing_in_local) > 0:
                print("keys present in HF but missing in local model (sample up to 10):")
                for kk in list(missing_in_local)[:10]:
                    print("  ", kk)

        for k in sd_keys_hf:
            if k not in sd:
                # if the HF key doesn't exist in our model, list it and continue so we can see more mismatches
                print(f"SKIP: HF key not found in local model: {k}")
                continue
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                if sd_hf[k].shape[::-1] != sd[k].shape:
                    raise AssertionError(f"transposed shape mismatch for key {k}: hf {tuple(sd_hf[k].shape)} transposed -> {tuple(sd_hf[k].shape[::-1])} != local {tuple(sd[k].shape)}")
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                if sd_hf[k].shape != sd[k].shape:
                    raise AssertionError(f"shape mismatch for key {k}: hf {tuple(sd_hf[k].shape)} != local {tuple(sd[k].shape)}")
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
#------------------------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")

num_return_sequences = 5
max_length = 30

#-------
import tiktoken 

class DataLoaderLite:
    def __init__(self, B, T) -> None:
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B*T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    
#-------
train_loader = DataLoaderLite(B=4, T=32)

model = GPT(GPTConfig)
model.to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
for i in range(500):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

print(loss)



import sys; sys.exit(0)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded)


