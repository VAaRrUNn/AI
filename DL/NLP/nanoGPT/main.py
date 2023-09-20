import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import tqdm

# hyperparameters
batch_size = 16
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0


torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class SelfAttention(nn.Module):
  def __init__(self, head_size):
    """
    Implementation of self attention

    NOTE: This should only be used with MultiHeadAttention cause It doesn't return the same shape as input
    """

    super().__init__()
    self.k = nn.Linear(n_embd, head_size)
    self.q = nn.Linear(n_embd, head_size)
    self.v = nn.Linear(n_embd, head_size)

    # used when we want it to be stored in state_dict but not trainable, can access using self.tril
    self.register_buffer("tril", torch.tril(
            torch.ones(block_size, block_size)))
    
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    '''
    Args:
      x: input of shape (B, T, n_embd)
    '''
    B, T, C = x.shape

    k = self.k(x) # (B, T, n_head)
    q = self.q(x) # (B, T, n_head)
    v = self.v(x) # (B, T, n_head)

    w = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)

    # IMPORTANT: [:T, :T] cause the shape of x will be undefined when generation.
    w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    w = F.softmax(w, dim=-1) # (B, T, T)
    w = self.dropout(w)

    out = w @ v # (B, T, n_head)
    # print(f"k {k.shape}, q: {q.shape}, v: {v.shape}, out: {out.shape}")
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, head_size):
    """
    Implementation of Masked multihead attention

    NOTE: This is unusual implementation of MultiHeadAttention, normally we combine everything in single k, q, v.
    IMPORTANT: We have to define all the SelfAttention blocks in here in order for them to produce different values
    """
    super().__init__()
    self.selfAttention = nn.ModuleList([SelfAttention(head_size=head_size) for _ in range(n_heads)])               
    self.projection = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    """
    Args:
      x: inputs of shape (B, T, C)
        B: batch_size
        T: block_size
        C: n_embd
    """
    x = torch.cat([l(x) for l in self.selfAttention], dim=-1) # (B, T, head_size) * n_heads
    # out = torch.cat(out, dim=-1) # (B, T, head_size * n_heads) -> (B, T, C)
    x = self.dropout(self.projection(x))
    return x


class FeedForward(nn.Module):
  """
  Performs feed forward after the MultiHeadAttention
  """
  def __init__(self, n_embd):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.layers(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.attention = MultiHeadAttention(n_heads=n_heads, head_size=head_size)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    self.ffw = FeedForward(n_embd)

  def forward(self, x):
    x = x + self.attention(self.ln1(x)) # (B, T, C)
    x = x + self.ffw(self.ln2(x)) # (B, T, C)
    return x


class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (vocab_size, C)
    self.position_embedding = nn.Embedding(block_size, n_embd) # (T, C)
    self.block = Block(n_embd, n_heads=n_heads)
    self.ln = nn.LayerNorm(n_embd)
    self.final = nn.Linear(n_embd, vocab_size)

  def forward(self, x, targets=None):
    '''
    GPT forward pass
    Args:
      x: input of shape (B, T)
      targets: targets of shape (B, T)
    '''
    B, T = x.shape
    token_emb = self.token_embedding_table(x) # (B, T, C)
    pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C)
    x = token_emb + pos_emb # (B, T, C)
    attention = self.block(x) # (B, T, C)
    out = self.ln(attention) # (B, T, C) ,.. [C is n_embd]
    logits = self.final(out) # (B, T, vocab_size)

    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate0(self, x, max_seq_len=100):
    """
    For generating outputs from the model given a input x
    Args:
      x: input of shape (B, T).
    """
    x = x.to(device)
    for i in range(max_seq_len):
      logits, loss = self(x)
      logits = logits[:, -1, :] # (B, C)
      logits = F.softmax(logits, dim=-1)
      pred = torch.multinomial(logits, num_samples=1) # (B, 1)
      x = torch.cat([x, pred], dim=-1)  # (B, ?)
    return x

  def generate(self, idx, max_new_tokens=10):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 32  # what is the maximum context length for predictions?
max_iters = 50000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
n_heads = 4
n_layer = 4
dropout = 0.0


model = GPT()
model = model.to(device)
model.load_state_dict(torch.load("nano_gpt.pt", map_location=device))

optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
train_loss = []

def train(model,
          optimizer,
          epochs):
  model.train()

  for epoch in tqdm(range(epochs)):
    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # stats
    train_loss.append(loss.item())




# train(model=model,
#       optimizer=optimizer,
#       epochs=max_iters)


# import matplotlib.pyplot as plt
# plt.plot(range(len(train_loss)), train_loss)

torch.save(model.state_dict(), "nano_gpt.pt")


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))