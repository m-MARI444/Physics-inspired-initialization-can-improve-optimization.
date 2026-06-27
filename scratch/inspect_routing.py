import torch
import torch.nn.functional as F
from model.pssa_gpt import PSSAGPT

# Load tokenizer and compact vocabulary
vocab_path = "checkpoints/pssa_gpt_vocab.pth"
bpe_to_compact, compact_to_bpe = torch.load(vocab_path)
checkpoint = "checkpoints/pssa_gpt.pth"

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PSSAGPT(
    vocab_size=2203,
    d_model=64,
    num_layers=3,
    num_slots=5,
    num_entities=4,
    num_scopes=3
).to(device)

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
actor_mask = torch.zeros(50257, dtype=torch.bool, device=device)
for tid in [tokenizer.encode(a)[0] for a in ["Spot", "Lily", "Tim", "Kitty"]]:
    actor_mask[tid] = True

story = "Spot played with Lily. He liked the new toy she gave him."
raw_tokens = tokenizer.encode(story)
compact_tokens = [bpe_to_compact.get(tid, 0) for tid in raw_tokens]
tokens_tensor = torch.tensor([compact_tokens], dtype=torch.long, device=device)
decoded_tokens = [tokenizer.decode([t]) for t in raw_tokens]

print("Tokens:", decoded_tokens)

with torch.no_grad():
    logits, gates, slots, pre_wave, adj, entities, retrievals, write_candidates, scope_weights, recon_loss = model(
        tokens_tensor,
        actor_mask=actor_mask
    )

pron_step = None
for t, decoded_w in enumerate(decoded_tokens):
    if decoded_w.strip().lower() == "he":
        pron_step = t
        break

print("Pronoun Step:", pron_step)
print("Retrieval scores at pronoun step (Top Layer):")
hier_scores = retrievals[0, pron_step, -1, :]
print(hier_scores)

# Trace Spot's steps
spot_step = None
for t, decoded_w in enumerate(decoded_tokens):
    if decoded_w.strip().lower() == "spot":
        spot_step = t
        break

print("\nSpot Step:", spot_step)
print("Spot update U_t shape:", write_candidates[0, spot_step, -1, 2, :].shape)
# Let's inspect similarity to E_prev at spot step
# E_prev is the entity banks at spot_step
# We can retrieve it by looking at entities at step spot_step - 1 (or 0)
E_prev = entities[0, max(0, spot_step-1), -1, :, :, :]
u_t = write_candidates[0, spot_step, -1, 2, :]
E_flat = E_prev.view(12, 64)
similarity = torch.matmul(u_t, E_flat.t()) / (64 ** 0.5)
alloc_soft = F.softmax(similarity / 0.15, dim=-1)
print("Spot similarity to registers:")
print(similarity)
print("Spot alloc_soft:")
print(alloc_soft)

# Calculate Shannon Entropy and Ambiguity Index
h_alloc = - (alloc_soft * torch.log2(alloc_soft + 1e-9)).sum(dim=-1)
alpha = torch.sigmoid(2.0 * (h_alloc - 1.2))
print("Shannon Entropy:", h_alloc.item())
print("Ambiguity Index alpha:", alpha.item())
