# Relational Distillation: Distilling Transformer Attention Maps to PSSA Entity Graphs

This document specifies the mathematical formulation and implementation plan for **Relational Distillation** in the Persistent-Sparse-Semantic-Architecture (PSSA). This method transfers structural syntax and relational dependencies from a pre-trained Transformer teacher (e.g., GPT-2) to a PSSA student before the pre-training campaign starts.

---

## 1. Scientific Motivation

Traditional knowledge distillation in heterogeneous architectures focuses on **activation matching** (aligning hidden states layer-by-layer) or **logit matching** (KL-divergence on output probabilities). While effective for transferring static semantic representations, these methods do not explicitly transfer the **relational structure** of language.

A Transformer models relationships implicitly via its multi-head attention maps, showing how strongly tokens attend to one another (e.g., connecting a subject to its verb or object across long context windows). 
PSSA represents these relationships **explicitly** as a directed causal dependency graph $D \in \mathbb{R}^{M \times M}$ between active memory slots.

**Relational Distillation** projects the Transformer's token-level attention maps directly into PSSA's slot-level dependency graph. This allows PSSA to inherit the teacher's causal and relational parsing directly at Step 0, significantly accelerating pre-training.

```text
Transformer Teacher                             PSSA Student
[Token Sequence T]                             [Active Memory Slots M]
       │                                                 │
       ▼                                                 ▼
Attention Maps (A) ───[Projection Operator P]───► Causal Graph (D)
       │                                                 │
       └───────────────────[Minimize Loss]───────────────┘
```

---

## 2. Mathematical Formulation

### 2.1 Variables and Dimensions

*   **Transformer Attention Map**: For a given sequence length $T$ and attention head $h \in \{1, \dots, H\}$, the teacher attention map is:
    $$A_h \in \mathbb{R}^{T \times T}$$
    where $\sum_j A_{h, i, j} = 1.0$ represents the softmax-normalized attention scores.

*   **PSSA Entity Dependency Graph**: For $M$ active memory slots (where $M = N_{scopes} \times N_{entities}$, default $3 \times 4 = 12$), PSSA's directed dependency graph at layer $l$ is:
    $$D_l \in \mathbb{R}^{M \times M}$$
    where $D_{l, i, j} \in [0, 1]$ is computed via the slot-to-slot dependency projection:
    $$D_{l} = \text{sigmoid}\left(\frac{Q_{dep} K_{dep}^T}{\sqrt{d_{dep}}}\right)$$

---

### 2.2 The Attention Projection Operator

To align the token-to-token attention space $\mathbb{R}^{T \times T}$ with the slot-to-slot dependency space $\mathbb{R}^{M \times M}$, we construct an **Alignment Projection Operator** $P \in \mathbb{R}^{T \times M}$.

During PSSA's forward pass, each token step $t \in \{1, \dots, T\}$ generates a retrieval allocation vector $\text{AllocScores}_t \in \mathbb{R}^M$ mapping the current input to the active entity memory slots. We collect these vectors across the sequence length to form the projection matrix:
$$P = \begin{bmatrix} \text{AllocScores}_1 \\ \text{AllocScores}_2 \\ \vdots \\ \text{AllocScores}_T \end{bmatrix} \in \mathbb{R}^{T \times M}$$

where $P_{t, m}$ represents the probability/affinity of token $t$ being routed to memory slot $m$. We normalize $P$ along the columns to represent slot-to-token allocation weights:
$$\bar{P}_{t, m} = \frac{P_{t, m}}{\sum_{k=1}^T P_{k, m} + \epsilon}$$

Using this operator, we project the Transformer's attention map $A_h$ from token space to slot space:
$$A^{\text{projected}}_h = \bar{P}^T A_h \bar{P} \in \mathbb{R}^{M \times M}$$

---

### 2.3 Relational Distillation Loss

The Relational Distillation loss $E_{\text{relation}}$ minimizes the Frobenius norm distance between the PSSA dependency matrix $D_l$ and the projected attention maps across all heads $H$ and layers:

$$E_{\text{relation}} = \frac{1}{L \cdot H} \sum_{l=1}^L \sum_{h=1}^H \| D_l - A^{\text{projected}}_{l, h} \|_F^2$$

The total Step-0 initialization objective is regularized by PSSA's physical energy terms:

$$\mathcal{L}_{\text{init}} = E_{\text{pred}} + \lambda_{\text{relation}} E_{\text{relation}} + \lambda_{\text{conn}} E_{\text{conn}} + \lambda_{\text{stab}} E_{\text{stab}} + \lambda_{\text{info}} E_{\text{info}}$$

---

## 3. Implementation Plan

### 3.1 Code Additions

#### 1. In `pssa/energy.py`:
Add the relation projection and loss calculations.
```python
def compute_relational_energy(dependency_matrix, teacher_attention_maps, allocation_probabilities):
    """
    Args:
        dependency_matrix (Tensor): PSSA slot dependency graph [batch, M, M]
        teacher_attention_maps (Tensor): Transformer attention maps [batch, num_heads, T, T]
        allocation_probabilities (Tensor): Token-to-slot routing affinities [batch, T, M]
    Returns:
        Tensor: Scalar relational loss
    """
    batch_size, T, M = allocation_probabilities.shape
    num_heads = teacher_attention_maps.shape[1]
    
    # Column-normalize allocation probabilities
    col_sums = allocation_probabilities.sum(dim=1, keepdim=True) + 1e-8
    P_bar = allocation_probabilities / col_sums # [batch, T, M]
    
    total_loss = 0.0
    for h in range(num_heads):
        A_h = teacher_attention_maps[:, h, :, :] # [batch, T, T]
        # Project token attention to slot space: P_bar^T @ A_h @ P_bar
        A_proj = torch.bmm(torch.bmm(P_bar.transpose(1, 2), A_h), P_bar) # [batch, M, M]
        total_loss += torch.mean((dependency_matrix - A_proj) ** 2)
        
    return total_loss / num_heads
```

#### 2. In `pssa/optimizer.py`:
Expose `optimize_least_action` to support `teacher_attention` and `E_relation` constraints.

---

### 3.2 Step-0 Execution Workflow

1.  **Teacher Hook Setup**: Register PyTorch forward hooks on the teacher Transformer (e.g., `gpt2`) to extract both:
    *   Hidden layer activations ($H^{\text{teacher}}$)
    *   Multi-head self-attention matrices ($A^{\text{teacher}}$)
2.  **Telemetry Hook Setup**: Pass `return_telemetry=True` to the PSSA student to collect:
    *   Active entity dependency matrices ($D$)
    *   Token allocation scores ($P$)
3.  **Pre-optimization Phase**: For 250 steps before the campaign starts, feed a text batch to both models, compute $\mathcal{L}_{\text{init}}$, and perform backpropagation to initialize PSSA's weights.
4.  **Campaign Launch**: Begin FineWeb-EDU pre-training starting with structured weights.
