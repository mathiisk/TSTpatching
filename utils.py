from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# ------ Hook Utilities ------
def _register_hooks(model: nn.Module, substrings: List[str], hook_fn) -> dict:
    handles = {}
    for name, module in model.named_modules():
        if any(sub in name for sub in substrings):
            handles[module] = module.register_forward_hook(hook_fn)
    return handles

def _remove_hooks(handles: dict) -> None:
    for handle in handles.values(): handle.remove()

# ------ Activation Caching & Patching ------
def run_and_cache(model: nn.Module, x: torch.Tensor, targets: List[str]) -> dict:
    cache = {}
    def hook(m, inp, out):
        val = out[0] if isinstance(out, tuple) else out
        cache[m] = val.detach().clone()
    handles = _register_hooks(model, targets, hook)
    model(x)
    _remove_hooks(handles)
    return cache


def get_attention_saliency(
    model: nn.Module,
    instance: torch.Tensor,
    layer_idx: int,
    head_idx: int
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    instance = instance.unsqueeze(0).to(device)

    attn_result = {}

    sa = model.transformer_encoder.layers[layer_idx].self_attn
    original_forward = sa.forward

    def forward_with_weights(query, key, value, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return original_forward(query, key, value, **kwargs)

    sa.forward = forward_with_weights

    def hook(module, input, output):
        weights = output[1]  # (batch, heads, seq_len, seq_len)
        if weights is not None:
            attn_result['weights'] = weights.detach().cpu()

    handle = sa.register_forward_hook(hook)

    try:
        _ = model(instance)
    finally:
        handle.remove()
        sa.forward = original_forward  # Restore original

    full_attn = attn_result['weights'][0, head_idx]  # (seq_len, seq_len)
    saliency = full_attn.mean(dim=0)  # mean over queries (axis 0)

    return saliency  # (seq_len,)


def patch_activations(model: nn.Module, x: torch.Tensor, cache: dict, targets: List[str]) -> torch.Tensor:
    def patch_hook(m, inp, out):
        return cache[m]
    handles = _register_hooks(model, targets, patch_hook)
    out = model(x)
    _remove_hooks(handles)
    return out

# ------ Probability Utilities ------
def get_probs(model: nn.Module, instance: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(instance.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs

# ------ Transformer-Specific Helpers ------
def get_encoder_inputs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    z = x.transpose(1,2)
    for conv, bn in ((model.conv1, model.bn1), (model.conv2, model.bn2), (model.conv3, model.bn3)):
        z = torch.relu(bn(conv(z)))
    z = z.transpose(1,2)
    return z + model.pos_enc

# ------ Attention Head Patching ------
def patch_attention_head(model: nn.Module, clean: torch.Tensor, corrupt: torch.Tensor,
                         layer_idx: int, head_idx: int) -> torch.Tensor:
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)
    layer_mod = model.transformer_encoder.layers[layer_idx]
    attn_mod = layer_mod.self_attn
    cache = {}
    handle = attn_mod.register_forward_hook(lambda m, i, o: cache.setdefault(m, o[0].detach().clone()))
    _ = model(clean_b)
    handle.remove()
    clean_val = cache[attn_mod]  # (1, seq_len, d_model)
    B, S, E = clean_val.shape
    H = attn_mod.num_heads
    d = E // H
    clean_heads = clean_val.view(B, S, H, d)[:, :, head_idx, :]
    def patch_hook(m, inp, out):
        out_val = out[0]
        out_heads = out_val.view(B, S, H, d)
        out_heads[:, :, head_idx, :] = clean_heads
        return out_heads.reshape(B, S, E)
    handle2 = attn_mod.register_forward_hook(patch_hook)
    logits = model(corrupt_b)
    handle2.remove()
    return logits

# ------ Sweeping over Heads ------
def sweep_heads(model: nn.Module, clean: torch.Tensor, corrupt: torch.Tensor,
                            num_classes: int) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)
    L = len(model.transformer_encoder.layers)
    H = model.transformer_encoder.layers[0].self_attn.num_heads
    patch_probs = np.zeros((L, H, num_classes))
    for l in range(L):
        for h in range(H):
            logits = patch_attention_head(model, clean, corrupt, l, h)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            patch_probs[l, h] = probs
    return patch_probs

# ------ Plotting ------
def plot_influence(patch_probs: np.ndarray, baseline_probs: np.ndarray, true_label: int) -> None:
    delta = patch_probs[:, :, true_label] - baseline_probs[true_label]
    L, H = delta.shape
    fig, ax = plt.subplots(figsize=(H * 0.6, L * 0.6))
    sns.heatmap(delta, annot=True, fmt="+.2f",
                xticklabels=[f"H{h}" for h in range(H)],
                yticklabels=[f"L{l}" for l in range(L)],
                center=0, cmap="coolwarm", ax=ax)
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("ΔP(true) by Layer & Head")
    plt.tight_layout(); plt.show()


def plot_mini_deltas(patch_probs: np.ndarray, baseline_probs: np.ndarray, class_labels: List[str]=None) -> None:
    pp = np.asarray(patch_probs)
    bp = np.asarray(baseline_probs)
    if pp.ndim != 3 or bp.ndim != 1 or pp.shape[2] != bp.shape[0]:
        raise ValueError(f"Shapes mismatch: patch_probs {pp.shape}, baseline_probs {bp.shape}")
    L, H, C = pp.shape
    cl = class_labels or [str(i) for i in range(C)]
    delta = pp - bp[None, None, :]
    vmax = np.abs(delta).max()
    fig, axes = plt.subplots(L, H, figsize=(2.2 * H, 2 * L), squeeze=False)
    for i in range(L):
        for j in range(H):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            data = delta[i, j, :].reshape(1, C)
            ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect='auto')
            for k in range(C):
                ax.text(k, 0, f"{data[0, k]:+.2f}", ha='center', va='center', fontsize=6)
            if i == 0:
                ax.set_title(f"H{j}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"L{i}", rotation=0, labelpad=12, fontsize=9)
    plt.suptitle("Change in Class Probabilities from Patching Each Head", y=1.02)
    plt.tight_layout(); plt.show()


def plot_timeseries_with_attention_overlay(instances: List[torch.Tensor], saliencies: List[torch.Tensor], labels: List[str], title: str = "Time Series with Attention Overlay") -> None:
    n = len(instances)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)

    for idx, (instance, saliency) in enumerate(zip(instances, saliencies)):
        instance = instance.detach().cpu().numpy()
        saliency = saliency.detach().cpu().numpy()

        saliency = saliency / (saliency.max() + 1e-8)

        ax = axes[idx, 0]
        seq_len, input_dim = instance.shape
        time = np.arange(seq_len)

        for d in range(input_dim):
            ax.plot(time, instance[:, d], label=f"dim {d}", linewidth=1.5)

        for t in range(seq_len):
            alpha = float(saliency[t])
            ax.axvspan(t - 0.5, t + 0.5, color='orange', alpha=alpha * 0.4)

        ax.set_title(labels[idx] if labels else f"Instance {idx}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True)
        if input_dim <= 5:
            ax.legend(fontsize=7)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

