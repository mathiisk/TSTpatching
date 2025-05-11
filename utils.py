from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import torch
import torch.nn as nn
import networkx as nx

# HOOK FUNCTIONS
def _register_hooks(model: nn.Module, substrings: List[str], hook_fn) -> dict:
    handles = {}
    for name, module in model.named_modules():
        if any(sub in name for sub in substrings):
            handles[module] = module.register_forward_hook(hook_fn)
    return handles

def _remove_hooks(handles: dict) -> None:
    for handle in handles.values(): handle.remove()


# CACHING ATTENTION FUNCTIONS
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

# PROBABILITY FUNCTION
def get_probs(model: nn.Module, instance: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(instance.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs

# TRANSFORMER HELPER
def get_encoder_inputs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    z = x.transpose(1,2)
    for conv, bn in ((model.conv1, model.bn1), (model.conv2, model.bn2), (model.conv3, model.bn3)):
        z = torch.relu(bn(conv(z)))
    z = z.transpose(1,2)
    return z + model.pos_enc

# PATCH SINGLE ATTENTION HEAD
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


from contextlib import contextmanager

@contextmanager
def with_head_patch(model, clean, layer_idx, head_idx):
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)

    # Cache clean output
    layer_mod = model.transformer_encoder.layers[layer_idx]
    attn_mod = layer_mod.self_attn
    cache = {}

    def save_hook(m, i, o):
        cache[m] = o[0].detach().clone()

    handle1 = attn_mod.register_forward_hook(save_hook)
    _ = model(clean_b)
    handle1.remove()

    clean_val = cache[attn_mod]
    B, S, E = clean_val.shape
    H = attn_mod.num_heads
    d = E // H
    clean_heads = clean_val.view(B, S, H, d)[:, :, head_idx, :]

    # Patch hook
    def patch_hook(m, inp, out):
        out_val = out[0]
        heads = out_val.view(B, S, H, d)
        heads[:, :, head_idx, :] = clean_heads
        return heads.reshape(B, S, E)

    handle2 = attn_mod.register_forward_hook(patch_hook)
    try:
        yield
    finally:
        handle2.remove()



def patch_mlp_activation(model: nn.Module, clean: torch.Tensor, corrupt: torch.Tensor, layer_idx: int) -> torch.Tensor:
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)
    layer_mod = model.transformer_encoder.layers[layer_idx]
    mlp_layer = layer_mod.linear2  # Patch after second linear layer

    cache = {}

    def cache_hook(m, inp, out):
        cache[m] = out.detach().clone()

    handle_cache = mlp_layer.register_forward_hook(cache_hook)
    _ = model(clean_b)
    handle_cache.remove()

    def patch_hook(m, inp, out):
        return cache[m]

    handle_patch = mlp_layer.register_forward_hook(patch_hook)
    logits = model(corrupt_b)
    handle_patch.remove()

    return logits

def patch_attention_head_at_position(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    pos_idx: int
) -> torch.Tensor:
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)

    layer_mod = model.transformer_encoder.layers[layer_idx]
    attn_mod = layer_mod.self_attn
    cache = {}

    def cache_hook(m, inp, out):
        cache[m] = out[0].detach().clone()

    handle_cache = attn_mod.register_forward_hook(cache_hook)
    _ = model(clean_b)
    handle_cache.remove()

    clean_val = cache[attn_mod]  # (1, seq_len, d_model)
    B, S, E = clean_val.shape
    H = attn_mod.num_heads
    d = E // H

    clean_heads = clean_val.view(B, S, H, d)

    def patch_hook(m, inp, out):
        out_val = out[0]
        heads = out_val.view(B, S, H, d)
        heads[:, pos_idx, head_idx, :] = clean_heads[:, pos_idx, head_idx, :]
        return heads.reshape(B, S, E)

    handle_patch = attn_mod.register_forward_hook(patch_hook)
    logits = model(corrupt_b)
    handle_patch.remove()

    return logits


def patch_mlp_at_position(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    layer_idx: int,
    pos_idx: int
) -> torch.Tensor:
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)

    layer_mod = model.transformer_encoder.layers[layer_idx]
    mlp_layer = layer_mod.linear2
    cache = {}

    def cache_hook(m, inp, out):
        cache[m] = out.detach().clone()

    handle_cache = mlp_layer.register_forward_hook(cache_hook)
    _ = model(clean_b)
    handle_cache.remove()

    clean_val = cache[mlp_layer]  # (batch, seq_len, d_model)

    def patch_hook(m, inp, out):
        patched = out.clone()
        patched[:, pos_idx, :] = clean_val[:, pos_idx, :]
        return patched

    handle_patch = mlp_layer.register_forward_hook(patch_hook)
    logits = model(corrupt_b)
    handle_patch.remove()

    return logits


# SWEEP OVER HEADS
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


def sweep_mlp_layers(model: nn.Module, clean: torch.Tensor, corrupt: torch.Tensor, num_classes: int) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)
    L = len(model.transformer_encoder.layers)
    patch_probs = np.zeros((L, num_classes))
    for l in range(L):
        logits = patch_mlp_activation(model, clean, corrupt, l)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        patch_probs[l] = probs
    return patch_probs


def sweep_attention_head_positions(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    num_classes: int
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)

    patch_probs = np.zeros((num_layers, num_heads, seq_len, num_classes))

    for l in range(num_layers):
        for h in range(num_heads):
            for pos in range(seq_len):
                logits = patch_attention_head_at_position(model, clean, corrupt, l, h, pos)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                patch_probs[l, h, pos] = probs

    return patch_probs  # (layers, heads, positions, classes)


def sweep_mlp_positions(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    num_layers: int,
    seq_len: int,
    num_classes: int
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)

    patch_probs = np.zeros((num_layers, seq_len, num_classes))

    for l in range(num_layers):
        for pos in range(seq_len):
            logits = patch_mlp_at_position(model, clean, corrupt, l, pos)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            patch_probs[l, pos] = probs

    return patch_probs  # (layers, positions, classes)


def find_critical_patches(
    patch_probs: np.ndarray,
    baseline_probs: np.ndarray,
    true_label: int,
    threshold: float = 0.05
) -> List[Tuple[int, int, int, float]]:
    delta = patch_probs[:, :, :, true_label] - baseline_probs[true_label]
    critical = []

    L, H, P = delta.shape

    for l in range(L):
        for h in range(H):
            for p in range(P):
                if delta[l, h, p] > threshold:
                    critical.append((l, h, p, delta[l, h, p]))

    return critical



def build_causal_graph(critical_patches: List[Tuple[int, int, int, float]]) -> nx.DiGraph:
    G = nx.DiGraph()

    for layer, head, pos, delta in critical_patches:
        timestep_node = f"Time {pos}"
        head_node = f"L{layer}H{head}"

        G.add_edge(timestep_node, head_node, weight=delta)

    return G


# PLOTTING FUNCTIONS
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


def plot_mlp_influence(patch_probs: np.ndarray, baseline_probs: np.ndarray, true_label: int) -> None:
    delta = patch_probs[:, true_label] - baseline_probs[true_label]
    L = delta.shape[0]
    fig, ax = plt.subplots(figsize=(L * 0.6, 4))
    sns.barplot(x=np.arange(L), y=delta, hue=np.arange(L), palette="coolwarm", dodge=False, ax=ax, legend=False)
    ax.set_xlabel("Layer")
    ax.set_ylabel("ΔP(True Label)")
    ax.set_title("ΔP(True) by MLP Layer")
    plt.tight_layout()
    plt.show()


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


def plot_head_position_patch_heatmap(
    patch_probs: np.ndarray,
    baseline_probs: np.ndarray,
    true_label: int,
    layer_idx: int,
    head_idx: int,
    title: str = "Attention Head Patch Effect by Position"
) -> None:
    """
    Plots heatmap for a specific head's ΔP(true_label) across positions.
    patch_probs: (layers, heads, positions, classes)
    """
    delta = patch_probs[layer_idx, head_idx, :, true_label] - baseline_probs[true_label]
    seq_len = delta.shape[0]

    fig, ax = plt.subplots(figsize=(12, 2))
    sns.heatmap(delta[np.newaxis, :], cmap="coolwarm", center=0,
                xticklabels=20, yticklabels=[f"L{layer_idx} H{head_idx}"],
                cbar_kws={'label': 'ΔP(True Label)'}, ax=ax)
    ax.set_xlabel("Position (Timestep)")
    ax.set_ylabel("")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()



def plot_mlp_position_patch_heatmap(
        patch_probs: np.ndarray,
        baseline_probs: np.ndarray,
        true_label: int,
        title: str = "Patch Effect by Layer and Position"
) -> None:
    delta = patch_probs[:, :, true_label] - baseline_probs[true_label]
    L, P = delta.shape

    fig, ax = plt.subplots(figsize=(12, 0.5 * L))
    sns.heatmap(delta, cmap="coolwarm", center=0,
                xticklabels=20, yticklabels=[f"Layer {i}" for i in range(L)],
                cbar_kws={'label': 'ΔP(True Label)'}, ax=ax)
    ax.set_xlabel("Position (Timestep)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_causal_graph(G: nx.DiGraph, title="Critical Circuits Graph") -> None:
    pos = nx.spring_layout(G, k=0.8, seed=42)

    weights = [G[u][v]['weight'] for u,v in G.edges()]
    edge_colors = ["red" if w > 0 else "blue" for w in weights]

    # Color nodes by type
    node_colors = []
    for node in G.nodes():
        if node.startswith("Time"):
            node_colors.append("lightgreen")
        else:
            node_colors.append("lightblue")

    # Size nodes by degree
    node_sizes = []
    for node in G.nodes():
        deg = G.degree(node)
        node_sizes.append(300 + deg * 100)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=[abs(w)*10 for w in weights], arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    plt.axis('off')
    plt.show()



def plot_structured_graph(G):
    pos = {}
    # Separate node groups
    time_nodes = [n for n in G.nodes if 'Time' in n]
    head_nodes = [n for n in G.nodes if 'H' in n]

    # Assign positions: time nodes in one row, heads in another
    for i, node in enumerate(sorted(time_nodes)):
        pos[node] = (i, 1)
    for i, node in enumerate(sorted(head_nodes)):
        pos[node] = (i, 0)

    # Size and color nodes by degree or role
    node_sizes = [500 + 1000 * G.degree(n) for n in G.nodes]
    node_colors = ['seagreen' if 'Time' in n else 'steelblue' for n in G.nodes]

    # Draw with directional arrows
    plt.figure(figsize=(14, 6))
    nx.draw_networkx(G, pos, with_labels=True, node_color=node_colors,
                     node_size=node_sizes, edge_color='maroon',
                     arrows=True, font_size=8, width=1.5)
    plt.title("Structured Attribution Graph (Time → Head)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def patch_multiple_attention_heads_positions(
    model: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    critical_edges: List[Tuple[str, str, dict]]
) -> torch.Tensor:
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)

    cache = {}

    # Step 1: Capture clean activations
    hook_layers = set()
    for u, v, _ in critical_edges:
        layer = int(v[1])
        hook_layers.add(layer)

    hook_layers = list(hook_layers)

    def cache_hook(m, inp, out):
        if m not in cache:
            cache[m] = out[0].detach().clone()

    handles_cache = []
    for layer_idx in hook_layers:
        attn_mod = model.transformer_encoder.layers[layer_idx].self_attn
        handles_cache.append(attn_mod.register_forward_hook(cache_hook))

    _ = model(clean_b)
    for h in handles_cache: h.remove()

    # Step 2: Patch during corrupted forward
    def patch_hook_factory(layer_idx):
        def patch_fn(m, inp, out):
            out_val = out[0]
            B, S, E = out_val.shape
            H = m.num_heads
            d = E // H
            heads = out_val.view(B, S, H, d)

            clean_val = cache[m]
            clean_heads = clean_val.view(B, S, H, d)

            for u, v, _ in critical_edges:
                l = int(v[1])
                h = int(v[3])
                pos = int(u.split()[1])

                if l == layer_idx:
                    heads[:, pos, h, :] = clean_heads[:, pos, h, :]

            return heads.reshape(B, S, E)

        return patch_fn

    handles_patch = []
    for layer_idx in hook_layers:
        attn_mod = model.transformer_encoder.layers[layer_idx].self_attn
        handles_patch.append(attn_mod.register_forward_hook(patch_hook_factory(layer_idx)))

    logits = model(corrupt_b)

    for h in handles_patch: h.remove()

    return logits


def capture_all_heads(model, x, num_layers, num_heads):
    cache = {}

    def make_hook(layer_idx, head_idx):
        def hook(m, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            B, S, E = val.shape
            d = E // num_heads
            heads = val.view(B, S, num_heads, d)
            cache[(layer_idx, head_idx)] = heads[:, :, head_idx, :].detach().clone()
        return hook

    handles = []
    for l in range(num_layers):
        mod = model.transformer_encoder.layers[l].self_attn
        for h in range(num_heads):
            handles.append(mod.register_forward_hook(make_hook(l, h)))

    _ = model(x.unsqueeze(0).to(next(model.parameters()).device))

    for h in handles:
        h.remove()

    return cache



def sweep_head_to_head_influence(model, clean, corrupt) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)

    L = len(model.transformer_encoder.layers)
    H = model.transformer_encoder.layers[0].self_attn.num_heads

    # Step 1: get baseline outputs
    corrupt_cache = capture_all_heads(model, corrupt, L, H)

    influence = np.zeros((L, H, L, H))  # [source_head, target_head]

    for l_src in range(L):
        for h_src in range(H):
            with with_head_patch(model, clean, l_src, h_src):
                patched_cache = capture_all_heads(model, corrupt, L, H)

                for l_tgt in range(L):
                    for h_tgt in range(H):
                        if (l_tgt, h_tgt) in corrupt_cache:
                            diff = torch.nn.functional.mse_loss(
                                patched_cache[(l_tgt, h_tgt)],
                                corrupt_cache[(l_tgt, h_tgt)],
                                reduction='mean'
                            ).item()
                            influence[l_src, h_src, l_tgt, h_tgt] = diff

    return influence



def build_head_causal_graph(influence_matrix: np.ndarray, threshold: float = 0.1) -> nx.DiGraph:
    G = nx.DiGraph()
    L, H, _, _ = influence_matrix.shape
    for l1 in range(L):
        for h1 in range(H):
            G.add_node(f"L{l1}H{h1}")
            for l2 in range(L):
                for h2 in range(H):
                    score = influence_matrix[l1, h1, l2, h2]
                    if l1 != l2 and score > threshold:
                        G.add_edge(f"L{l1}H{h1}", f"L{l2}H{h2}", weight=score)
    return G


def sweep_head_to_output_deltas(model, clean, corrupt, true_label, num_classes):
    """
    Returns a [L, H] matrix where each entry is the ΔP(true_label)
    caused by patching that attention head.
    """
    model.eval()
    device = next(model.parameters()).device
    clean_b = clean.unsqueeze(0).to(device)
    corrupt_b = corrupt.unsqueeze(0).to(device)

    # Get baseline probabilities from corrupted input
    with torch.no_grad():
        baseline_logits = model(corrupt_b)
        baseline_probs = torch.softmax(baseline_logits, dim=1)[0].cpu().numpy()
        baseline_p = baseline_probs[true_label]

    # Init matrix
    L = len(model.transformer_encoder.layers)
    H = model.transformer_encoder.layers[0].self_attn.num_heads
    delta_p = np.zeros((L, H))

    for l in range(L):
        for h in range(H):
            patched_logits = patch_attention_head(model, clean, corrupt, l, h)
            patched_probs = torch.softmax(patched_logits, dim=1)[0].detach().cpu().numpy()
            delta = patched_probs[true_label] - baseline_p
            delta_p[l, h] = delta

    return delta_p

