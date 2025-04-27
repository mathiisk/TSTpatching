from typing import Callable, List, Dict, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import torch

def register_hooks(
    model: torch.nn.Module,
    layers: List[str],
    hook_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]
) -> Dict[str, torch.utils.hooks.RemovableHandle]:
    """
    Given a model and a list of module name substrings (e.g. ['encoder.layers.2.mlp']),
    register the same hook_fn on each matching submodule.
    Returns a dict mapping layer name to its handle, so you can remove them later.
    """
    handles = {}
    for name, module in model.named_modules():
        if any(substr in name for substr in layers):
            handles[name] = module.register_forward_hook(hook_fn)
    return handles


def remove_hooks(handles: Dict[str, torch.utils.hooks.RemovableHandle]) -> None:
    """Convenience to remove all hooks at once."""
    for h in handles.values():
        h.remove()


def run_and_cache(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    hook_layers: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Runs the model on `prompt` while caching activations at `hook_layers`.
    Returns a dict mapping layer names to their activations.
    """
    cache = {}
    handles = register_hooks(model, hook_layers,
                             lambda m, i, o: cache.setdefault(m, o.detach().clone()))
    _ = model(prompt)
    remove_hooks(handles)
    return cache


def patch_activations(
    model: torch.nn.Module,
    base_prompt: torch.Tensor,
    cache: Dict[str, torch.Tensor],
    hook_layers: List[str]
) -> torch.Tensor:
    """
    Runs the model on `base_prompt`, but at each `hook_layer` overwrites the internal
    activations with those from `cache`. Returns the final logits.
    """
    outputs = {}
    # Register a hook that replaces the real activation with the cached one:
    def patch_fn(module, inp, out):
        return cache[module]
    handles = register_hooks(model, hook_layers, patch_fn)
    logits = model(base_prompt)
    remove_hooks(handles)
    return logits


def compute_metrics(
    logits_clean: torch.Tensor,
    logits_patched: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, float]:
    """
    Returns a dict with:
      - 'logit_diff': (logit_clean - logit_patched).mean().item()
      - 'acc_drop': clean_acc - patched_acc
      - 'topk_change': ...
    """
    probs_clean = torch.softmax(logits_clean, dim=-1)
    probs_patch = torch.softmax(logits_patched, dim=-1)

    clean_preds = probs_clean.argmax(dim=-1)
    patch_preds = probs_patch.argmax(dim=-1)

    clean_acc = (clean_preds == labels).float().mean().item()
    patch_acc = (patch_preds == labels).float().mean().item()

    logit_diff = (logits_clean - logits_patched).mean().item()
    return {
        'logit_diff': logit_diff,
        'acc_drop': clean_acc - patch_acc,
        'clean_acc': clean_acc,
        'patched_acc': patch_acc
    }


def plot_patching_results(results: List[Dict[str, float]], metric: str = 'acc_drop'):
    layers = [r['layer'] for r in results]
    values = [r[metric] for r in results]
    plt.figure()
    plt.plot(layers, values, marker='o')
    plt.xlabel('Layer or Component')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Patching {metric.replace("_", " ")} by Layer')
    plt.show()


