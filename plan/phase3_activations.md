# Phase 3 — Activation Extraction

Forward-hook utilities, timestep routing, and the activation buffer that feeds Phase 4 SAE training. See [README](README.md) for navigation.

## Hook utilities

`src/diffmechint/hooks/activation_taps.py`:

```python
class ResidualStreamTap:
    """Attaches forward hooks to a chosen subset of SiTBlocks and
    streams (B, T, D) activations into an ActivationBuffer."""
    
    def __init__(self, model, block_indices: list[int], buffer: ActivationBuffer):
        self.handles = [
            model.blocks[i].register_forward_hook(self._capture(i))
            for i in block_indices
        ]
    
    def _capture(self, idx):
        def hook(module, input, output):
            self.buffer.write(layer=idx, t=current_t(), x=output.detach())
        return hook
    
    def detach(self): for h in self.handles: h.remove()
```

The `current_t()` helper reads the timestep from a `ContextVar` that the
training/inference loop sets before each model call — avoids threading a
`t` argument through every hook.

## Timestep routing

`src/diffmechint/hooks/timestep_router.py`:

The Revelio grid uses three discrete timesteps `t ∈ {25, 200, 500}` (out
of 1000). The router bins continuous `t ∈ [0, 1]` into these targets when
sampling activations for analysis. For SAE training, sample uniformly in
`t ∈ [0, 1]` *or* stratify into the three bins — both supported via a
`stratify: uniform | revelio` config flag.

## Activation buffer

`src/diffmechint/hooks/activation_buffer.py`:

In-memory ring buffer plus optional shard-to-disk. Default capacity 1M
tokens × `D_max`. When full, flushes to
`$FAST/diffmechint/activations/<run_id>/<ckpt>/<layer>_<t_bin>.h5`.

## Where to tap

For SiT-B (12 blocks), tap at depths `{25%, 50%, 75%}` ⇒ blocks
`{3, 6, 9}` for SAE. For SiT-L (24) ⇒ `{6, 12, 18}`. For SiT-XL (28) ⇒
`{7, 14, 21}`. Reads the Revelio convention. Configurable via
`probe.layers: [3, 6, 9]`.

## Acceptance

Smoke: run inference with hooks attached on a 4-image batch; verify
`activation_buffer` contains `3 layers × 3 timesteps × 4 batch = 36`
records of shape `(T, D)` with the expected `D = 768/1024/1152` for
B/L/XL.
