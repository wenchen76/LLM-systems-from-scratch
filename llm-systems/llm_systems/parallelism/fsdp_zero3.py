"""
FSDP with ZeRO-3 (Full Sharding Strategy)

Implements Fully Sharded Data Parallelism where parameters, gradients, and
optimizer states are all sharded across ranks.

Key features:
1. Weight Pre-fetching: Eagerly all-gathers layer N+1 while layer N computes
2. Communication-Computation Overlap: Gradient reduce-scatter overlaps with
   backward computation via separate CUDA streams
3. Gradient Bucketing: Small gradients are aggregated into buckets before
   reduce-scatter to reduce NCCL metadata overhead
4. Resource Management: Full (unsharded) parameters are discarded immediately
   after their layer's computation completes

Autograd graph and backward ordering:

  Forward graph:
    embedding → Hook_0 → layer_0 → Hook_1 → layer_1 → ... → Hook_N → layer_N → EndHook → final_norm → lm_head

  Backward order (reverse of forward):
    lm_head.bw → final_norm.bw → EndHook.bw → layer_N.bw → Hook_N.bw → ... → layer_1.bw → Hook_1.bw → layer_0.bw → Hook_0.bw → embedding.bw

  - EndHook.backward(): Re-gathers layer_N params (for upcoming layer_N.bw),
    pre-fetches layer_{N-1} params
  - Hook_i.backward() (runs AFTER layer_i.bw): Reduce-scatters layer_i grads,
    re-gathers layer_{i-1} params, pre-fetches layer_{i-2} params
  - Hook_0.backward() (runs AFTER layer_0.bw): Reduce-scatters layer_0 grads
  - Remaining params (embedding/final_norm/lm_head): reduce-scattered in
    finish_gradient_synchronization() after loss.backward() completes, since
    embedding is the first forward op and no differentiable hook can precede it
"""

import torch
import torch.distributed as dist
from torch import Tensor


class FSDPUnit:
    """Wraps a single module (e.g., one TransformerBlock) for FSDP sharding.

    Each unit manages:
    - Sharded parameter storage (1/world_size of each param)
    - All-gather for reconstructing full params before compute
    - Reduce-scatter for distributing gradients after backward
    """

    def __init__(self, module: torch.nn.Module, rank: int, world_size: int,
                 process_group=None):
        self.module = module
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.training = True

        # Original parameter metadata for reshaping
        self.param_names: list[str] = []
        self.param_shapes: list[torch.Size] = []
        self.param_numels: list[int] = []

        # Flat sharded storage
        self.flat_shard: Tensor | None = None  # 1/world_size of all params
        self.flat_full: Tensor | None = None   # full gathered params (transient)

        # Communication handles
        self.gather_handle = None
        self.scatter_handle = None

        # Gradient bucket
        self.flat_grad: Tensor | None = None

        self._shard_parameters()

    def _shard_parameters(self):
        """Flatten all params into one contiguous buffer, keep only local shard."""
        params = list(self.module.parameters())
        if not params:
            return

        for name, param in self.module.named_parameters():
            self.param_names.append(name)
            self.param_shapes.append(param.shape)
            self.param_numels.append(param.numel())

        # Flatten all parameters into a single contiguous tensor
        flat_full = torch.cat([p.data.detach().flatten() for p in params])

        # Pad to be divisible by world_size
        total_numel = flat_full.numel()
        padded_numel = ((total_numel + self.world_size - 1) // self.world_size) * self.world_size
        if padded_numel > total_numel:
            flat_full = torch.cat([flat_full, flat_full.new_zeros(padded_numel - total_numel)])

        self.padded_numel = padded_numel
        self.original_numel = total_numel
        self.shard_numel = padded_numel // self.world_size

        # Each rank keeps only its shard
        self.flat_shard = flat_full[self.rank * self.shard_numel:
                                    (self.rank + 1) * self.shard_numel].clone()
        self.flat_shard = torch.nn.Parameter(self.flat_shard, requires_grad=True)

        # Replace module parameters with views into a placeholder
        # (they will be overwritten during all-gather before compute)
        for param in params:
            param.requires_grad_(False)
            param.data = torch.empty(0, device=param.device, dtype=param.dtype)

    def all_gather_params(self, async_op=True):
        """Reconstruct full parameters from all shards via all-gather."""
        if self.flat_shard is None:
            return

        # Allocate buffer for full (unsharded) parameters
        self.flat_full = torch.empty(
            self.padded_numel, device=self.flat_shard.device,
            dtype=self.flat_shard.dtype
        )

        # Create list of shard-sized views into flat_full for all_gather
        shard_list = list(self.flat_full.chunk(self.world_size))

        self.gather_handle = dist.all_gather(
            shard_list, self.flat_shard.data,
            group=self.process_group, async_op=async_op
        )

        if not async_op:
            self._install_full_params()

    def wait_gather(self):
        """Wait for the all-gather to complete and install full params."""
        if self.gather_handle is not None:
            self.gather_handle.wait()
            self.gather_handle = None
        self._install_full_params()

    def _install_full_params(self):
        """Set module parameters to views of the gathered full tensor."""
        if self.flat_full is None:
            return

        full_data = self.flat_full[:self.original_numel]

        offset = 0
        self.param_views = []
        for (name, param), shape, numel in zip(
            self.module.named_parameters(), self.param_shapes, self.param_numels
        ):
            view = full_data[offset:offset + numel].view(shape)
            param.data = view
            if self.training:
                param.requires_grad_(True)
            self.param_views.append(view)
            offset += numel

        self.full_data_with_grad = full_data

    def discard_full_params(self):
        """Free the unsharded parameter buffer to save memory."""
        self.flat_full = None
        self.full_data_with_grad = None

    def reduce_scatter_grads(self, comm_stream: torch.cuda.Stream, async_op=True):
        """Reduce-scatter gradients: each rank gets gradient for its shard only."""
        if self.full_data_with_grad is None:
            return

        # Collect the gradient from the full param buffer
        if self.full_data_with_grad.grad is not None:
            full_grad = self.full_data_with_grad.grad[:self.original_numel]
        else:
            # Gradients might be on individual param views
            grads = []
            for param in self.module.parameters():
                if param.grad is not None:
                    grads.append(param.grad.flatten())
                else:
                    grads.append(torch.zeros(param.numel(), device=param.device,
                                             dtype=param.dtype))
            full_grad = torch.cat(grads)

        # Pad gradient to match padded_numel
        if full_grad.numel() < self.padded_numel:
            full_grad = torch.cat([
                full_grad,
                full_grad.new_zeros(self.padded_numel - full_grad.numel())
            ])

        # Average gradients across ranks
        full_grad = full_grad / self.world_size

        # Record event AFTER full_grad is ready on compute stream
        ready_event = torch.cuda.current_stream().record_event()

        # Allocate buffer for reduced shard gradient
        self.flat_grad = torch.empty(
            self.shard_numel, device=self.flat_shard.device,
            dtype=self.flat_shard.dtype
        )

        # Split full gradient into per-rank chunks for reduce_scatter
        input_list = list(full_grad.chunk(self.world_size))

        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(ready_event)
            self.scatter_handle = dist.reduce_scatter(
                self.flat_grad, input_list,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
                async_op=async_op
            )

    def wait_scatter(self):
        """Wait for reduce-scatter and apply gradient to shard."""
        if self.scatter_handle is not None:
            self.scatter_handle.wait()
            self.scatter_handle = None

        if self.flat_grad is not None and self.flat_shard.grad is None:
            self.flat_shard.grad = self.flat_grad.clone()
        elif self.flat_grad is not None:
            self.flat_shard.grad.copy_(self.flat_grad)

        # Free memory
        self.flat_full = None
        self.full_data_with_grad = None
        self.flat_grad = None


class FSDP(torch.nn.Module):
    """Fully Sharded Data Parallelism wrapper with ZeRO-3.

    Wraps a BasicsTransformerLM (or similar model with a `.layers` ModuleList)
    and shards parameters, gradients, and optimizer states across ranks.

    Usage:
        model = BasicsTransformerLM(...).to(device)
        fsdp_model = FSDP(model)
        optimizer = AdamW(fsdp_model.parameters(), lr=1e-3)

        for data in dataloader:
            optimizer.zero_grad()
            output = fsdp_model(data)
            loss = output.mean()
            loss.backward()
            fsdp_model.finish_gradient_synchronization()
            optimizer.step()
    """

    def __init__(self, module: torch.nn.Module, process_group=None):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.process_group = process_group

        # Communication stream for overlap
        self.comm_stream = torch.cuda.Stream()

        # Identify FSDP-wrappable units: each transformer layer is one unit,
        # plus one unit for non-layer params (embeddings, final norm, lm_head)
        self.fsdp_units: list[FSDPUnit] = []
        self.unit_modules: list[torch.nn.Module] = []

        self._wrap_model(module)
        self.module = module

    def _wrap_model(self, module: torch.nn.Module):
        """Create FSDPUnit for each transformer layer + one for remaining params."""
        # Wrap each transformer layer as its own FSDP unit
        if hasattr(module, 'layers'):
            for layer in module.layers:
                unit = FSDPUnit(layer, self.rank, self.world_size,
                                self.process_group)
                self.fsdp_units.append(unit)
                self.unit_modules.append(layer)

        # Wrap remaining (non-layer) parameters as a single unit
        # This includes: token_embeddings, positional_encoder, final_norm, lm_head
        remaining = _RemainingParams(module)
        if list(remaining.parameters()):
            unit = FSDPUnit(remaining, self.rank, self.world_size,
                            self.process_group)
            self.fsdp_units.append(unit)
            self.unit_modules.append(remaining)

    def parameters(self):
        """Yield only the sharded parameters (for the optimizer)."""
        for unit in self.fsdp_units:
            if unit.flat_shard is not None:
                yield unit.flat_shard

    def train(self, mode=True):
        self.module.train(mode)
        for unit in self.fsdp_units:
            unit.training = mode
        return self

    def eval(self):
        return self.train(False)

    def forward(self, *inputs, **kwargs):
        """Forward pass with eager pre-fetching of next layer's params.

        Builds the autograd graph as:
          embedding → Hook_0 → layer_0 → ... → Hook_N → layer_N → EndHook → final_norm → lm_head

        So backward runs as:
          lm_head.bw → final_norm.bw → EndHook.bw → layer_N.bw → Hook_N.bw → ... → layer_0.bw → Hook_0.bw → embedding.bw

        - EndHook.bw: re-gathers layer_N params, pre-fetches layer_{N-1}
        - Hook_i.bw (after layer_i.bw): reduce-scatters layer_i grads,
          re-gathers layer_{i-1} params, pre-fetches layer_{i-2}
        - Remaining params (embedding/final_norm/lm_head) are reduce-scattered
          in finish_gradient_synchronization() after backward completes
        """
        remaining_unit = self.fsdp_units[-1] if len(self.fsdp_units) > len(self.module.layers) else None
        layer_units = self.fsdp_units[:len(self.module.layers)] if hasattr(self.module, 'layers') else []
        num_layers = len(layer_units)

        # --- Gather remaining (embedding/lm_head) params ---
        if remaining_unit is not None:
            remaining_unit.all_gather_params(async_op=False)

        # --- Run embedding ---
        _, sequence_length = inputs[0].size()
        x = self.module.token_embeddings(inputs[0])

        # --- Gather layer 0 (sync, need it now) ---
        layer_units[0].all_gather_params(async_op=False)

        # --- Pre-fetch layer 1 (async, while layer 0 computes) ---
        if num_layers > 1:
            layer_units[1].all_gather_params(async_op=True)

        # --- Process each transformer layer ---
        for i in range(num_layers):
            unit = layer_units[i]
            layer = self.module.layers[i]

            # Insert Hook_i BEFORE layer_i.
            # In backward, Hook_i.bw runs AFTER layer_i.bw, so:
            #   - layer_i's gradients are already computed → can reduce-scatter
            #   - layer_{i-1} needs params for its upcoming backward → re-gather
            if self.training:
                prev_unit = layer_units[i - 1] if i > 0 else None
                prefetch_unit = layer_units[i - 2] if i > 1 else None
                x = _PostLayerBackwardHook.apply(
                    x, unit, self.comm_stream, prev_unit, prefetch_unit
                )

            # Wait for this layer's all-gather to finish
            unit.wait_gather()

            # Pre-fetch layer i+2 (layer i+1 was already pre-fetched last iteration)
            if i + 2 < num_layers:
                layer_units[i + 2].all_gather_params(async_op=True)

            # Compute layer i forward
            x = layer(x)

            # Discard full params — activations are saved by autograd,
            # full params will be re-gathered during backward by the hooks
            unit.discard_full_params()

        # --- Insert EndHook AFTER last layer ---
        # In backward, EndHook.bw runs BEFORE layer_N.bw, so:
        #   - Re-gathers layer_N's params (needed for layer_N.bw)
        #   - Pre-fetches layer_{N-1}'s params
        if self.training and num_layers > 0:
            prefetch_unit = layer_units[-2] if num_layers > 1 else None
            x = _PreLayerBackwardHook.apply(
                x, layer_units[-1], self.comm_stream, prefetch_unit
            )

        # --- Final norm + lm_head ---
        x = self.module.final_norm(x)
        output = self.module.lm_head(x)

        # Inference: discard remaining unit
        if remaining_unit is not None and not self.training:
            remaining_unit.discard_full_params()

        return output

    def finish_gradient_synchronization(self):
        """Wait for all pending reduce-scatter operations to complete.

        Also handles reduce-scatter for the remaining unit (embedding/final_norm/lm_head).
        We do this here rather than in an autograd hook because embedding is the
        first operation in forward (so last in backward), and we can't place a
        differentiable hook before it — inputs[0] is integer token IDs with no
        gradient flow. By the time loss.backward() returns, all grads are computed.
        """
        # Reduce-scatter remaining unit grads (embedding/final_norm/lm_head)
        remaining_unit = self.fsdp_units[-1] if len(self.fsdp_units) > len(self.module.layers) else None
        if remaining_unit is not None:
            remaining_unit.reduce_scatter_grads(self.comm_stream, async_op=True)
            remaining_unit.discard_full_params()

        # Wait for all reduce-scatter ops (layer units + remaining)
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        for unit in self.fsdp_units:
            unit.wait_scatter()

    def zero_grad(self):
        for unit in self.fsdp_units:
            if unit.flat_shard is not None and unit.flat_shard.grad is not None:
                unit.flat_shard.grad.zero_()
            for param in unit.module.parameters():
                param.grad = None


class _PostLayerBackwardHook(torch.autograd.Function):
    """Placed BEFORE layer_i in the forward graph.

    In backward, this runs AFTER layer_i.backward() completes, because
    backward reverses the forward order:
        forward:  Hook_i → layer_i
        backward: layer_i.bw → Hook_i.bw

    When Hook_i.backward() executes:
    1. layer_i's gradients have been computed → reduce-scatter them
    2. layer_{i-1}.backward() is about to run → re-gather its params
    3. layer_{i-2}.backward() runs after that → pre-fetch its params
    """

    @staticmethod
    def forward(ctx, x, unit: FSDPUnit, comm_stream: torch.cuda.Stream,
                prev_unit: FSDPUnit | None, prefetch_unit: FSDPUnit | None):
        ctx.unit = unit
        ctx.comm_stream = comm_stream
        ctx.prev_unit = prev_unit
        ctx.prefetch_unit = prefetch_unit
        return x

    @staticmethod
    def backward(ctx, grad_output):
        unit = ctx.unit
        comm_stream = ctx.comm_stream

        # --- Step 1: Reduce-scatter layer_i's gradients ---
        # layer_i.backward() just finished, so gradients exist now.
        # Stream synchronization is handled inside reduce_scatter_grads.
        unit.reduce_scatter_grads(comm_stream, async_op=True)

        # Discard full params (no longer needed)
        unit.discard_full_params()

        # --- Step 2: Ensure layer_{i-1}'s params are ready ---
        # layer_{i-1}.backward() is about to run and needs full params.
        # If a pre-fetch was already started (by the previous hook's step 3),
        # just wait for it. Otherwise, do a fresh synchronous gather.
        if ctx.prev_unit is not None:
            if ctx.prev_unit.gather_handle is not None:
                ctx.prev_unit.wait_gather()
            else:
                ctx.prev_unit.all_gather_params(async_op=False)

        # --- Step 3: Pre-fetch layer_{i-2}'s params ---
        # layer_{i-2}.backward() will run after layer_{i-1}, start gathering early.
        if ctx.prefetch_unit is not None:
            ctx.prefetch_unit.all_gather_params(async_op=True)

        return grad_output, None, None, None, None


class _PreLayerBackwardHook(torch.autograd.Function):
    """Placed AFTER the last layer in the forward graph.

    In backward, this runs BEFORE the last layer's backward:
        forward:  layer_N → EndHook
        backward: EndHook.bw → layer_N.bw

    When EndHook.backward() executes:
    1. layer_N.backward() is about to run → re-gather its params
    2. layer_{N-1}.backward() runs after that → pre-fetch its params
    """

    @staticmethod
    def forward(ctx, x, unit: FSDPUnit, comm_stream: torch.cuda.Stream,
                prefetch_unit: FSDPUnit | None):
        ctx.unit = unit
        ctx.comm_stream = comm_stream
        ctx.prefetch_unit = prefetch_unit
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Re-gather last layer's params for its upcoming backward
        ctx.unit.all_gather_params(async_op=False)

        # Pre-fetch second-to-last layer's params
        if ctx.prefetch_unit is not None:
            ctx.prefetch_unit.all_gather_params(async_op=True)

        return grad_output, None, None, None


class _RemainingParams(torch.nn.Module):
    """Helper to wrap non-layer parameters (embeddings, final norm, lm_head)
    as a single FSDP unit."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._remaining_modules = torch.nn.ModuleDict()

        for name, child in model.named_children():
            if name != 'layers':
                self._remaining_modules[name] = child

    def named_parameters(self, *args, **kwargs):
        return self._remaining_modules.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._remaining_modules.parameters(*args, **kwargs)
