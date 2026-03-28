import torch
import torch.distributed as dist


class GradBucket:
    """Accumulates parameter gradients and triggers async all-reduce when full."""

    def __init__(self, num_params: int):
        self.num_params = num_params
        self._params: list[torch.nn.Parameter] = []

    def add_param(self, param: torch.nn.Parameter):
        self._params.append(param)
        if len(self._params) == self.num_params:
            result = self._all_reduce()
            self._params = []
            return result
        return None

    def _all_reduce(self):
        flat_grads = torch._utils._flatten_dense_tensors([p.grad for p in self._params])
        flat_grads /= dist.get_world_size()
        handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
        return handle, self._params, flat_grads


class DDP(torch.nn.Module):
    """Bucketed Distributed Data Parallel wrapper.

    Each rank holds a full copy of the model. Gradients are averaged across
    ranks via async all-reduce, grouped into fixed-size buckets to overlap
    communication with backward computation.

    Interface mirrors FSDP: call finish_gradient_synchronization() after
    loss.backward() and before optimizer.step().
    """

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self._pending_reductions: list[tuple] = []
        self._param_to_bucket: dict[torch.nn.Parameter, GradBucket] = {}
        self._bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self._buckets: list[GradBucket] = []
        self._init_buckets()

    def _init_buckets(self):
        """Broadcast rank 0 weights and assign parameters to gradient buckets.

        Parameters are iterated in reverse order (matching PyTorch DDP's
        convention) so that buckets fill in backward-pass order, maximising
        overlap between communication and computation.
        """
        curr_bytes = 0
        curr_params: list[torch.nn.Parameter] = []

        for param in reversed(list(self.module.parameters())):
            dist.broadcast(param.data, src=0)

            if not param.requires_grad:
                continue

            curr_params.append(param)
            curr_bytes += param.data.nbytes

            if curr_bytes >= self._bucket_size_bytes:
                bucket = GradBucket(len(curr_params))
                self._buckets.append(bucket)
                for p in curr_params:
                    self._param_to_bucket[p] = bucket
                curr_params = []
                curr_bytes = 0

            param.register_post_accumulate_grad_hook(self._grad_hook)

        if curr_params:
            bucket = GradBucket(len(curr_params))
            self._buckets.append(bucket)
            for p in curr_params:
                self._param_to_bucket[p] = bucket

    def _grad_hook(self, param: torch.nn.Parameter):
        result = self._param_to_bucket[param].add_param(param)
        if result is not None:
            self._pending_reductions.append(result)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all pending all-reduce ops and install averaged gradients."""
        for handle, params, flat_grads in self._pending_reductions:
            handle.wait()
            for param, grad in zip(
                params,
                torch._utils._unflatten_dense_tensors(flat_grads, params),
            ):
                param.grad = grad
        self._pending_reductions.clear()
