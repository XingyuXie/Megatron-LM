from collections import defaultdict
import logging
import math
from typing import Dict

import torch
import torch.distributed as dist

from torch.distributed.algorithms.ddp_comm_hooks import default
from torch.distributed import distributed_c10d
import math
from megatron import print_rank_0

__all__ = [
     "lowbit_hook",
     "LowbitState",
     "fp32_compress_hook"
]

logger = logging.getLogger(__name__)

def fp32_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.buffer().clone().detach().div_(world_size)

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor
    
    return fut.then(decompress)

class LowbitState:

    __slots__ = [
        "process_group",
        # The fields below are the hyperparameters that often need to be tuned by the user.,
        "start_lowbit_iter",
        "error_beta",
        "ref_lr",
        "grad_scale",
        # The fields below are the binary hyperparameters recommended to be turned on for performance and accuracy.
        "use_error_feedback",
        "use_ref_point",
        # The fields below are internal state.
        "error_dict",
        "refpoint_dict",
        "iter",
        "scale_step",
        "scale_iter",
        "compressed_info",
        "compressed_flag"
    ]

    def __init__(
        self,
        process_group,
        start_lowbit_iter=20,
        use_error_feedback=True,
        error_beta=0.95,
        use_ref_point=True,
        ref_lr=1.0,
        grad_scale= 262144.0,
        scale_step = 2048
    ):
        logger.info(
            "Lowbit Optimizaer config: start_powerSGD_iter = %s; "
            "use_error_feedback = %s;  use_ref_point = %s;  grad_scale=%s",
            start_lowbit_iter,
            use_error_feedback,
            use_ref_point,
            grad_scale
        )

        self.process_group = process_group
        # Deferring compression util step 'start_powerSGD_iter' can have one advantage:
        # 1) There is an internal optimization of rebuilding buckets process in DDP,
        # in order to save the memory space.
        # This step takes place after the first iteration.
        # However, this means that the shape of input bucketized tensors is subject to change,
        # which will complicate the implementations of error feedback and warm-up.
        # Running vanilla allreduce in the first few iterations can avoid this complexity.
        if (use_error_feedback) and start_lowbit_iter <= 1:
            raise ValueError(
                "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                "because PowerSGD can only be applied after the first two iterations in DDP."
            )
        self.start_lowbit_iter = start_lowbit_iter
        # Error feedback is usually crucial for both for convergence and generalization,
        # because lowbit compression is a biased compressor,
        # i.e., compressing and decompressing a random gradient does not yield the original in expectation.
        self.use_error_feedback = use_error_feedback
        self.error_beta = error_beta
        self.use_ref_point = use_ref_point
        self.ref_lr = ref_lr
        # Since there is only a single state instance for all the input buckets,
        # need to maintain a dictionary that maps each bucket index to the local error.
        self.error_dict: Dict[int, torch.Tensor] = {}
        self.refpoint_dict: Dict[int, torch.Tensor] = {}
        # Iteration/step in the training loop.
        self.iter = 0
        self.scale_iter = -1
        self.grad_scale = grad_scale
        self.scale_step = scale_step
        self.compressed_info = torch.iinfo(torch.int8)
        self.compressed_flag = torch.cuda.FloatTensor([0.0])

    def __getstate__(self):
        """
        Returns a ``Dict[str, Any]`` which will be pickled and saved.
        ``process_group`` is not serializable and excluded from
        a returned state.
        """
        logger.warning(
            "NOTE: Process group is not serializable and excluded from a saved state."
        )
        return {
            slot: getattr(self, slot)
            for slot in self.__slots__ if slot != "process_group"
        }

    def __setstate__(self, state):
        r"""
        Takes a provided ``state`` and retrieves ``LowbitState``.
        ``process_group`` is set to default.
        """
        self.process_group = distributed_c10d._get_default_group()
        logger.warning(
            "NOTE: Process group will be set to a default group (i.e. the world size).\
                If a different group is desired, please set `self.process_group` after PowerSGD state is loaded."
        )
        for slot, value in state.items():
            setattr(self, slot, value)

    def maybe_increase_iter(self, bucket):
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_last():
            self.iter += 1
            # self.scale_iter+=1
            # # bool_value = torch.tensor(self.compressed_flag).int()
            # dist.all_reduce(self.compressed_flag, op=torch.distributed.ReduceOp.MAX)
            # if self.compressed_flag.item() > 0:
            #     self.grad_scale *= 0.5
            #     self.grad_scale = max(self.grad_scale, 1e-8)
            #     self.scale_iter = 0
                
            # elif self.scale_iter > self.scale_step:
            #     self.grad_scale *= 2.0
            #     self.grad_scale = min(self.grad_scale, 2048)
            #     self.scale_iter = 0
            # self.compressed_flag.fill_(0.0)

        if self.iter == self.start_lowbit_iter:
            logger.info(
                "Start to apply Low-bit Opt after %s iterations.", self.iter
            )




def lowbitopt_hook(
    state: LowbitState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    Args:
        state (Optimizer.state): State information to configure the compression, mainly used for error update.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1,
                                  start_powerSGD_iter=10, min_compression_rate=0.5)
        >>> ddp_model.register_comm_hook(state, powerSGD_hook)
    """  # noqa: B950
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run vanilla allreduce in the first `start_powerSGD_iter` iterations.
    if state.iter < state.start_lowbit_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply Lowbit Optimization after `start_lowbit_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.index()
    total_length = input_tensor.shape[0]
    
    
    if state.use_ref_point:
        if bucket_index in state.refpoint_dict:
            input_tensor.add_(state.refpoint_dict[bucket_index], alpha=-1.0)
        else:
            logger.info(
                "A zero tensor of length %s that represents global refrence point is created.",
                total_length
            )
            state.refpoint_dict[bucket_index] = torch.zeros(
                total_length, device=device, dtype=dtype
            )
    
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
            # assert not torch.isnan(input_tensor).any(), 'input tensor should not contain nan after error'
        else:
            logger.info(
                "A zero tensor of length %s that represents local error is created.",
                total_length
            )
            state.error_dict[bucket_index] = torch.zeros(
                total_length, device=device, dtype=dtype
            )
    if state.use_ref_point or state.use_error_feedback:
        local_var = input_tensor.abs()
        
    #  
    compressed_tensor = input_tensor.mul_(state.grad_scale/world_size).to(torch.int8)
    # if torch.isinf(compressed_tensor).any():
    #     print("input tensor range is {} --- {}".format(torch.min(input_tensor),torch.max(input_tensor)))
    #     print("scale_factor is {}".format(state.grad_scale))
    # assert not torch.isinf(compressed_tensor).any(), 'local grad should not contain inf'
    if state.use_error_feedback:
        local_var.copy_(compressed_tensor)
        input_tensor.add_(local_var,alpha=-1.0).div_(state.grad_scale/world_size)
        state.error_dict[bucket_index].lerp_(input_tensor, 1.-state.error_beta)
        # implememt err*state.error_beta + (1-error_beta)*(input_tensor-compressed_tensor)/scale
        # state.error_dict[bucket_index].mul_(state.error_beta)
        # state.error_dict[bucket_index].add_(input_tensor, alpha=(1.-state.error_beta)/state.grad_scale)
        # input_tensor.copy_(compressed_tensor)
        # state.error_dict[bucket_index].add_(input_tensor, alpha= -(1.-state.error_beta)/state.grad_scale)
        # assert not torch.isnan(state.error_dict[bucket_index]).any(), 'Error Feedback should not contain nan'
    
    
    # compressed_tensor.div_(world_size)
    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        grads = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        grads.copy_(fut.value()[0]).div_(state.grad_scale)
        
        
        # 检测是否存在inf或者-inf
        # print_rank_0("max value {}; min value {};".
        #              format(torch.max(grads).item(),torch.min(grads).item())
        # )
        nan_flag = torch.isinf(grads).any()
        if nan_flag:
            state.compressed_flag+=1
            if state.use_error_feedback:
                state.error_dict[bucket_index].zero_()
            # grads.clamp_(min=state.compressed_info.min,
            #             max=state.compressed_info.max)
        #     print_rank_0("grad overflow:  max value {}; min value {};".
        #              format(torch.max(grads).item(),torch.min(grads).item())
        # )
            
        # grads.div_(state.grad_scale)


        # assert not torch.isinf(grads).any(), 'grad after all reduce contain INF'
        if state.use_ref_point:
            if not nan_flag:
                grads.add_(state.refpoint_dict[bucket_index], alpha=state.ref_lr)
                # h_g_sgn = grads.sign()
                state.refpoint_dict[bucket_index].copy_(grads)
                # state.refpoint_dict[bucket_index].div_(1.0+state.ref_lr)
                
                # assert not torch.isnan(state.refpoint_dict[bucket_index]).any(), 'Ref Point 0 should not contain nan'
                torch.abs(grads, out=local_var)
                local_var.mul_(4).add_(state.ref_lr**2).sqrt_().add_(state.ref_lr)
                # assert not torch.isnan(local_var).any(), 'local_var should not contain nan'
                state.refpoint_dict[bucket_index].mul_(2.0).div_(local_var)
                # if torch.isnan(state.refpoint_dict[bucket_index]).any():
                #     print("local var range is {} --- {}".format(torch.min(local_var),torch.max(local_var)))
                # assert not torch.isnan(state.refpoint_dict[bucket_index]).any(), 'Ref Point 1 should not contain nan'
                torch.abs(grads, out=local_var)
                state.refpoint_dict[bucket_index].square_().mul_(local_var)
                # if torch.max(local_var) * state.grad_scale * 1.1 >= state.compressed_info.max:
                #     state.compressed_flag=True
                # assert not torch.isinf(state.refpoint_dict[bucket_index]).any(), 'Ref Point 2 should not contain inf'
            else: state.refpoint_dict[bucket_index].zero_()

        
        state.maybe_increase_iter(bucket)
        # safe_scale_factor(local_var, state)
        # print("Finsh our step {}".format(state.iter)    )
        return grads

    return fut.then(decompress)


def safe_scale_factor(state: LowbitState):
    # assert that input tensor does not contain INF
    # assert not torch.isinf(tensor).any(), 'gradient tensor should not contain INF'
    # tensor_max = torch.max(tensor.abs()) + eps

    # keep halving state.grad_scale until tensor_max * state.grad_scale < max_float16
    # while tensor_max * state.grad_scale >= max_float16:
    if state.compressed_flag:
        state.grad_scale *= 0.5
        if state.grad_scale < 1e-4: 
            state.grad_scale = 1e-4
            # break
        state.scale_iter = 0
        
    elif state.scale_iter > state.scale_step:
        state.grad_scale *= 2.0
        state.scale_iter = 0

        # state.grad_scale = pow(2, round(math.log2(state.grad_scale)))
        

