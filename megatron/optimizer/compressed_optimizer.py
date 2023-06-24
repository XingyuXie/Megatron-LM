# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron distributed optimizer."""


from apex.optimizers import FusedAdam as Adam
import math
import torch
import time

from megatron import print_rank_0
from megatron.core import mpu, tensor_parallel

from .optimizer import _zero_grad_group_helper
from .distrib_optimizer import DistributedOptimizer


class CompressedDistributedOptimizer(DistributedOptimizer):
    """Distributed optimizer, for all data types (fp16, bf16, and fp32).

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        use_contiguous_buffers_in_local_ddp: if true, the local DDP model
            is using a contiguous buffer to hold the model grads.
        fp16: if true, the model is running in fp16.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        models: list of models (i.e., the virtual pipelining models). This
            is used by the distributed optimizer for mapping parameters.
    """


    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler, models):
        """
        See top of class definition for argument descriptions.

        The steps in this method create the core mapping between DDP grad
        buffers, parameters, and parameter shard ranges, that is needed for
        converting between model param indexes and main parameter shard
        indexes. This method also updates the optimizer parameter groups
        with the newly created shards.
        """

        

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, params_dtype, grad_scaler, models)
        
        self.lowbit_scale = 65535.0
        self.lowbit_gap = 2048
        self.error_beta = 0.95
        self.ref_lr = 1.0
        self.pre_loss_scale = self.grad_scaler.scale
    
    
        
    def get_lowbit_buffer_dp_views(self):
        """
        Get shard views of each of the DDP's param/grad buffers.

        Additionally, return references to the entire buffers, for use
        in _reduce_scatter_base and _all_gather_base.
        """

        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Buffer views.
        view_items = []
        for model_index, model in enumerate(self.models):
            for dtype, mem_buffer in model._grad_buffers.items():
                buf = mem_buffer.data
                assert buf.numel() % data_parallel_world_size == 0
                err_buf = model._local_error_feedbacks.get(dtype)
                ref_buf = model._local_ref_points.get(dtype)
                shard_size = int(buf.numel() / data_parallel_world_size)
                buf_views = [buf[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)]
                # err_buf_views = [err_buf[(r*shard_size):((r+1)*shard_size)]
                #              for r in range(data_parallel_world_size)] \
                #                  if err_buf is not None else None
                ref_buf_views = [ref_buf[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)] \
                                 if ref_buf is not None else None
                view_items.append((model_index, dtype, 
                                   buf, buf_views,
                                   err_buf,
                                   ref_buf, ref_buf_views))

        return view_items

    def reduce_model_grads(self, args, timers):
        """
        Reduce-scatter model grads.

        The DDP's grad buffer is used for the reduce-scatter, and thus no
        tensors are dynamically allocated.

        Note: this is a different order of reduction, versus the non-
        distributed optimizer, which reduces: 1) layernorm grads, 2) all
        grads, 3) embedding grads.
        """
        # All-reduce layer-norm grads (for sequence parallelism).
        timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.allreduce_layernorm_grads(args)
        timers('layernorm-grads-all-reduce').stop()

        # All-reduce embedding grads.
        timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.allreduce_embedding_grads(args)
        timers('embedding-grads-all-reduce').stop()

        # Reduce-scatter setup.
        timers('grads-reduce-scatter', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()
        
        # # Scale grad buffers by '1 / data_parallel_world_size'.
        # for model in self.models:
        #     for dtype, gbuf in model._grad_buffers.items():
        #         gbuf.data /= data_parallel_world_size
                
        # Reduce-scatter all grads.
        gbuf_view_items = self.get_lowbit_buffer_dp_views()
        if self.pre_loss_scale != self.grad_scaler.scale:
            scale_change = self.grad_scaler.scale/self.pre_loss_scale
        else: scale_change = None
        for index, (model_index, dtype,
                    gbuf, gbuf_views,
                    err_buf,
                    ref_buf, ref_buf_views) \
            in enumerate(gbuf_view_items):
            use_ref_point = ref_buf is not None
            use_error_feedback =  err_buf is not None
            if not use_ref_point and not use_error_feedback:
                torch.distributed.reduce_scatter_tensor(
                    gbuf_views[data_parallel_rank],
                    gbuf.div_(data_parallel_world_size),
                    group = data_parallel_group,
                )
                timers('grads-reduce-scatter').stop()
                return
            if use_ref_point:
                if scale_change is not None:
                    ref_buf.mul_(scale_change)
                gbuf.add_(ref_buf, alpha=-self.ref_lr)
            local_var = gbuf.clone()
            if use_error_feedback:
                if scale_change is not None:
                    err_buf.mul_(scale_change) 
                local_var.add_(err_buf)
            
            
            compressed_tensor = local_var.mul_(self.lowbit_scale/data_parallel_world_size).to(torch.int8)
            
            if use_error_feedback:
                local_var.copy_(compressed_tensor).div_(-self.lowbit_scale/data_parallel_world_size)
                local_var.add_(gbuf)
                err_buf.add_(local_var, 1.-self.error_beta)
            shard_size = int(gbuf.numel() / data_parallel_world_size)
            r = data_parallel_rank - 1
            compressed_tensor_views = compressed_tensor[(r*shard_size):((r+1)*shard_size)]
            
            torch.distributed.reduce_scatter_tensor(
                compressed_tensor_views,
                compressed_tensor,
                group = data_parallel_group,
            )
            gbuf_views[data_parallel_rank].copy_(compressed_tensor_views).div_(self.lowbit_scale)
            
        
            if use_ref_point:
                gbuf.add_(ref_buf, alpha=self.ref_lr)
                ref_buf.copy_(gbuf)
                
                # update the scatter part
                ref_buf_views[data_parallel_rank].div_(1.0+self.ref_lr)
                # local_var_view = local_var[(r*shard_size):((r+1)*shard_size)]
                # torch.abs(gbuf_views[data_parallel_rank], out=local_var_view)
                # local_var_view.mul_(4).add_(self.ref_lr**2).sqrt_().add_(self.ref_lr)
                # ref_buf_views[data_parallel_rank].mul_(2.0).div_(local_var_view)
                # torch.abs(gbuf_views[data_parallel_rank], out=local_var_view)
                # ref_buf_views[data_parallel_rank].square_().mul_(local_var_view)
                
            local_var=None
            ref_buf_views=None
            compressed_tensor=None
            compressed_tensor_views=None
                
        timers('grads-reduce-scatter').stop()
        self.pre_loss_scale = self.grad_scaler.scale
        # print_rank_0('done with grad reduce. Compilation time: {:.3f} seconds; grad compression is {}'
        #       .format(time.time() - start_time, self.grad_compression))

    @torch.no_grad()
    def step(self, args, timers):
        update_flag, grad_norm, num_zeros_in_grad = super().step(self, args, timers)
        if not update_flag:
            for model_index, model in enumerate(self.models):
                for dtype, mem_buffer in model._local_error_feedbacks.items():
                    mem_buffer.zero()
                    ref_buf = model._local_ref_points.get(dtype)
                    if ref_buf is not None:
                        ref_buf.zero()
        return update_flag, grad_norm, num_zeros_in_grad