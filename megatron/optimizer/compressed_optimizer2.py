# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron distributed optimizer."""


from apex.optimizers import FusedAdam as Adam
import math
import torch
import time

from megatron import print_rank_0
from megatron.core import mpu, tensor_parallel

from ..tree import tree_reduce_scatter

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
                lowbit_grad_buf = model._lowbit_grad_buffers.get(dtype)
                # ref_buf = model._local_ref_points.get(dtype)
                
                shard_size = int(buf.numel() / data_parallel_world_size)
                buf_views = [buf[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)]
                
                if lowbit_grad_buf is not None:
                    # torch.cuda.current_stream().wait_stream(model._hook_stream)
                    # ref_buf = ref_buf.data
                    lowbit_grad_buf = lowbit_grad_buf.data
                    # ref_buf_views = [ref_buf[(r*shard_size):((r+1)*shard_size)]
                    #          for r in range(data_parallel_world_size)]
                    lowbit_grad_buf_views = [lowbit_grad_buf[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)]
                    view_items.append((model_index, dtype, 
                                       buf, buf_views,
                                    #    ref_buf,ref_buf_views,
                                       lowbit_grad_buf,lowbit_grad_buf_views,
                                       model.lowbit_scale, model.ref_lr))
                else:
                    view_items.append((model_index, dtype, 
                                    buf, buf_views,
                                    # None, None,
                                    None, None,
                                    None, None))
        return view_items


    @torch.no_grad()
    def reduce_model_grads(self, args, timers):
        """
        Reduce-scatter model grads.

        The DDP's grad buffer is used for the reduce-scatter, and thus no
        tensors are dynamically allocated.

        Note: this is a different order of reduction, versus the non-
        distributed optimizer, which reduces: 1) layernorm grads, 2) all
        grads, 3) embedding grads.
        """
         # Reduce-scatter setup.
        timers('grads-reduce-scatter', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        # torch.cuda.nvtx.range_push("grads-reduce-scatter")
        # data_parallel_rank = mpu.get_data_parallel_rank()
        # data_parallel_world_size = mpu.get_data_parallel_world_size()
        # data_parallel_group = mpu.get_data_parallel_group()

        data_parallel_rank = mpu.get_data_parallel_rank_tree()
        data_parallel_world_size = mpu.get_data_parallel_world_size_tree()
        data_parallel_group = mpu.get_data_parallel_group_tree()


        # # Scale grad buffers by '1 / data_parallel_world_size'.
        # for model in self.models:
        #     for dtype, gbuf in model._grad_buffers.items():
        #         gbuf.data /= data_parallel_world_size
               
        # Reduce-scatter all grads.
        gbuf_view_items = self.get_lowbit_buffer_dp_views()
        for index, (model_index, dtype,
                    gbuf, gbuf_views,
                    # ref_buf, ref_buf_views,
                    lowbit_grad_buf, lowbit_grad_buf_views,
                    lowbit_scale, ref_lr) \
            in enumerate(gbuf_view_items):
            if lowbit_grad_buf is None:
                # torch.distributed.reduce_scatter_tensor(
                #     gbuf_views[data_parallel_rank],
                #     gbuf.div_(data_parallel_world_size),
                #     group = data_parallel_group,
                # )
                tree_reduce_scatter(
                    gbuf.div_(data_parallel_world_size),
                    group = data_parallel_group,
                )
                continue
            # shard_size = int(gbuf.numel() / data_parallel_world_size)
            # local_ref = ref_buf.to(torch.cuda.current_device(), non_blocking=True)
            compressed_tensor_views = lowbit_grad_buf_views[data_parallel_rank]
            
            # torch.distributed.reduce_scatter_tensor(
            #     compressed_tensor_views,
            #     lowbit_grad_buf,
            #     group = data_parallel_group,
            # )
            tree_reduce_scatter(
                lowbit_grad_buf,
                group = data_parallel_group,
            )
            gbuf_views[data_parallel_rank-1].\
                copy_(compressed_tensor_views).div_(lowbit_scale)
            gbuf_views[data_parallel_rank].\
                add_(gbuf_views[data_parallel_rank-1])

            # update refer point
            # print_rank_0('now {}; ref dtype {}; gbuf dtype {}'.
            #              format(dtype, ref_buf.dtype, gbuf.dtype))
            # local_ref.add_(gbuf)
            # local_view = local_ref[(data_parallel_rank*shard_size):\
            #     ((data_parallel_rank+1)*shard_size)]
            # gbuf_views[data_parallel_rank].copy_(local_view)
            # ref_buf.copy_(local_ref.to("cpu", non_blocking=True))

            # # update the scatter part
            # ref_buf_views[data_parallel_rank].div_(1.0+ref_lr)
            
            # local_var_view = local_var[(r*shard_size):((r+1)*shard_size)]
            # torch.abs(gbuf_views[data_parallel_rank], out=local_var_view)
            # local_var_view.mul_(4).add_(self.ref_lr**2).sqrt_().add_(self.ref_lr)
            # ref_buf_views[data_parallel_rank].mul_(2.0).div_(local_var_view)
            # torch.abs(gbuf_views[data_parallel_rank], out=local_var_view)
            # ref_buf_views[data_parallel_rank].square_().mul_(local_var_view)

        # for model_index, model in enumerate(self.models):
        #     torch._foreach_add_(model.main_grad_list, model.ref_point_list)
        #     torch._foreach_zero_(model.ref_point_list)
        #     torch._foreach_add_(model.ref_point_list, model.main_grad_list)
        #torch.cuda.nvtx.range_pop()
        timers('grads-reduce-scatter').stop()
        
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
        
        # print_rank_0('done with grad reduce. Compilation time: {:.3f} seconds; grad compression is {}'
        #       .format(time.time() - start_time, self.grad_compression))

    @torch.no_grad()
    def step(self, args, timers):
        update_flag, grad_norm, num_zeros_in_grad = super().step(args, timers)
        if self.pre_loss_scale != self.grad_scaler.scale:
            scale_change = self.grad_scaler.scale/self.pre_loss_scale
            for model_index, model in enumerate(self.models):
                for dtype, err_buf in model._local_error_feedbacks.items():
                    err_buf.data.mul_(scale_change)
                    ref_buf = model._local_ref_points.get(dtype)
                    ref_buf.data.mul_(scale_change) 
        if not update_flag:
            for model_index, model in enumerate(self.models):
                for dtype, mem_buffer in model._local_error_feedbacks.items():
                    mem_buffer.zero()
                    ref_buf = model._local_ref_points.get(dtype)
                    ref_buf.zero()
        self.pre_loss_scale = self.grad_scaler.scale
        return update_flag, grad_norm, num_zeros_in_grad