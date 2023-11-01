# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron distributed optimizer."""


from apex.optimizers import FusedAdam as Adam
import math
import torch
import time

from megatron import print_rank_0
from megatron.core import mpu, tensor_parallel

from ..tree import tree_reduce_scatter
from functools import reduce

from .optimizer import _zero_grad_group_helper
from .distrib_optimizer import DistributedOptimizer
from itertools import chain



def compress_tensor(input_tensor):
    # Ensure that the input tensor has dtype int8.
    if input_tensor.dtype != torch.int8:
        raise ValueError("Input tensor must have dtype torch.int8.")
    mask1 = torch.tensor([15], dtype=torch.int8).to(input_tensor)
    # print("Before bitwise_and_", flush=True)
    input_tensor[:input_tensor.numel()//2].bitwise_and_(mask1)
    # print("Before bitwise_left_shift_", flush=True)
    input_tensor[input_tensor.numel()//2:].bitwise_left_shift_(4)

    # print("Before bitwise_or", flush=True)
    input_tensor[:input_tensor.numel()//2].bitwise_or_(input_tensor[input_tensor.numel()//2:])
    # input_tensor[idx_part1] = torch.bitwise_or(input_tensor[idx_part1], input_tensor[idx_part2])
    

def decompress_tensor(compressed: torch.tensor, out_tensor: torch.tensor):
    mask1 = torch.tensor([15], dtype=torch.int8).to(compressed)
    mask2 = torch.tensor([-16], dtype=torch.int8).to(compressed)
    torch.bitwise_right_shift(
        torch.bitwise_left_shift(
            torch.bitwise_and(compressed, mask1), 4
        ), 4, out=out_tensor[:out_tensor.numel()//2]
    )
    torch.bitwise_right_shift(torch.bitwise_and(compressed, mask2), 4, out=out_tensor[out_tensor.numel()//2:])
    return out_tensor

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
        self.low_bit_overflow = torch.cuda.FloatTensor([0.0])
        self.low_bit_step_gap = 0.0
        self.lowbit_gap = 1024    
        
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
                
                shard_size = int(buf.numel() / data_parallel_world_size)
                buf_views = [buf[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)]
                
                if lowbit_grad_buf is not None and model._init_grad_flag:
                    # torch.cuda.current_stream().wait_stream(model._hook_stream)
                    lowbit_grad_buf = lowbit_grad_buf.data
                    lowbit_grad_buf_views = []
                    reused_lowbit_buf_views = []
                    # print("Before chain", flush=True)
                    # half_idx = buf.numel() // 2
                    # idx_part1 = [i for r in range(data_parallel_world_size) for i in range(r * shard_size, r * shard_size + shard_size // 2)]
                    # idx_part2 = [i for r in range(data_parallel_world_size) for i in range(r * shard_size + shard_size // 2, (r + 1) * shard_size)]
                    
                    # print("Before reorder", flush=True)
                    # part2 = lowbit_grad_buf[idx_part2]
                    # lowbit_grad_buf[:half_idx].copy_(lowbit_grad_buf[idx_part1])
                    # lowbit_grad_buf[half_idx:].copy_(part2)
                    # print("Before compress_tensor", flush=True)
                    # compress_tensor(lowbit_grad_buf)
                    for r in range(data_parallel_world_size):
                        start_idx = r * shard_size
                        end_idx = (r + 1) * shard_size
                        mid_idx = (start_idx + end_idx) // 2

                        compress_tensor(lowbit_grad_buf[start_idx: end_idx])
                        lowbit_grad_buf_views.append(lowbit_grad_buf[start_idx: mid_idx])
                        reused_lowbit_buf_views.append(lowbit_grad_buf[mid_idx: end_idx])
                    decompress_buf = lowbit_grad_buf[:shard_size]

                    view_items.append((model_index, dtype, 
                                       buf, buf_views,
                                    #    idx_part1, idx_part2,
                                       lowbit_grad_buf, decompress_buf, 
                                       lowbit_grad_buf_views, reused_lowbit_buf_views,
                                       model.lowbit_scale))
                else:
                    view_items.append((model_index, dtype, 
                                    buf, buf_views,
                                    None, None,
                                    None, None,
                                    None))
            if not model._init_grad_flag:
                model._init_grad_flag = True
                # if torch.distributed.get_rank() == 0:
                #     print("SET _init_grad_flag", flush=True)
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
        # torch.cuda.nvtx.range_push("grads-reduce-scatter")
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()
               
        # Reduce-scatter all grads.
        gbuf_view_items = self.get_lowbit_buffer_dp_views()
        for index, (model_index, dtype,
                    gbuf, gbuf_views,
                    # idx_part1, idx_part2,
                    lowbit_grad_buf, decompress_buf,
                    lowbit_grad_buf_views, local_lowbit_buf_views,
                    lowbit_scale) \
            in enumerate(gbuf_view_items):
            if lowbit_grad_buf is None:
                torch.distributed.reduce_scatter_tensor(
                    gbuf_views[data_parallel_rank],
                    gbuf.div_(data_parallel_world_size),
                    group = data_parallel_group,
                )
                continue
            # shard_size = int(gbuf.numel() / (2*data_parallel_world_size))
            # # compressed_tensor_views = lowbit_grad_buf_views[data_parallel_rank]
            # lowbit_grad_buf_gather = torch.zeros_like(lowbit_grad_buf[0:gbuf.numel()//2],
            #                                           dtype=lowbit_grad_buf.dtype,
            #                                           device=torch.cuda.current_device())
            # local_lowbit_buf_views = [lowbit_grad_buf_gather[(r*shard_size):((r+1)*shard_size)]
            #                  for r in range(data_parallel_world_size)]
            
            # compress_tensor(lowbit_grad_buf, idx_part1, idx_part2)
            torch.distributed.all_to_all(
                local_lowbit_buf_views,
                lowbit_grad_buf_views,
                # output_split_sizes=[shard_size]*data_parallel_world_size,
                # input_split_sizes=[shard_size]*data_parallel_world_size,
                group = data_parallel_group
            )
            def custom_reduce_fn(x, y):
                x.add_(decompress_tensor(y, decompress_buf),
                       alpha = 1.0/(lowbit_scale*data_parallel_world_size))
                return x
            
            result = reduce(
                lambda x, y: custom_reduce_fn(x, y),
                local_lowbit_buf_views,
                gbuf_views[data_parallel_rank]
            )
        timers('grads-reduce-scatter').stop()
        
       
        
        # print_rank_0('done with grad reduce. Compilation time: {:.3f} seconds; grad compression is {}'
        #       .format(time.time() - start_time, self.grad_compression))

    @torch.no_grad()
    def step(self, args, timers):
        self.low_bit_step_gap+=1.0
        if self.low_bit_step_gap > 512:
        #% 512 == 0:
            self.low_bit_step_gap = 0.0
            for model_index, model in enumerate(self.models):
                # model.ref_beta = min(model.ref_beta+1.0/(45_000),0.99)
                for dtype, mem_buffer in model._local_error_feedbacks.items():
                    mem_buffer.zero()
                for param_in_mem in model.skip_param:
                    param_in_mem.local_error_gpu_test.zero_()
            
        # if self.low_bit_step_gap == 10000:
        #     for model_index, model in enumerate(self.models):
        #             model.lowbit_scale = 2048.0*64.0
        #             print("Rescale err_buf", flush=True)
                    #  mem_buffer.data.div_(1000.0)
        # # data_parallel_world_size = mpu.get_data_parallel_world_size()
        # # data_parallel_group = mpu.get_data_parallel_group()
        # torch.distributed.all_reduce(self.low_bit_overflow,==============
        #                              op=torch.distributed.ReduceOp.MAX,
        #                              group=self.get_model_parallel_group()
        #                             )
        # # Update across all data parallel instances.
        # if  (self.low_bit_overflow.item() > 0):
        #     # self._copy_model_grads_to_main_grads()
        #     self.low_bit_overflow.fill_(0.0)
        #     self.low_bit_step_gap = 0.0
            
            
        #     for model_index, model in enumerate(self.models):
        #         scale = max(model.lowbit_scale*0.5, 1.0) / model.lowbit_scale
        #         model.lowbit_scale = max(model.lowbit_scale*0.5, 1.0)
        #         model.compressed_overflow.fill_(0.0)
                
        #         # model.ref_beta = min(model.ref_beta+1.0/(45_000),0.99)
        #         for dtype, mem_buffer in model._local_error_feedbacks.items():
        #             # mem_buffer.zero()
        #             mem_buffer.data.mul_(scale)
        #             # ref_buf = model._local_ref_points.get(dtype)
        #             # # ref_buf.data.mul_(scale)
        #             # torch.distributed.all_reduce(ref_buf.data,
        #             #                  group=data_parallel_group)
        #             # ref_buf.data.div_(data_parallel_world_size)
        #             # # ref_buf.zero()
        #             # pre_buf = model._pre_grad_buffers.get(dtype)
        #             # torch.distributed.all_reduce(pre_buf.data,
        #             #                  group=data_parallel_group)
        #             # pre_buf.data.div_(data_parallel_world_size)
        #         local_lowbit_scale = torch.tensor([model.lowbit_scale], dtype=torch.float32).cuda()
        #         low_bit_all = torch.empty([torch.distributed.get_world_size(), 1], dtype=torch.float32).cuda()
        #         torch.distributed.all_gather_into_tensor(low_bit_all,
        #                                             local_lowbit_scale)
        #             # print("Rescale err_buf", flush=True)
        #         if torch.distributed.get_rank() == 0:
        #             print("Decay lowbit_scale to {} ;".format(low_bit_all), flush=True)
        #     # self._copy_model_grads_to_main_grads()
        #     return False, None, None
        # else:
        #     # print("Rank{}: Pass the low_bit_overflow at step {};".format(torch.distributed.get_rank(), self.low_bit_step_gap), flush=True)
        #     self.low_bit_step_gap+=1.0
        #     if self.low_bit_step_gap > self.lowbit_gap:
        #         self.low_bit_step_gap = 0.0
                
        #         for model_index, model in enumerate(self.models):
        #             model.lowbit_scale *= 2.0
                    
        #             # model.ref_beta = min(model.ref_beta+1.0/(45_000),0.999)
        #             for dtype, mem_buffer in model._local_error_feedbacks.items():
        #                 mem_buffer.data.mul_(2.0)
        #                 # ref_buf = model._local_ref_points.get(dtype)
        #                 # # ref_buf.zero()
        #                 # # ref_buf.data.mul_(2.0)
        #                 # torch.distributed.all_reduce(ref_buf.data,
        #                 #                 group=data_parallel_group)
        #                 # ref_buf.data.div_(data_parallel_world_size)
        #                 # pre_buf = model._pre_grad_buffers.get(dtype)
        #                 # torch.distributed.all_reduce(pre_buf.data,
        #                 #                 group=data_parallel_group)
        #                 # pre_buf.data.div_(data_parallel_world_size)
        #             local_lowbit_scale = torch.tensor([model.lowbit_scale], dtype=torch.float32).cuda()
        #             low_bit_all = torch.empty([torch.distributed.get_world_size(), 1], dtype=torch.float32).cuda()
        #             torch.distributed.all_gather_into_tensor(low_bit_all,
        #                                             local_lowbit_scale)
        #             if torch.distributed.get_rank() == 0:
        #                 print("Enlarge lowbit_scale to {};".format(low_bit_all), flush=True)
        update_flag, grad_norm, num_zeros_in_grad = super().step(args, timers)
        # if self.pre_loss_scale != self.grad_scaler.scale:
        #     scale_change = self.grad_scaler.scale/self.pre_loss_scale
        #     for model_index, model in enumerate(self.models):
        #         for dtype, err_buf in model._local_error_feedbacks.items():
        #             err_buf.data.mul_(scale_change)
        #             ref_buf = model._local_ref_points.get(dtype)
        #             ref_buf.data.mul_(scale_change) 
        # if not update_flag:
        #     for model_index, model in enumerate(self.models):
        #         for dtype, mem_buffer in model._local_error_feedbacks.items():
        #             mem_buffer.zero()
        #             ref_buf = model._local_ref_points.get(dtype)
        #             ref_buf.zero()
        #             if torch.distributed.get_rank() == 0:
        #                 print("Rezero ref_buf", flush=True)
        # self.pre_loss_scale = self.grad_scaler.scale
        return update_flag, grad_norm, num_zeros_in_grad 