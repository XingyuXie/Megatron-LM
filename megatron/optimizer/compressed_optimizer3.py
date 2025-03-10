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
                ref_buf = model._local_error_feedbacks.get(dtype)#_local_ref_points.get(dtype)
                
                shard_size = int(buf.numel() / data_parallel_world_size)
                buf_views = [buf[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)]
                
                if lowbit_grad_buf is not None and model._init_grad_flag:
                    # torch.cuda.current_stream().wait_stream(model._hook_stream)
                    ref_buf = ref_buf.data
                    lowbit_grad_buf = lowbit_grad_buf.data
                    ref_buf_views = [ref_buf[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)]
                    lowbit_grad_buf_views = [lowbit_grad_buf[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)]
                    view_items.append((model_index, dtype, 
                                       buf, buf_views,
                                       ref_buf,ref_buf_views,
                                       lowbit_grad_buf,lowbit_grad_buf_views,
                                       model.lowbit_scale,  model._error_norm[dtype]))
                else:
                    view_items.append((model_index, dtype, 
                                    buf, buf_views,
                                    None, None,
                                    None, None,
                                    None, None))
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
                    ref_buf, ref_buf_views,
                    lowbit_grad_buf, lowbit_grad_buf_views,
                    lowbit_scale,  _error_norm) \
            in enumerate(gbuf_view_items):
            if lowbit_grad_buf is None:
                torch.distributed.reduce_scatter_tensor(
                    gbuf_views[data_parallel_rank],
                    gbuf.div_(data_parallel_world_size),
                    group = data_parallel_group,
                )
                continue

            # norm_before = torch.tensor(math.sqrt(_error_norm[0]),
            #                        device=torch.cuda.current_device()).div_(data_parallel_world_size)
            # torch.distributed.all_reduce(norm_before,
            #         group=data_parallel_group,
            #         async_op=True)
            # torch.distributed.all_reduce(
            #         ref_buf.div_(data_parallel_world_size),
            #         group = data_parallel_group,
            # )
            # self.low_bit_overflow += compressed_overflow
            # gbuf_views[data_parallel_rank].add_(ref_buf_views[data_parallel_rank], alpha=1.0/lowbit_scale)
            # torch.distributed.all_reduce(self.low_bit_overflow,
            #                          op=torch.distributed.ReduceOp.MAX,
            #                          group=self.get_model_parallel_group()
            #                         )
            # if (self.low_bit_overflow.item() > 0): 
            #     continue
            # local_ref = gbuf_views[data_parallel_rank].clone()
            shard_size = int(gbuf.numel() / data_parallel_world_size)
            # compressed_tensor_views = lowbit_grad_buf_views[data_parallel_rank]
            lowbit_grad_buf_gather = torch.zeros_like(lowbit_grad_buf,
                                                      dtype=lowbit_grad_buf.dtype,
                                                      device=torch.cuda.current_device())
            local_lowbit_buf_views = [lowbit_grad_buf_gather[(r*shard_size):((r+1)*shard_size)]
                             for r in range(data_parallel_world_size)]
            torch.distributed.all_to_all(
                local_lowbit_buf_views,
                lowbit_grad_buf_views,
                # output_split_sizes=[shard_size]*data_parallel_world_size,
                # input_split_sizes=[shard_size]*data_parallel_world_size,
                group = data_parallel_group
            )
            # gbuf_views[data_parallel_rank-1].zero_()
            reduce(lambda x, y: x.add_(y, alpha = 1.0/(lowbit_scale*data_parallel_world_size)), 
                   local_lowbit_buf_views,
                   gbuf_views[data_parallel_rank]
            )
            # gbuf_views[data_parallel_rank].mul_(2.0)
            # norm_after = torch.norm(gbuf_views[data_parallel_rank])
            # all_norm_after = [torch.zeros_like(norm_after) for _ in range(data_parallel_world_size)]
            # torch.distributed.all_gather(all_norm_after,norm_after,
            #         group=data_parallel_group,
            #         async_op=True)
            # norm_after = math.sqrt(sum([local_norm**2 for local_norm in all_norm_after]))
            # gbuf_views[data_parallel_rank].mul_(norm_before/(1e-8+norm_after))
            # compressed_tensor_views = lowbit_grad_buf_views[data_parallel_rank]
            # local_var = torch.zeros_like(gbuf_views[data_parallel_rank], dtype = gbuf_views[data_parallel_rank-1].dtype)
            # try:
            #     assert (gbuf_views[data_parallel_rank] == lowbit_grad_buf_views[data_parallel_rank]).all()
            # except AssertionError:
            #     if torch.distributed.get_rank() == 0:
            #         diff = gbuf_views[data_parallel_rank] - lowbit_grad_buf_views[data_parallel_rank]
            #         diff_idx = (diff.abs() > 1e-6)
            #         count_diff = diff_idx.sum().item()  # 使用.item()来获取Python标量
            #         ref_diff = lowbit_grad_buf_views[data_parallel_rank][diff_idx]
            #         # print("hook print: lowbit_grad size is {}, main_grad size is {}".format(
            #         #     param.lowbit_grad.numel(),  # 使用.numel()来获取元素数量
            #         #     param.main_grad.numel()), flush=True)
            #         print("gbuf size is {}, diff count is {}".format(
            #             lowbit_grad_buf_views[data_parallel_rank].numel(),
            #             count_diff), flush=True)
            #         print("stats: max is {}, mean is {}, min is {}".format(
            #             ref_diff.max().item(),  # 使用.item()来获取Python标量
            #             ref_diff.mean().item(),  # 使用.item()来获取Python标量
            #             ref_diff.min().item()), flush=True)  # 使用.item()来获取Python标量
            # torch.distributed.reduce_scatter_tensor(
            #     lowbit_grad_buf_views[data_parallel_rank],
            #     lowbit_grad_buf.div_(data_parallel_world_size),
            #     group = data_parallel_group,
            # )
            # torch.distributed.reduce_scatter_tensor(
            #     gbuf_views[data_parallel_rank],
            #     gbuf.div_(data_parallel_world_size),
            #     group = data_parallel_group,
            # )
            
            # if torch.distributed.get_rank() == 0:
            #         print("reduce_scatter_tensor lowbit_grad_buf, type is {}".format(lowbit_grad_buf.dtype)
            #               , flush=True)
            # gbuf_views[data_parallel_rank].copy_(lowbit_grad_buf_views[data_parallel_rank])
                # add_(lowbit_grad_buf_views[data_parallel_rank], alpha = 1.0/(lowbit_scale))
            # ref_buf_views[data_parallel_rank].mul_(0.98).add_(gbuf_views[data_parallel_rank], alpha = 0.02)
            # gbuf_views[data_parallel_rank-1].div_(lowbit_scale)
            # pec = gbuf_views[data_parallel_rank].norm(2.0)*0.1*lowbit_scale
            # rescale = torch.clamp(pec / (gbuf_views[data_parallel_rank-1].norm(2.0) + 1e-8),
            #                       max=1.0)
            # torch.distributed.all_reduce(rescale,
            #                          op=torch.distributed.ReduceOp.MIN,
            #                          group=data_parallel_group
            #                         )
            # gbuf_views[data_parallel_rank].\
            #         add_(gbuf_views[data_parallel_rank-1].mul_(rescale/lowbit_scale))
            # # gbuf.copy_(lowbit_grad_buf_gather)
            # # sum_low_bit = torch.sum(torch.stack(gbuf_views), dim=0)
            # max_value = torch.max(gbuf_views[data_parallel_rank-1])
            # min_value = torch.min(gbuf_views[data_parallel_rank-1])
            
            # if max_value > 128.0 \
            #     or min_value<-128.0:
            #             # print(sum_low_bit, flush=True)
            #         print("Rank{}: Max Value is {}; Min Value is {};".format(torch.distributed.get_rank(), max_value, min_value), flush=True)
            #         self.low_bit_overflow.fill_(1.0)
            #         return
            # # else: low_bit_overflow = torch.cuda.FloatTensor([0.0])
            # # torch.distributed.all_reduce(low_bit_overflow,
            # #                          op=torch.distributed.ReduceOp.MAX,
            # #                          group=data_parallel_group
            # #                         )
            # # self.low_bit_overflow += low_bit_overflow
            # if (self.low_bit_overflow.item() == 0):
            #     gbuf_views[data_parallel_rank].\
            #         add_(gbuf_views[data_parallel_rank-1].div_(lowbit_scale))
            # lowbit_grad_buf.zero_()
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