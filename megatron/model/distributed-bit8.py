# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC
from abc import abstractmethod
import math

import torch
from contextlib import nullcontext
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_args
from megatron.core import mpu
from .module import MegatronModule
import time

from typing import Dict, Callable

from sortedcontainers import SortedDict
from megatron import get_timers
from functools import partial


import heapq

class ParamScheduler:
    def __init__(self):
        self.params = []
        self.timestamp = 0
        self.first_pass = True
        self.param_index = {}  # 用于保存每个param的时间戳
        self.sorted_params_cache = {}  # 为每个param保存排序后的参数列表
        self.first_param = None

    def add_param(self, param):
        if self.first_pass:
            size = param.data.nelement()
            self.params.append((param, size, self.timestamp))
            self.param_index[param] = self.timestamp
            self.timestamp += 1
            if len(self.params)==1: self.first_param = param 

    def _get_sorted_params_after(self, current_param):
        current_timestamp = self.param_index.get(current_param)
        if current_timestamp is None:
            raise ValueError("The current_param is not found in the added parameters.")
        
        valid_params = [(-size, timestamp, param) for param, size, timestamp in self.params if timestamp > current_timestamp]
        heapq.heapify(valid_params)  # Convert valid_params into a valid heap
    
        return valid_params


    def get_next_params(self, current_param, n=1):
        # 使用列表的copy方法来复制缓存中的列表
        sorted_params = self.sorted_params_cache.get(current_param, []).copy()
        
        # 获取最大的n个元素
        largest_n_params = [heapq.heappop(sorted_params)[2] for _ in range(min(n, len(sorted_params)))]
    
        # 由于我们是从复制的列表中提取数据，因此不需要更新缓存
        return largest_n_params


    def finish_first_pass(self):
        for param, _, timestamp in self.params:
            self.sorted_params_cache[param] = self._get_sorted_params_after(param)
        self.first_pass = False





class ParamScheduler_old:
    def __init__(self):
        self.param_order = {}
        self.last_param = None
        self.first_pass = True
        self.first_param = None

    def add_param(self, param):
        if self.first_pass:
            if self.last_param is not None:
                self.param_order[id(self.last_param)] = param
            self.last_param = param
            if len(self.param_order)==0: self.first_param = param 

    def get_next_params(self, current_param, n=1):
        if not self.first_pass:
            next_params = []
            next_param = current_param
            for _ in range(n):
                next_param_id = id(next_param)
                if next_param_id in self.param_order:
                    next_param = self.param_order[next_param_id]
                    next_params.append(next_param)
                else:
                    # 如果没有更多的参数，就结束循环
                    break
            return next_params
        else:
            raise Exception("Still in the first pass, next params are not available")


    def finish_first_pass(self):
        self.first_pass = False


class MemoryManager:
    def __init__(self, gpu_memory, dtype, device = 'gpu'):
        if device == 'cpu':
            self.memory = torch.zeros(gpu_memory,
                                    dtype=dtype,
                                    device=torch.device("cpu"),
                                    requires_grad=False,
                                    pin_memory=True)
        else:
            self.memory = torch.zeros(gpu_memory,
                                    dtype=dtype,
                                    device=torch.cuda.current_device(),
                                    requires_grad=False)
        self.total_memory = gpu_memory
        self.free_blocks = SortedDict({0: gpu_memory})

    def allocate(self, shape):
        size = shape.numel()
        # 找到最合适的空闲内存块
        suitable_blocks = {k: v for k, v in self.free_blocks.items() if v >= size}
        if suitable_blocks:
            # 选择第一个符合大小需求的内存块
            start_idx, block_size = min(suitable_blocks.items(), key=lambda x: x[1])
            end_idx = start_idx + size
            
            # 更新 free_blocks
            if block_size > size:
                self.free_blocks[start_idx + size] = block_size - size
                
            del self.free_blocks[start_idx]
            
            return True, self.memory[start_idx:end_idx].view(shape), start_idx
        # print("Not enough memory available!")
        return False, None, None

    def free(self, start_idx, size):
        # 释放内存
        self.free_blocks[start_idx] = size

        # 合并相邻或重叠的空闲内存块
        keys = list(self.free_blocks.keys())
        merged_blocks = SortedDict()
        current_start, current_size = keys[0], self.free_blocks[keys[0]]

        for key in keys[1:]:
            if key <= current_start + current_size:
                current_size = max(current_size, key + self.free_blocks[key] - current_start)
            else:
                merged_blocks[current_start] = current_size
                current_start, current_size = key, self.free_blocks[key]

        merged_blocks[current_start] = current_size
        self.free_blocks = merged_blocks

    def reset(self):
        # 重置内存管理器，所有内存都处于空闲状态
        self.free_blocks = SortedDict({0: self.total_memory})
    
    def zero(self):
        """Reset the buffer to zero."""
        self.memory.zero_()


class MemoryBuffer:

    def __init__(self, numel, numel_padded, dtype, device = None):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        if device is not None:
            self.data = torch.zeros(self.numel_padded,
                                    dtype=self.dtype,
                                    device=device,
                                    requires_grad=False,
                                    pin_memory=True)
        else:
            self.data = torch.zeros(self.numel_padded,
                                    dtype=self.dtype,
                                    device=torch.cuda.current_device(),
                                    requires_grad=False)
    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor



class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module


    @abstractmethod
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)



class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers,
                 grad_compression=False):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        self.grad_compression=True#grad_compression
        self._before_opt_step = False
        if self.grad_compression:
            self.lowbit_scale = 2048.0*64.0#torch.tensor([2048*2.0], dtype=torch.float32).cuda()
            self.error_beta = 0.8
            self.ref_beta = 0.95
            self.ref_lr = 1.0

            self.compressed_overflow = torch.cuda.FloatTensor([0.0])
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        self._grad_buffer_param_index_map = None
        if self.use_contiguous_buffers:
            self._grad_buffers = {}
            if self.grad_compression: 
                self._local_error_feedbacks = {}
                self._local_ref_points = {}
                self._pre_grad_buffers = {}
                self._lowbit_grad_buffers = {}
                ## prefetch
                self._ref_prefetch = {}
                self._err_prefetch = {}
                self._param_stream = {}
                self._error_norm = {}
                self._init_grad_flag = False
            self._grad_buffer_param_index_map = {}
            data_parallel_world_size = mpu.get_data_parallel_world_size()

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()

            

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():

                # If using distributed optimizer, pad memory buffer to be
                # multiple of data_parallel_world_size. (This padding is done
                # due to a constraint with the reduce_scatter op, which requires
                # all tensors have equal size. See: optimizer.py.)
                num_elements_padded = data_parallel_world_size * \
                    int(math.ceil(num_elements / data_parallel_world_size))

                # Allocate grad buffer.
                self._grad_buffers[dtype] = MemoryBuffer(num_elements,
                                                         num_elements_padded,
                                                         dtype)
                if self.grad_compression:
                    # and \
                    # dtype not in (torch.int8, torch.int16, torch.int32, torch.int64): 
                    self._lowbit_grad_buffers[dtype] = MemoryBuffer(num_elements,
                                                        num_elements_padded,
                                                        torch.int8)
                    self._local_error_feedbacks[dtype] = MemoryBuffer(num_elements,
                                                            num_elements_padded,
                                                            param.dtype)#, device=torch.device("cpu"))
                    self._local_ref_points[dtype] = MemoryBuffer(num_elements,
                                                            num_elements_padded,
                                                            param.dtype)#, #device=torch.device("cpu"))
                    
                    # self._pre_grad_buffers[dtype] = MemoryBuffer(num_elements,
                    #                                         num_elements_padded,
                    #                                         param.dtype)#, #device=torch.device("cpu"))

                    ## prefetch 
                    # num_elements = int(num_elements*0.2)
                    # self._err_prefetch[dtype] = MemoryManager(num_elements,
                    #                                         param.dtype)
                    # self._ref_prefetch[dtype] = MemoryManager(num_elements,
                    #                                         param.dtype)


            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            self.skip_param = set()
            for param_name, param in self.module.named_parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    
                    if self.grad_compression:
                    # and \
                    # dtype not in (torch.int8, torch.int16, torch.int32, torch.int64): 
                    #     if torch.distributed.get_rank() == 0:
                    #         print("name is {}; Size is {};".format(param_name,param.data.nelement()), flush=True)
                        # if 'word_embeddings.weigh' in param_name:
                        #     param.local_error_gpu_test = torch.zeros_like(param, dtype=param.dtype)
                        #     param.local_ref_gpu_test = torch.zeros_like(param, dtype=param.dtype)
                        #     self.skip_param.add(param)
                        # else:
                        param.local_error = \
                            self._local_error_feedbacks[dtype].get(
                            param.data.shape, type_num_elements[dtype])
                        param.local_ref = \
                            self._local_ref_points[dtype].get(
                            param.data.shape, type_num_elements[dtype])
                        # param.pre_grad = \
                        #     self._pre_grad_buffers[dtype].get(
                        #     param.data.shape, type_num_elements[dtype])
                        param.lowbit_grad = \
                            self._lowbit_grad_buffers[dtype].get(
                            param.data.shape, type_num_elements[dtype])
                        self._param_stream[param] = \
                            torch.cuda.Stream(device=torch.cuda.current_device())

                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])   
                        
                    if dtype not in self._grad_buffer_param_index_map:
                        self._grad_buffer_param_index_map[dtype] = {}
                    self._grad_buffer_param_index_map[dtype][param] = (
                        type_num_elements[dtype],
                        type_num_elements[dtype] + param.data.nelement(),
                    )

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)

            ## prefetch part
            self.reset_reverse_param_iter()
            self.param_order = ParamScheduler()
            
            

    ## prefetch part
    def reset_reverse_param_iter(self):
        if not self.grad_compression: return
        for dtype, mem in self._err_prefetch.items():
            mem.reset()
            self._ref_prefetch[dtype].reset()
            self._error_norm[dtype] = (1e-8, 1e-8)


    def param_hook_gpu_tensor(self, param):
        if getattr(param, 'local_error_gpu', None) is None and param not in self.skip_param:
            timers = get_timers()
            # stream = torch.cuda.Stream()
            # with self.stream_context():
            dtype = torch.float if \
                self.accumulate_allreduce_grads_in_fp32 else param.dtype
            flag, err_tensor, err_idx = self._err_prefetch[dtype].allocate(param.data.shape)
            # assert flag
            if not flag: return False
            # _, ref_tensor, ref_idx = self._ref_prefetch[dtype].allocate(param.data.shape)
            
            timers('prefetch-load-CPU2GPU-prefetch', log_level=2).start()
            with torch.cuda.stream(self._param_stream[param]):#(self._cpu_to_gpu_stream):
                err_tensor.copy_(param.local_error, non_blocking=True)
                # ref_tensor.copy_(param.local_ref, non_blocking=True)
            # param.local_ref_gpu = param.local_ref.cuda(non_blocking=True)
            # param.local_error_gpu = param.local_error.cuda(non_blocking=True)
            timers('prefetch-load-CPU2GPU-prefetch').stop()
            # param.local_ref_gpu = ref_tensor
            param.local_error_gpu = err_tensor
            param.err_idx = err_idx
            # param.ref_idx = ref_idx
        return True
        
    def prefetch_param_hook_gpu_tensor(self, param, skip_first = False):
        if not skip_first and not self.param_hook_gpu_tensor(param): return
        param_loads = self.param_order.get_next_params(param)
        for param_load in param_loads:
            if not self.param_hook_gpu_tensor(param_load): return

    @torch.no_grad()
    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                # The gradient function of linear layers is fused with GEMMs
                param.main_grad.add_(param.grad.data)
                # param.lowbit_grad.copy_(param.main_grad)
                # try:
                #     assert (param.lowbit_grad == param.main_grad).all()
                # except AssertionError:
                #     if torch.distributed.get_rank() == 0:
                #         diff = param.lowbit_grad - param.main_grad
                #         diff_idx = (diff.abs() > 1e-6)
                #         count_diff = diff_idx.sum().item()  # 使用.item()来获取Python标量
                #         ref_diff = param.lowbit_grad[diff_idx]
                #         print("hook print: lowbit_grad size is {}, main_grad size is {}".format(
                #             param.lowbit_grad.numel(),  # 使用.numel()来获取元素数量
                #             param.main_grad.numel()), flush=True)
                #         print("hook print: gbuf size is {}, diff count is {}".format(
                #             param.lowbit_grad.numel(),
                #             count_diff), flush=True)
                #         print("hook print: stats: max is {}, mean is {}, min is {}".format(
                #             ref_diff.max().item(),  # 使用.item()来获取Python标量
                #             ref_diff.mean().item(),  # 使用.item()来获取Python标量
                #             ref_diff.min().item()), flush=True)  # 使用.item()来获取Python标量

                # if param not in self.skip_param: self.param_order.add_param(param)
                if self._before_opt_step and hasattr(param, 'local_error') \
                    and not self.param_order.first_pass:
                    # and self._init_grad_flag:
                    # timers = get_timers()
                    # dtype = torch.float if \
                    #     self.accumulate_allreduce_grads_in_fp32 else param.dtype
                    
                    # if getattr(param, 'local_error_gpu_test', None) is not None:
                    #     # local_ref = param.local_ref_gpu_test
                    #     local_error = param.local_error_gpu_test
                    #     # if torch.distributed.get_rank() == 0:
                    #     #     print("In GPU;", flush=True)
                    # elif getattr(param, 'local_error_gpu', None) is None:
                    #     timers('prefetch-load-CPU2GPU-prefetch', log_level=2).start()
                    #     with torch.cuda.stream(self._param_stream[param]):#(self._cpu_to_gpu_stream):
                    #         # local_ref = torch.empty_like(param.main_grad, dtype=param.dtype)
                    #         local_error = torch.empty_like(param.main_grad, dtype=param.dtype)
                    #         # local_ref.copy_(param.local_ref, non_blocking=True)
                    #         local_error.copy_(param.local_error, non_blocking=True)
                    #     timers('prefetch-load-CPU2GPU-prefetch').stop()

                    # else:                       
                    #     # local_ref = param.local_ref_gpu
                    #     local_error = param.local_error_gpu
                    local_ref = param.local_ref
                    local_error = param.local_error
                    # pre_grad = param.pre_grad
                    # data_parallel_world_size = mpu.get_data_parallel_world_size()
                    ## wait io stream
                    # torch.cuda.current_stream().wait_stream(self._param_stream[param])

                    
                    local_var = param.grad
                    local_main_grad = param.main_grad               
                    # param.lowbit_grad.copy_(local_main_grad)
                    # local_main_grad.add_(local_ref, alpha=-1.0)
                    # local_var.copy_(local_main_grad)
                    # grad_norm = local_main_grad.norm(2.0).div_(mpu.get_data_parallel_world_size())
                    # torch.distributed.all_reduce(grad_norm,
                    #                              group=mpu.get_tensor_model_parallel_group(),
                    # )
                    

                    
                    if not self._init_grad_flag:
                        local_ref.zero_().add_(param.data, 
                                        alpha = (1.-self.ref_beta))
                        # local_ref.copy_(local_main_grad)
                        # torch.distributed.all_reduce(local_ref,
                        #                          group=mpu.get_tensor_model_parallel_group(),
                        # )
                        # local_main_grad.div_(mpu.get_data_parallel_world_size())
                        # pre_grad.add_(local_main_grad, alpha = -(1.0-self.ref_beta))
                        param.grad = None
                        # if torch.distributed.get_rank() == 0:
                        #     print("_init_grad_flag;", flush=True)
                        return
                    else:
                        # local_ref.mul_(self.ref_beta).add_(local_main_grad, alpha = (1.0-self.ref_beta))
                        # pre_grad.add_(local_main_grad, alpha = (1.0-self.ref_beta))
                        # inv_local_lr = (torch.sqrt(pre_grad).add_(1e-8)).div(self.ref_lr).mul_(param.data)
                        local_ref.add_(param.data, alpha = -(1.0-self.ref_beta))#.clamp_(min=-1.0, max=1.0)
                        # tmp_local_ref = local_ref
                        #.mul_(2.0)
                        # tmp_local_ref = local_ref.pow(3.0)#.mul_(2.0)
                        sign_ref = local_ref.sign()
                        tmp_local_ref = local_ref.square().mul_(sign_ref).mul_(1.5)
                        # local_ref.div_(local_ref.norm(2.0)+1e-8).mul_(grad_norm)


                    torch.cuda.nvtx.range_push("Err-FeedBack&Ref-Point")

                    # local_ref_new = local_ref+pre_grad
                    # local_main_grad.mul_(2.0).add_(pre_grad, alpha = -1.0)
                    local_main_grad.add_(tmp_local_ref, alpha = -1.0).add_(local_error)#.clamp_(min=-1.0, max=1.0)
                    #.add_(local_error, alpha=1.0/self.lowbit_scale)
                    # local_main_grad.add_(local_ref, alpha = -1.0)
                    # local_var.add_(local_error, alpha=1.0/self.lowbit_scale)

                    # local_var.add_(local_ref_new, alpha=-1.0)
                    # local_main_grad.mul_(self.lowbit_scale)
                    # max_value = torch.max(local_main_grad)
                    # min_value = torch.min(local_main_grad)
                    # # if torch.distributed.get_rank() == 0:
                    # #     print("Rank{}: Max Value is {}; Min Value is {};".format(torch.distributed.get_rank(),
                    # #                                         max_value, min_value), flush=True)
                    # if max_value > 8191*32 or min_value < -8192*32:
                    #     self.compressed_overflow.fill_(1.0)
                    # torch.clamp(local_main_grad, min=-128, max=127, out=pre_grad)
                    param.lowbit_grad.copy_(local_main_grad.mul(self.lowbit_scale)\
                                            .clamp_(min=-127, max=127).to(torch.int8))
                    #.to(torch.int8))
                
                
                    # update error
                    # local_var.copy_(param.lowbit_grad).div_(self.lowbit_scale)
                    # local_var.add_(local_main_grad, alpha=-1.0)#.div_(self.lowbit_scale/data_parallel_world_size)
                    # local_error.copy_(local_var)
                    local_main_grad.add_(param.lowbit_grad, alpha=-1./self.lowbit_scale)
                    local_error.mul_(self.error_beta).add_(local_main_grad, alpha = (1.-self.error_beta))
                    
                    local_main_grad.copy_(tmp_local_ref)
                    # local_main_grad.zero_()
                    # torch.sqrt(pre_grad, out=inv_local_lr)
                    # inv_local_lr.add_(1e-8).div(self.ref_lr).mul_(param.data)
                    local_ref.mul_(self.ref_beta).add_(param.data, alpha = (1.-self.ref_beta))
                                        # alpha = (1.0-self.ref_beta)*(1-0.1*self.ref_lr))

                    torch.cuda.nvtx.range_pop()
                    
                    
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook


    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()


    def broadcast_params(self):
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data,
                                        src=mpu.get_data_parallel_src_rank(),
                                        group=mpu.get_data_parallel_group())


    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        start_time = time.time()
        if self._grad_buffers is not None:
            for _, buffer_ in self._grad_buffers.items():
                if self.grad_compression:
                    buffer_.data.div_(mpu.get_data_parallel_world_size()/1024.0)
                    tmp = buffer_.data.to(torch.int8)
                    torch.distributed.all_reduce(
                        tmp, group=mpu.get_data_parallel_group())
                    buffer_.data.copy_(tmp).div_(1024.0)
                else:
                    buffer_.data /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        buffer_.data, group=mpu.get_data_parallel_group())
        else:
            # Otherwise, bucketize and all-reduce
            buckets = {}
            # Pack the buckets.
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                if self.grad_compression:
                    coalesced.div_(mpu.get_data_parallel_world_size()/1024.0)
                    tmp = coalesced.to(torch.int8)
                    torch.distributed.all_reduce(
                        tmp, group=mpu.get_data_parallel_group())
                    coalesced=tmp
                else:
                    coalesced /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        coalesced, group=mpu.get_data_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)
                    if self.grad_compression: buf.div_(1024.0)
                    
        print('done with grad reduce. Compilation time: {:.3f} seconds; grad compression is {}'.format(time.time() - start_time),
              self.grad_compression)
