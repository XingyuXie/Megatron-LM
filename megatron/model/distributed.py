# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC
from abc import abstractmethod
import math

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_args
from megatron.core import mpu
from .module import MegatronModule
import time

from typing import Dict


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
                                    pin_memory= True)
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
        self.grad_compression=grad_compression
        self._before_opt_step = False
        if self.grad_compression:
            self._local_error_feedbacks = None
            self._lowbit_grad_buffers = None
            self._local_ref_points = None
            self.lowbit_scale = 32768.0
            self.lowbit_gap = 2048
            self.error_beta = 0.95
            self.ref_lr = 1.0
            print("We will compresse grad to int8!!!")
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
                self._lowbit_grad_buffers = {}
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
                if self.grad_compression and \
                    dtype not in (torch.int8, torch.int16, torch.int32, torch.int64): 
                    self._lowbit_grad_buffers[dtype] = MemoryBuffer(num_elements,
                                                        num_elements_padded,
                                                        torch.int8)
                    self._local_error_feedbacks[dtype] = MemoryBuffer(num_elements,
                                                            num_elements_padded,
                                                            dtype, device=torch.device("cpu"))
                    self._local_ref_points[dtype] = MemoryBuffer(num_elements,
                                                            num_elements_padded,
                                                            dtype, device=torch.device("cpu"))
                    


            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            # self.main_grad_list = []
            # self.ref_point_list = []
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype])
                    if self.grad_compression and \
                    dtype not in (torch.int8, torch.int16, torch.int32, torch.int64): 
                        # param.local_error = \
                        #     torch.zeros_like(param.main_grad, dtype=dtype)
                        # param.local_ref = \
                        #     torch.zeros_like(param.main_grad, dtype=dtype)
                        # self.main_grad_list.append(param.main_grad)
                        # self.ref_point_list.append(param.local_ref)
                        param.local_error = \
                            self._local_error_feedbacks[dtype].get(
                            param.data.shape, type_num_elements[dtype])
                        param.local_ref = \
                            self._local_ref_points[dtype].get(
                            param.data.shape, type_num_elements[dtype])
                        param.lowbit_grad = \
                            self._lowbit_grad_buffers[dtype].get(
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
            # self.used_param = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)
                    # self.used_param.append(param)

            ## prefetch part
            # self.element_thd = sum(type_num_elements.values())*0.5
            # if torch.distributed.get_rank() == 0:
            #     print("Init Reset Prefetch", flush=True)
            self.prefetch_stream = torch.cuda.Stream()
            self.reset_reverse_param_iter()
            ## prefetch part
            self.element_thd = sum(
                [param.data.nelement() \
                 for param in self.module.parameters() \
                    if param.requires_grad]
            )*0.2
            

    ## prefetch part
    def reset_reverse_param_iter(self):
        self.reverse_param_iter = reversed([param for param in self.module.parameters() if param.requires_grad])
        self.prefetch_var: Dict[torch.nn.parameter.Parameter, torch.tensor] = {}
        self.skip_param = set()
        self.current_param_num = 0.0
        # if torch.distributed.get_rank() == 0:
        #     print("Reset Prefetch", flush=True)

    ## prefetch part        
    def prefetch_local_param(self, flag=False):
        # print("current_param_num: ", self.current_param_num)
        if self.current_param_num > self.element_thd:
            if not flag: return
            try:
                with torch.cuda.stream(self.prefetch_stream):
                    for _ in range(2):
                        param = next(self.reverse_param_iter)
                        if not (param in self.skip_param):
                            self.prefetch_var[param] = (
                                param.local_ref.to(param.main_grad, non_blocking=True),
                                param.local_error.to(param.main_grad, non_blocking=True)
                            )
                            self.current_param_num += param.data.nelement()
                    # if torch.distributed.get_rank() == 0:
                    #     print("Continue Prefetch: {}% !".format(self.current_param_num*20/self.element_thd), flush=True)
            except StopIteration: 
                # print("Finish Prefetch:!", flush=True)
                pass
            return
        with torch.cuda.stream(self.prefetch_stream):
            for param in self.reverse_param_iter:
                # try:
                if not (param in self.skip_param):
                    self.prefetch_var[param] = (
                        param.local_ref.to(param.main_grad, non_blocking=True),
                        param.local_error.to(param.main_grad, non_blocking=True)
                    )
                    self.current_param_num += param.data.nelement()
                    # if torch.distributed.get_rank() == 0:
                    #     print("Add Param: {}% !".format(self.current_param_num*20/self.element_thd), flush=True)
                    if self.current_param_num > self.element_thd: break
            # except StopIteration:
            #     print("The Prefetch iterator is exhausted!", flush=True)

    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                # The gradient function of linear layers is fused with GEMMs
                param.main_grad.add_(param.grad.data)
                if self._before_opt_step and hasattr(param, 'local_ref'):
                    # data_parallel_world_size = mpu.get_data_parallel_world_size()
                    # param.grad.copy_(param.main_grad)
                    # local_var = param.grad
                    # # local_var.zero_().add_(gbuf)
                    # local_var.add_(param.local_ref)
                    
                    # compressed_tensor = local_var.mul_(self.lowbit_scale/data_parallel_world_size).to(torch.int8)
                    
                    # local_var.copy_(compressed_tensor).div_(-self.lowbit_scale/data_parallel_world_size)
                    
                    # param.local_ref.add_(param.main_grad,  alpha=self.ref_lr/(1.0+self.ref_lr))
                    # param.local_ref.add_(local_var,  alpha=-1.+self.error_beta)
                    if param in self.prefetch_var:
                        # print("Use prefetch_var!!!", flush=True)
                        local_ref = self.prefetch_var[param][0]
                        local_error = self.prefetch_var[param][1]
                    else:
                        print("No prefetch_var!", flush=True)
                        local_ref = param.local_ref.to(param.main_grad, non_blocking=True)
                        local_error = param.local_error.to(param.main_grad, non_blocking=True)
                        self.skip_param.add(param)
                    data_parallel_world_size = mpu.get_data_parallel_world_size()
                    param.main_grad.add_(local_ref, alpha=-self.ref_lr)
                    param.grad.copy_(param.main_grad)
                    local_var = param.grad
                    # local_var.zero_().add_(gbuf)
                    local_var.add_(local_error)
                    compressed_tensor = local_var.mul_(self.lowbit_scale/data_parallel_world_size).to(torch.int8)
                    param.lowbit_grad.copy_(compressed_tensor)
                    # update error
                    local_var.copy_(compressed_tensor).div_(-self.lowbit_scale/data_parallel_world_size)
                    local_var.add_(param.main_grad)
                    local_error.add_(local_var, alpha=1.-self.error_beta)
            
                    # update ref
                    local_ref.add_(param.main_grad, alpha=1.-self.error_beta)
                    local_ref.div_(1+self.ref_lr)
                    # param.local_ref.add_(param.main_grad)
                    param.main_grad.copy_(local_ref)
                    # with torch.cuda.stream(self.prefetch_stream):
                    param.local_ref.copy_(local_ref, non_blocking=True)
                    param.local_error.copy_(local_error, non_blocking=True)
                    # compressed_tensor=None
                    # local_error = None
                    # local_ref = None
                    with torch.cuda.stream(self.prefetch_stream):
                        if param in self.prefetch_var:
                            del self.prefetch_var[param]
                            self.current_param_num -= param.data.nelement()
                            # if torch.distributed.get_rank() == 0:
                            #     print("Continue Prefetch!", flush=True)
                            self.prefetch_local_param(True)
                    
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
                    
        print('done with grad reduce. Compilation time: {:.3f} seconds; grad compression is {}'.format(time.time() - start_time), self.grad_compression)