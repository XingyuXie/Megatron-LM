import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch._C._distributed_c10d import ReduceOp
import torch
import os
import math


def main():
    dist.init_process_group("nccl")
    group_level = build_tree_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    # create local model
    tensor = (rank + 1) * torch.ones(world_size).cuda()
    tree_all_reduce(tensor, group_level=group_level)


def build_tree_group(group_ranks=None):
    if group_ranks is None:
        group_ranks = list(range(dist.get_world_size()))
    my_rank = dist.get_rank()
    num_levels = int(math.log2(len(group_ranks)))
    group_level = []
    for level in range(0, num_levels):
        match_ranks = set()
        step_size = 2 ** level
        for i in range(0, len(group_ranks)):
            if i + step_size >= len(group_ranks):
                break
            if group_ranks[i] not in match_ranks:
                match_ranks.add(group_ranks[i])
                match_ranks.add(group_ranks[i + step_size])
                new_group = dist.new_group((group_ranks[i], group_ranks[i + step_size]))
                # print('Rank {}: {}'.format(my_rank, (group_ranks[i], group_ranks[i + step_size])))
            else:
                continue
            if group_ranks[i] == my_rank or group_ranks[i + step_size] == my_rank:
                group_level.append(new_group)
    print('Rank {}: {}'.format(dist.get_rank(), list(map(lambda x: dist.get_process_group_ranks(x),
                                                         group_level))))
    return group_level


def tree_reduce_scatter(x, group_level=None, op=ReduceOp.SUM, async_op=False):
    group_ranks = list(range(dist.get_world_size()))
    group_range = dict((r, [0, len(x)]) for r in group_ranks)
    my_rank = dist.get_rank()
    num_levels = int(math.log2(len(group_ranks)))
    for level in range(0, num_levels):
        sub_group = group_level[level]
        sub_group_ranks = dist.get_process_group_ranks(sub_group)
        tensor = x[group_range[my_rank][0]:group_range[my_rank][1]]
        half_size = (group_range[my_rank][1] - group_range[my_rank][0]) // 2
        if my_rank == sub_group_ranks[0]:
            tensor_views = x[group_range[my_rank][0]:group_range[my_rank][0] + half_size]
            group_range[my_rank][1] = group_range[my_rank][0] + half_size
        else:
            tensor_views = x[group_range[my_rank][0] + half_size:group_range[my_rank][1]]
            group_range[my_rank][0] = group_range[my_rank][0] + half_size
        print('Rank {}, Group Rank {}, '
              'Level {}: Before reduce on group {} -> {}'.format(dist.get_rank(), sub_group.rank(),
                                                                 level, sub_group_ranks, x))
        dist.reduce_scatter_tensor(
            tensor_views,
            tensor,
            op=op,
            async_op=False if level != num_levels - 1 else async_op,
            group=sub_group
        )
        print('Rank {}, Group Rank {}, '
              'Level {}: After reduce on group {} -> {}'.format(dist.get_rank(), sub_group.rank(),
                                                                level, sub_group_ranks, x))
    return x


def tree_all_reduce(x, group_level=None, op=ReduceOp.SUM, async_op=False):
    group_ranks = list(range(dist.get_world_size()))
    group_range = dict((r, [0, len(x)]) for r in group_ranks)
    my_rank = dist.get_rank()
    num_levels = int(math.log2(len(group_ranks)))
    for level in range(0, num_levels):
        sub_group = group_level[level]
        sub_group_ranks = dist.get_process_group_ranks(sub_group)
        tensor = x[group_range[my_rank][0]:group_range[my_rank][1]]
        half_size = (group_range[my_rank][1] - group_range[my_rank][0]) // 2
        if my_rank == sub_group_ranks[0]:
            tensor_views = x[group_range[my_rank][0]:group_range[my_rank][0] + half_size]
            group_range[my_rank][1] = group_range[my_rank][0] + half_size
        else:
            tensor_views = x[group_range[my_rank][0] + half_size:group_range[my_rank][1]]
            group_range[my_rank][0] = group_range[my_rank][0] + half_size
        print('Rank {}, Group Rank {}, '
              'Level {}: Before reduce on group {} -> {}'.format(dist.get_rank(), sub_group.rank(),
                                                                 level, sub_group_ranks, x))
        dist.reduce_scatter_tensor(
            tensor_views,
            tensor,
            op=op,
            async_op=False if level != num_levels - 1 else async_op,
            group=sub_group
        )
        print('Rank {}, Group Rank {}, '
              'Level {}: After reduce on group {} -> {}'.format(dist.get_rank(), sub_group.rank(),
                                                                level, sub_group_ranks, x))
    for r in range(dist.get_world_size()):
        if r == my_rank:
            dist.all_gather_into_tensor(
                x,
                x[group_range[my_rank][0]:group_range[my_rank][1]],
                async_op=async_op,
                group=None
            )
            print('Rank {}: After gather -> {}'.format(dist.get_rank(), x))
    return x


if __name__ == "__main__":
    main()
