import os

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as all_gather_grads


def is_main_process() -> bool:
    return dist.get_rank() == 0


def setup_ddp() -> None:
    if "LOCAL_RANK" not in os.environ:
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )

    dist.barrier(device_ids=[local_rank])

def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def pad_tensor_to_same_length(tensor: Tensor, world_size: int) -> Tensor:
    """
    Pad tensors across distributed ranks to the same length.
    """
    # Gather lengths from all devices and pad tensors to the same length.
    local_len = torch.tensor([tensor.size(1)], device=tensor.device, dtype=torch.long)
    gathered_lens = [torch.empty_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    max_len = torch.stack(gathered_lens).max().item()

    if local_len.item() < max_len:
        size = [tensor.shape[0]] + [max_len - local_len.item()] + list(tensor.shape[2:])
        tensor = torch.cat([tensor, tensor.new_zeros(size)], dim=1)

    return tensor


def dist_gather(tensor: Tensor, rank: int, world_size: int) -> Tensor:
    if tensor.dim() > 1:
        tensor = pad_tensor_to_same_length(tensor, world_size)

    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    gathered[rank] = tensor

    return torch.cat(gathered, dim=0)


def dist_gather_with_grads(tensor: Tensor, world_size: int) -> Tensor:
    if tensor.dim() > 1:
        tensor = pad_tensor_to_same_length(tensor, world_size)

    tensor = tensor.contiguous()
    gathered = all_gather_grads(tensor)

    return torch.cat(gathered, dim=0)
