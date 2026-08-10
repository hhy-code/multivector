from typing import Any

import torch
from torch import Tensor
import yaml


def insert_prefix_token_id(tensor: Tensor, prefix_id: int) -> Tensor:
    prefix_tensor = torch.full(
        (tensor.size(0), 1),
        prefix_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor[:, :1], prefix_tensor, tensor[:, 1:]], dim=1)


def insert_prefix_token_ids(tensor: Tensor, prefix_ids: list[int]) -> Tensor:
    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=tensor.device)
    prefix_tensor = prefix_tensor.unsqueeze(0).expand(tensor.size(0), -1)
    return torch.cat([tensor[:, :1], prefix_tensor, tensor[:, 1:]], dim=1)


def insert_suffix_token_ids(tensor: Tensor, suffix_ids: list[int]) -> Tensor:
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=tensor.device)
    suffix_tensor = suffix_tensor.unsqueeze(0).expand(tensor.size(0), -1)
    return torch.cat([tensor[:, :-1], suffix_tensor, tensor[:, -1:]], dim=1)


def mask_sep(mask: Tensor) -> Tensor:
    sep_positions = (mask.sum(dim=-1) - 1).long()
    return mask.scatter(1, sep_positions[:, None], 0)


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(config: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(config, str):
        config = _load_config(config)

    if not isinstance(config, dict):
        raise TypeError("The loaded model configuration must be a dictionary")

    return config


def save_config(config: dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config, f)