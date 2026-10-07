import os
from dataclasses import dataclass
from typing import Any

from safetensors.torch import load_file
import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers.utils import ModelOutput
from transformers import AutoTokenizer

from utils.dist import dist_gather, dist_gather_with_grads
from utils.utils import load_config


@dataclass
class EncoderOutput(ModelOutput):
    loss: Tensor | None = None
    scores: Tensor | None = None


class BaseModel(nn.Module):
    """Base class for multi-vector retrieval models."""
    def __init__(self, temp: float | None = None) -> None:
        super().__init__()
        self.temp = 1.0 if temp is None else float(temp)
        if not 0 < self.temp < float("inf"):
            raise ValueError("`temp` must be positive")
        self.gather_across_devices = True
        self.score_margins = False
        self.loss_dict = {}

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> Any:
        raise NotImplementedError

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor) -> Any:
        raise NotImplementedError

    @classmethod
    def score(cls, Q: dict[str, Tensor], D: dict[str, Tensor], pairwise: bool = True) -> Tensor:
        raise NotImplementedError

    def compute_loss(self, Q: dict[str, Tensor], D: dict[str, Tensor]) -> Tensor:
        # gather documents from all devices while preserving gradients (except `mask`).
        gather = self.gather_across_devices and dist.is_initialized()
        if gather:
            all_D = {}
            for k, v in D.items():
                if "mask" in k:
                    gathered_v = dist_gather(
                        v,
                        dist.get_rank(),
                        dist.get_world_size()
                    )
                else:
                    gathered_v = dist_gather_with_grads(
                        v,
                        dist.get_world_size()
                    )
                all_D[k] = gathered_v
        else:
            all_D = D

        # use in-batch negative sampling by default.
        scores = self.score(Q, all_D, pairwise=False)
        Nq = Q["mv_repr"].size(0)
        Nd_per_device = D["mv_repr"].size(0)
        if Nq == 0 or Nd_per_device == 0 or Nd_per_device % Nq:
            raise ValueError("Each query must have the same non-zero number of documents.")
        Nd_per_query = Nd_per_device // Nq
        labels = torch.arange(Nq, device=scores.device, dtype=torch.long) * Nd_per_query
        # Note: ensure that each device has the same number of documents.
        if gather:
            labels = labels + dist.get_rank() * Nd_per_device

        loss = nn.functional.cross_entropy(scores / self.temp, labels)
        self.loss_dict["retrieval_loss"] = loss.detach()
        if self.score_margins and scores.size(1) > 1:
            with torch.no_grad():
                positives = scores.detach().gather(1, labels[:, None]).squeeze(1)
                negatives = scores.detach().clone()
                negatives.scatter_(1, labels[:, None], -torch.inf)
                self.loss_dict["score_margin"] = (positives - negatives.amax(dim=1)).mean()
        return loss

    def forward(
        self,
        query: tuple[Tensor, Tensor],
        document: tuple[Tensor, Tensor],
        pairwise: bool = True
    ) -> EncoderOutput:
        Q = self.encode_qry(*query)
        D = self.encode_doc(*document)

        if self.training:
            self.loss_dict = {}
            loss = self.compute_loss(Q, D)

            return EncoderOutput(loss=loss)

        scores = self.score(Q=Q, D=D, pairwise=pairwise)
        return EncoderOutput(scores=scores)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseModel":
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, config: str | dict) -> "BaseModel":
        config = load_config(config)

        model = cls.from_config(config)

        if "ckpt_path" in config:
            ckpt_path = config["ckpt_path"]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file `{ckpt_path}` does not exist")

            ckpt = load_file(ckpt_path)
            ckpt.pop("bert.embeddings.position_ids", None)
            model.load_state_dict(ckpt)

        return model


class BaseTokenizer:
    def __init__(self, pretrained_model: str, qry_maxlen: int, doc_maxlen: int) -> None:
        self.tok = AutoTokenizer.from_pretrained(pretrained_model)
        self.qry_maxlen = qry_maxlen
        self.doc_maxlen = doc_maxlen

        # special tokens: [Q], [D], [SEP], [CLS], [PAD], [MASK]
        self.Q_token = "[Q]"
        self.D_token = "[D]"
        self.sep_token = self.tok.sep_token
        self.cls_token = self.tok.cls_token
        self.pad_token = self.tok.pad_token
        self.mask_token = self.tok.mask_token

        # the id corresponding to the special token
        self.Q_token_id = self.tok.convert_tokens_to_ids("[unused0]")
        self.D_token_id = self.tok.convert_tokens_to_ids("[unused1]")
        self.sep_token_id = self.tok.sep_token_id
        self.cls_token_id = self.tok.cls_token_id
        self.pad_token_id = self.tok.pad_token_id
        self.mask_token_id = self.tok.mask_token_id

    def tokenize_qry(self, texts: list[str], add_special_tokens: bool = False) -> list[list[str]]:
        raise NotImplementedError

    def tokenize_doc(self, texts: list[str], add_special_tokens: bool = False) -> list[list[str]]:
        raise NotImplementedError

    def tensorize_qry(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def tensorize_doc(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BaseTokenizer":
        raise NotImplementedError

    def save_pretrained(
        self,
        save_directory: str,
        **kwargs: Any,
    ) -> tuple[str, ...]:
        return self.tok.save_pretrained(save_directory, **kwargs)


def colbert_score(
    Q: Tensor,
    D: Tensor,
    D_mask: Tensor,
    Q_mask: Tensor | None = None,
    pairwise: bool = False
) -> Tensor:
    # Q: [Nq, Lq, D]
    # D: [Nd, Ld, D]
    # Q_mask: [Nq, Lq]
    # D_mask: [Nd, Ld]
    # out: [Nq, Nd] (pairwise = False)
    #      [Nq, Nd // Nq] (pairwise = True)
    Nq, _, dim = Q.shape
    Nd = D.size(0)

    if pairwise:
        assert Nd % Nq == 0
        N = Nd // Nq
        D = D.view(Nq, N, -1, dim)  # [Nq, N, Ld, d]
        P = torch.einsum("bid,bnjd->binj", Q.to(D.dtype), D)  # [Nq, Lq, N, Ld]
        P = P.masked_fill(~D_mask.view(Nq, 1, N, -1).bool(), value=-1e2)
    else:
        P = Q.view(-1, dim).to(D.dtype) @ D.view(-1, dim).T  # [Nq * Lq, Nd * Ld]
        P = P.reshape(*Q.shape[:2], *D.shape[:2])  # [Nq, Lq, Nd, Ld]
        P = P.masked_fill(~D_mask[None, None].bool(), value=-1e2)

    scores = P.amax(dim=-1)
    if Q_mask is not None:
        scores = (scores * Q_mask[..., None]).sum(dim=1) / Q_mask.sum(dim=-1, keepdim=True)
    else:
        scores = scores.sum(dim=1)

    return scores


def cls_score(Q: Tensor, D: Tensor, pairwise: bool = False) -> Tensor:
    Nq, Nd = Q.size(0), D.size(0)
    if pairwise:
        assert Nd % Nq == 0
        N = Nd // Nq
        D = D.reshape(Nq, N, D.size(-1))  # Q, N, d
        scores = (Q[:, None] * D).sum(-1)  # Q, N
    else:
        scores = Q @ D.T  # Q, D

    return scores
