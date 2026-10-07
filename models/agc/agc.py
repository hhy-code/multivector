import math
import os
from typing import Any

from safetensors.torch import load_file
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.colbert import ColBERT
from models.factories import register_model
from utils.utils import load_config


@register_model("agc")
class AGC(ColBERT):
    """Attention-Guided Clustering with learned universal queries."""

    def __init__(
        self,
        pretrained_model: str,
        dim: int,
        num_repr_vectors: int = 32,
        num_universal_queries: int = 32,
        temp: float | None = None,
    ) -> None:
        super().__init__(pretrained_model, dim, temp=temp)

        self.num_repr_vectors = num_repr_vectors
        self.num_universal_queries = num_universal_queries
        self.universal_queries = nn.Parameter(
            torch.empty(num_universal_queries, self.bert.config.hidden_size)
        )
        nn.init.normal_(
            self.universal_queries,
            mean=0.0,
            std=self.bert.config.initializer_range,
        )

    def _append_universal_queries(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        token_embeddings = self.bert.get_input_embeddings()(input_ids)
        B, L, hidden_size = token_embeddings.shape
        U = self.num_universal_queries

        positions = torch.arange(L + U, device=input_ids.device)[None]
        insert_positions = attention_mask.sum(dim=1, keepdim=True) - 1
        universal_mask = (
            (positions >= insert_positions)
            & (positions < insert_positions + U)
        )

        source_positions = torch.where(
            positions < insert_positions,
            positions,
            positions - U,
        ).clamp_(0, L - 1)
        source_positions = source_positions.expand(B, -1)
        expanded_embeddings = token_embeddings.gather(
            1,
            source_positions.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        expanded_mask = attention_mask.gather(1, source_positions)

        universal_indices = (positions - insert_positions).clamp_(0, U - 1)
        universal_embeddings = self.universal_queries[universal_indices]
        expanded_embeddings = torch.where(
            universal_mask.unsqueeze(-1),
            universal_embeddings,
            expanded_embeddings,
        )
        expanded_mask = torch.where(
            universal_mask,
            torch.ones_like(expanded_mask),
            expanded_mask,
        )
        document_mask = expanded_mask.bool() & ~universal_mask
        return expanded_embeddings, expanded_mask, document_mask, universal_mask

    def _encode_with_last_layer_input(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        last_layer = self.bert.encoder.layer[-1]
        layer_input: list[Tensor] = []

        def save_input(module: nn.Module, args: tuple[Tensor, ...]) -> None:
            layer_input.append(args[0])

        handle = last_layer.register_forward_pre_hook(save_input)
        try:
            hidden = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )[0]
        finally:
            handle.remove()

        return hidden, layer_input[0]

    def _saliency_scores(
        self,
        layer_input: Tensor,
        attention_mask: Tensor,
        document_mask: Tensor,
        universal_mask: Tensor,
    ) -> Tensor:
        B, L, _ = layer_input.shape
        self_attention = self.bert.encoder.layer[-1].attention.self
        num_heads = self_attention.num_attention_heads
        head_dim = self_attention.attention_head_size

        universal_hidden = layer_input[universal_mask].view(
            B,
            self.num_universal_queries,
            -1,
        )
        queries = self_attention.query(universal_hidden)
        keys = self_attention.key(layer_input)
        queries = queries.view(B, -1, num_heads, head_dim).transpose(1, 2)
        keys = keys.view(B, L, num_heads, head_dim).transpose(1, 2)

        attention = torch.matmul(queries, keys.transpose(-1, -2))
        attention = attention / math.sqrt(head_dim)
        attention = attention.masked_fill(
            ~attention_mask[:, None, None].bool(),
            torch.finfo(attention.dtype).min,
        )
        attention = attention.softmax(dim=-1)
        attention = self_attention.dropout(attention).mean(dim=1)
        attention = attention.masked_fill(
            ~document_mask[:, None],
            torch.finfo(attention.dtype).min,
        )
        return attention.softmax(dim=-1).mean(dim=1)

    def _attention_guided_clustering(
        self,
        D: Tensor,
        D_mask: Tensor,
        saliency: Tensor,
    ) -> dict[str, Tensor]:
        B, L, dim = D.shape
        m = self.num_repr_vectors

        saliency = saliency.masked_fill(~D_mask, -torch.inf)
        centroid_indices = saliency.topk(m, dim=1, sorted=False).indices
        centroid_mask = D_mask.gather(1, centroid_indices)
        centroids = D.gather(
            1,
            centroid_indices.unsqueeze(-1).expand(-1, -1, dim),
        )

        similarity = torch.bmm(
            F.normalize(D, p=2, dim=-1),
            F.normalize(centroids, p=2, dim=-1).transpose(1, 2),
        )
        similarity = similarity.masked_fill(~centroid_mask[:, None], -torch.inf)
        assignments = similarity.argmax(dim=-1)

        valid = D_mask.bool()
        batch_indices = torch.arange(B, device=D.device)[:, None].expand(B, L)
        global_assignments = (batch_indices * m + assignments)[valid]
        weights = saliency.masked_fill(~valid, 0)[valid].to(D.dtype)

        pooled = D.new_zeros(B * m, dim)
        pooled.index_add_(0, global_assignments, D[valid] * weights[:, None])
        weight_sums = D.new_zeros(B * m)
        weight_sums.index_add_(0, global_assignments, weights)
        pooled = pooled / weight_sums.clamp_min(1e-8).unsqueeze(-1)
        pooled = F.normalize(pooled.view(B, m, dim), p=2, dim=-1)

        return {
            "mv_repr": pooled,
            "mv_mask": centroid_mask,
        }

    def encode_doc(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        inputs_embeds, expanded_mask, document_mask, universal_mask = (
            self._append_universal_queries(input_ids, attention_mask)
        )
        hidden, layer_input = self._encode_with_last_layer_input(
            inputs_embeds,
            expanded_mask,
        )
        saliency = self._saliency_scores(
            layer_input,
            expanded_mask,
            document_mask,
            universal_mask,
        )
        D = F.normalize(self.linear(hidden), p=2, dim=-1)
        return self._attention_guided_clustering(D, document_mask, saliency)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AGC":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        dim = config.get("dim", 128)
        num_repr_vectors = config.get("num_repr_vectors", 32)
        num_universal_queries = config.get("num_universal_queries", 32)
        temp = config.get("temp")

        return cls(pretrained_model=pretrained_model, dim=dim, num_repr_vectors=num_repr_vectors, num_universal_queries=num_universal_queries, temp=temp)

    @classmethod
    def from_pretrained(cls, config: str | dict) -> "AGC":
        config = load_config(config)
        model = cls.from_config(config)

        if "ckpt_path" in config:
            ckpt_path = config["ckpt_path"]
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file `{ckpt_path}` does not exist")

            checkpoint = load_file(ckpt_path)
            checkpoint.pop("bert.embeddings.position_ids", None)
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            if set(missing) - {"universal_queries"} or unexpected:
                raise RuntimeError(
                    f"Missing keys: {missing}; unexpected keys: {unexpected}"
                )

        return model
