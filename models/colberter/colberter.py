from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel

from models.base import BaseModel, cls_score, colbert_score
from models.factories import register_model


@register_model("colberter")
class ColBERTer(BaseModel):
    def __init__(
        self,
        pretrained_model: str,
        cls_dim: int,
        dim: int,
        use_contextualized_stopwords: bool = True,
        stopword_loss_weight: float = 0.5,
        temp: float | None = None,
    ) -> None:
        super().__init__(temp=temp)

        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.cls_linear = nn.Linear(self.bert.config.hidden_size, cls_dim)
        self.token_linear = nn.Linear(self.bert.config.hidden_size, dim)
        self.score_merger = nn.Parameter(torch.zeros(1))

        self.use_contextualized_stopwords = use_contextualized_stopwords
        self.stopword_loss_weight = stopword_loss_weight
        if use_contextualized_stopwords:
            self.stopword_reducer = nn.Linear(dim, 1)
            nn.init.constant_(self.stopword_reducer.bias, 1.0)

    def _aggregate_words(
        self,
        token_repr: Tensor,
        word_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        B, _, dim = token_repr.shape
        n_words = word_ids.size(1)
        word_mask = word_ids > 0
        indices = (word_ids - 1).clamp_min(0)

        word_repr = token_repr.new_zeros(B, n_words, dim)
        word_repr.scatter_add_(
            1,
            indices[..., None].expand(-1, -1, dim),
            token_repr * word_mask[..., None],
        )

        counts = token_repr.new_zeros(B, n_words)
        counts.scatter_add_(1, indices, word_mask.to(token_repr.dtype))
        word_repr = word_repr / counts.clamp_min(1)[..., None]

        return word_repr, counts > 0

    def _encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        is_document: bool,
    ) -> dict[str, Tensor]:
        token_ids = input_ids[:, 0]
        word_ids = input_ids[:, 1]

        hidden = self.bert(token_ids, attention_mask=attention_mask)[0]
        cls_repr = self.cls_linear(hidden[:, 0])
        token_repr = self.token_linear(hidden)
        token_repr, token_mask = self._aggregate_words(token_repr, word_ids)

        if is_document and self.use_contextualized_stopwords:
            importance = F.relu(self.stopword_reducer(token_repr)).squeeze(-1)
            token_repr = token_repr * importance[..., None]
            self.stopword_loss = importance[token_mask].mean()

            if not self.training:
                token_mask = token_mask & (importance > 0)

        return {
            "cls_repr": cls_repr,
            "mv_repr": token_repr,
            "mv_mask": token_mask,
        }

    def encode_qry(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        return self._encode(input_ids, attention_mask, is_document=False)

    def encode_doc(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        return self._encode(input_ids, attention_mask, is_document=True)

    def compute_loss(self, Q: dict[str, Tensor], D: dict[str, Tensor]) -> Tensor:
        loss = super().compute_loss(Q, D)
        if self.use_contextualized_stopwords and self.stopword_loss_weight > 0:
            self.loss_dict["stopword_loss"] = self.stopword_loss.detach()
            loss = loss + self.stopword_loss_weight * self.stopword_loss
        return loss

    def score(
        self,
        Q: dict[str, Tensor],
        D: dict[str, Tensor],
        pairwise: bool = True,
    ) -> Tensor:
        cls_scores = cls_score(
            Q["cls_repr"],
            D["cls_repr"],
            pairwise=pairwise,
        )
        token_scores = colbert_score(
            Q=Q["mv_repr"],
            D=D["mv_repr"],
            D_mask=D["mv_mask"],
            pairwise=pairwise,
        )

        weight = torch.sigmoid(self.score_merger)
        return weight * cls_scores + (1 - weight) * token_scores

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ColBERTer":
        pretrained_model = config.get("pretrained_model", "bert-base-uncased")
        cls_dim = config.get("cls_dim", 128)
        dim = config.get("dim", 32)
        use_contextualized_stopwords = config.get(
            "use_contextualized_stopwords", True
        )
        stopword_loss_weight = config.get("stopword_loss_weight", 0.5)
        temp = config.get("temp")

        return cls(pretrained_model, cls_dim, dim, use_contextualized_stopwords, stopword_loss_weight, temp)
