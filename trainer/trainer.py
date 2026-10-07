import math
from typing import Any

import torch
import torch.distributed as dist
from transformers import Trainer

from dataset import RetrievalEvalDataset
from utils.dist import dist_gather


class MultiVectorTrainer(Trainer):
    def __init__(
        self,
        *args,
        gather_across_devices: bool = True,
        score_margins: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model.gather_across_devices = gather_across_devices
        self.model.score_margins = score_margins
        # Each model returns a mean metric; retain sample-weighted sums/counts.
        self._metric_totals: dict[str, torch.Tensor] = {}

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "module") or hasattr(model, "_orig_mod"):
            model = model.module if hasattr(model, "module") else model._orig_mod
        return model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs,
        )
        raw_model = self._unwrap_model(model)
        batch_size = inputs["query"][0].size(0)
        for name, value in raw_model.loss_dict.items():
            value = torch.as_tensor(value, device=loss.device).detach().float()
            totals = torch.stack((value * batch_size, value.new_tensor(batch_size)))
            if name in self._metric_totals:
                self._metric_totals[name].add_(totals)
            else:
                self._metric_totals[name] = totals
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        logs = dict(logs)
        # Flush on training logs, not on an intervening evaluation callback.
        if "loss" in logs or "train_loss" in logs:
            for name, local_totals in self._metric_totals.items():
                totals = local_totals.clone()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                logs[name] = (totals[0] / totals[1]).item()
            self._metric_totals.clear()
        super().log(logs, start_time)

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: RetrievalEvalDataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        del ignore_keys

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Trainer.evaluate requires an eval_dataset.")
        tokenizer = self.data_collator.tokenizer

        model = self._wrap_model(self.model, training=False)
        model.eval()
        encoder = model.module if hasattr(model, "module") else model

        rank, world_size = _distributed_rank_world_size()
        local_query_ids = dataset.query_ids[rank::world_size]
        batch_size = max(1, int(self.args.per_device_eval_batch_size))
        document_ids = sorted(dataset.corpus.keys())
        if not document_ids:
            raise ValueError("Cannot run full-corpus evaluation on an empty corpus.")
        batches_per_rank = math.ceil(len(document_ids) / (world_size * batch_size))
        keep = min(10, len(document_ids))
        pruning_totals = None

        # Query representations stay on their owning rank. Documents are sharded
        # across ranks, encoded locally, and then gathered for global retrieval.
        query_batches: list[tuple[list[str], Any, torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(local_query_ids), batch_size):
            batch_ids = local_query_ids[start : start + batch_size]
            query_inputs = self._prepare_input(
                tokenizer.tensorize_qry([dataset.queries[qid] for qid in batch_ids])
            )
            with self.compute_loss_context_manager():
                query_repr = encoder.encode_qry(*query_inputs)
            top_scores = torch.full(
                (len(batch_ids), keep),
                -torch.inf,
                device=self.args.device,
            )
            top_indices = torch.full(
                (len(batch_ids), keep),
                -1,
                device=self.args.device,
                dtype=torch.long,
            )
            query_batches.append((batch_ids, query_repr, top_scores, top_indices))

        local_document_indices = list(range(rank, len(document_ids), world_size))
        local_document_indices += [-1] * (
            batches_per_rank * batch_size - len(local_document_indices)
        )
        for doc_start in range(0, batches_per_rank * batch_size, batch_size):
            local_indices = local_document_indices[doc_start: doc_start + batch_size]
            document_inputs = self._prepare_input(
                tokenizer.tensorize_doc(
                    [dataset.corpus[document_ids[max(index, 0)]] for index in local_indices]
                )
            )
            with self.compute_loss_context_manager():
                local_document_repr = encoder.encode_doc(*document_inputs)

            if world_size > 1:
                document_repr = {
                    key: dist_gather(value, rank, world_size)
                    for key, value in local_document_repr.items()
                }
            else:
                document_repr = local_document_repr

            chunk_indices = torch.tensor(
                local_indices,
                device=self.args.device,
                dtype=torch.long,
            )
            pruning_stats = getattr(encoder, "eval_pruning_stats", None)
            if pruning_stats is not None:
                batch_totals = pruning_stats[chunk_indices >= 0].sum(dim=0, dtype=torch.float64)
                if pruning_totals is None:
                    pruning_totals = batch_totals
                else:
                    pruning_totals.add_(batch_totals)
            if world_size > 1:
                chunk_indices = dist_gather(chunk_indices, rank, world_size)
            for _, query_repr, top_scores, top_indices in query_batches:
                with self.compute_loss_context_manager():
                    scores = encoder.score(query_repr, document_repr, pairwise=False)
                scores = scores.float().masked_fill(chunk_indices.unsqueeze(0) < 0, -torch.inf)
                candidate_scores = torch.cat((top_scores, scores), dim=1)
                candidate_indices = torch.cat(
                    (top_indices, chunk_indices.unsqueeze(0).expand(scores.size(0), -1)),
                    dim=1,
                )
                new_scores, positions = candidate_scores.topk(keep, dim=1)
                top_scores.copy_(new_scores)
                top_indices.copy_(candidate_indices.gather(1, positions))

        local_ndcg_sum = 0.0
        for batch_ids, _, _, top_indices in query_batches:
            for row, query_id in enumerate(batch_ids):
                relevance = dataset.qrels[query_id]
                ranked_ids = [document_ids[index] for index in top_indices[row].tolist()]
                dcg = sum(
                    (2.0 ** relevance.get(document_id, 0) - 1.0)
                    / math.log2(position + 2.0)
                    for position, document_id in enumerate(ranked_ids)
                )
                ideal_relevance = sorted(relevance.values(), reverse=True)[:keep]
                idcg = sum(
                    (2.0 ** grade - 1.0) / math.log2(position + 2.0)
                    for position, grade in enumerate(ideal_relevance)
                )
                local_ndcg_sum += dcg / idcg if idcg > 0 else 0.0

        ndcg_sum, query_count = _distributed_sum_count(
            local_ndcg_sum,
            len(local_query_ids),
            device=self.args.device,
        )
        score = ndcg_sum / query_count if query_count > 0 else 0.0
        metrics = {f"{metric_key_prefix}_ndcg@10": score}
        if pruning_totals is not None:
            if world_size > 1:
                dist.all_reduce(pruning_totals, op=dist.ReduceOp.SUM)
            keep_prob_sum, drop_prob_sum, keep_count, drop_count, valid_count = pruning_totals.tolist()
            metrics.update({
                f"{metric_key_prefix}_keep_prob_mean": keep_prob_sum / max(keep_count, 1),
                f"{metric_key_prefix}_drop_prob_mean": drop_prob_sum / max(drop_count, 1),
                f"{metric_key_prefix}_gate_keep_ratio": keep_count / max(valid_count, 1),
            })
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )
        return metrics


def _distributed_rank_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _distributed_sum_count(
    local_sum: float,
    local_count: int,
    device: torch.device | str,
) -> tuple[float, int]:
    values = torch.tensor([local_sum, float(local_count)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item()), int(values[1].item())
