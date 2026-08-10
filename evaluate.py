import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore
from beir import util  # type: ignore
from beir.datasets.data_loader import GenericDataLoader  # type: ignore
import torch
from torch import Tensor
from tqdm import tqdm

from models.encoder import Encoder
from utils.utils import load_config


EVALUATION_K_VALUES = (10, 1000)
BEIR_DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a multi-vector retriever on a BEIR dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML evaluation configuration.",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        help="Show retrieval progress bar.",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Hide retrieval progress bar.",
    )
    parser.set_defaults(show_progress=None)
    return parser.parse_args()


def load_beir_dataset(
    dataset_path: Path,
    split: str,
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    dataset_path = dataset_path.expanduser()
    path_is_empty = not dataset_path.exists() or (
        dataset_path.is_dir() and not any(dataset_path.iterdir())
    )
    if path_is_empty:
        dataset_name = dataset_path.name
        if not dataset_name:
            raise ValueError("dataset_path must include the BEIR dataset name.")

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path = Path(
            util.download_and_unzip(
                BEIR_DATASET_URL.format(name=dataset_name),
                str(dataset_path.parent),
            )
        )

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(
        split=split
    )
    return corpus, queries, qrels


def _document_text(document: Mapping[str, Any]) -> str:
    title = str(document.get("title", "")).strip()
    text = str(document.get("text", "")).strip()
    return " ".join(part for part in (title, text) if part)


def _document_length(document: Mapping[str, Any]) -> int:
    """Cheaply approximate tokenized length without running the tokenizer twice."""
    title = str(document.get("title", "")).strip()
    text = str(document.get("text", "")).strip()
    return len(title) + len(text) + (len(" ") if title and text else 0)


def _to_device(batch: Mapping[str, Tensor], device: str) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _tokenize_document_chunk(
    corpus: Mapping[str, Mapping[str, Any]],
    document_ids: list[str],
    encoder: Encoder,
) -> tuple[Tensor, Tensor]:
    """Prepare one document chunk in page-locked CPU memory."""
    texts = [_document_text(corpus[document_id]) for document_id in document_ids]
    return encoder.tensorize_doc(texts, pin_memory=True)


def _prefetch_document_chunks(
    corpus: Mapping[str, Mapping[str, Any]],
    document_ids: list[str],
    encoder: Encoder,
    batch_size: int,
) -> Iterator[tuple[int, tuple[Tensor, Tensor]]]:
    """Yield tokenized chunks while preparing the next chunk in the background."""
    if batch_size <= 0:
        raise ValueError("corpus_batch_size must be positive.")

    starts = iter(range(0, len(document_ids), batch_size))
    first_start = next(starts, None)
    if first_start is None:
        return

    def submit(
        executor: ThreadPoolExecutor,
        doc_start: int,
    ) -> Future[tuple[Tensor, Tensor]]:
        chunk_ids = document_ids[doc_start : doc_start + batch_size]
        return executor.submit(
            _tokenize_document_chunk,
            corpus,
            chunk_ids,
            encoder,
        )

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="tokenizer",
    ) as executor:
        current_start = first_start
        current_future = submit(executor, current_start)

        while True:
            tokenized = current_future.result()
            next_start = next(starts, None)
            next_future = (
                submit(executor, next_start) if next_start is not None else None
            )

            # The generator remains suspended here while the consumer encodes
            # and scores this chunk, leaving next_future running concurrently.
            yield current_start, tokenized

            if next_start is None or next_future is None:
                break
            current_start = next_start
            current_future = next_future


def exact_retrieve(
    corpus: dict[str, dict[str, Any]],
    queries: dict[str, str],
    encoder: Encoder,
    top_k: int,
    query_batch_size: int,
    corpus_batch_size: int,
    cache_queries_on_device: bool = True,
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    """Return exact top-k results without materializing the full score matrix."""
    query_ids = list(queries)
    document_ids = list(corpus)

    # Stable length ordering makes each dynamic-padding batch more
    # homogeneous. Character length is used as a cheap proxy for token
    # length so the corpus does not have to be tokenized twice.
    document_ids.sort(
        key=lambda document_id: _document_length(corpus[document_id])
    )

    keep = min(top_k, len(document_ids))

    # Encode query
    query_repr = encoder.encode_qry(
        [queries[query_id] for query_id in query_ids],
        bsize=query_batch_size,
        to_cpu=not cache_queries_on_device,
    )
    device = torch.device(encoder.device)

    # Keep the running top-k as dense tensors on the scoring device. The old
    # per-query CPU cat/topk loop caused a device synchronization and many tiny
    # PyTorch operations for every query and corpus chunk.
    best_scores = torch.full(
        (len(query_ids), keep),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    best_indices = torch.full(
        (len(query_ids), keep),
        -1,
        dtype=torch.long,
        device=device,
    )

    # BEIR ignores a document whose id is identical to the query id. Resolve
    # those positions once instead of repeatedly scanning each chunk's ids.
    document_positions = {
        document_id: index for index, document_id in enumerate(document_ids)
    }
    identical_document_indices = torch.tensor(
        [document_positions.get(query_id, -1) for query_id in query_ids],
        dtype=torch.long,
        device=device,
    )

    document_chunks = _prefetch_document_chunks(
        corpus,
        document_ids,
        encoder,
        corpus_batch_size,
    )
    with tqdm(
        total=len(document_ids),
        desc="Retrieving",
        unit="doc",
        disable=not show_progress,
    ) as progress:
        for doc_start, tokenized in document_chunks:
            chunk_ids = document_ids[doc_start : doc_start + corpus_batch_size]

            # Passing the whole chunk size avoids concatenating variable-length
            # document tensors produced by "longest" tokenizer padding.
            document_repr = encoder.encode_tokenized_doc(
                tokenized,
                bsize=len(chunk_ids),
                to_cpu=False,
            )
            for query_start in range(0, len(query_ids), query_batch_size):
                query_end = min(query_start + query_batch_size, len(query_ids))
                query_chunk = {
                    key: value[query_start:query_end] for key, value in query_repr.items()
                }
                if not cache_queries_on_device:
                    query_chunk = _to_device(query_chunk, encoder.device)

                with torch.no_grad():
                    scores = encoder._model.score(
                        query_chunk,
                        document_repr,
                        pairwise=False,
                    ).float()

                local_identical_indices = (
                    identical_document_indices[query_start:query_end] - doc_start
                )
                identical_mask = (local_identical_indices >= 0) & (
                    local_identical_indices < len(chunk_ids)
                )
                local_queries = torch.arange(
                    query_end - query_start, device=device
                )[identical_mask]
                scores[
                    local_queries, local_identical_indices[identical_mask]
                ] = -torch.inf

                old_scores = best_scores[query_start:query_end]
                old_indices = best_indices[query_start:query_end]
                candidate_scores = torch.cat((old_scores, scores), dim=1)
                values, positions = torch.topk(candidate_scores, k=keep, dim=1)
                best_scores[query_start:query_end] = values
                # Positions below ``keep`` refer to the previous top-k; all other
                # positions directly encode an offset in the current corpus chunk.
                # This avoids materializing an int64 [batch, keep + chunk] tensor.
                previous_positions = positions.clamp_max(keep - 1)
                previous_indices = torch.gather(
                    old_indices, 1, previous_positions
                )
                best_indices[query_start:query_end] = torch.where(
                    positions < keep,
                    previous_indices,
                    doc_start + positions - keep,
                )

            del document_repr
            progress.update(len(chunk_ids))

    # This is the only transfer/synchronization needed for top-k bookkeeping.
    best_scores = best_scores.cpu()
    best_indices = best_indices.cpu()
    results: dict[str, dict[str, float]] = {}
    for query_index, query_id in enumerate(query_ids):
        result = {}
        for score, document_index in zip(
            best_scores[query_index].tolist(), best_indices[query_index].tolist()
        ):
            if score != float("-inf"):
                result[document_ids[document_index]] = score
        results[query_id] = result
    return results


def evaluate_with_beir(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    ndcg, _, recall, _ = EvaluateRetrieval.evaluate(
        qrels, results, list(EVALUATION_K_VALUES)
    )
    return {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@1000": recall["Recall@1000"],
    }


def run(
    config: dict[str, Any],
    show_progress: bool | None = None,
) -> dict[str, float]:
    data = config.get("data", config)
    evaluation = config.get("evaluation", config.get("eval", {}))

    query_batch_size = evaluation.get("query_batch_size", 32)
    corpus_batch_size = evaluation.get("corpus_batch_size", 32)
    if show_progress is None:
        show_progress = bool(evaluation.get("show_progress", True))
    split = str(data.get("split", "test"))
    corpus, queries, qrels = load_beir_dataset(Path(data["dataset_path"]), split)

    encoder = Encoder(config["model"])
    _synchronize(encoder.device)
    retrieval_start = time.perf_counter()
    results = exact_retrieve(
        corpus=corpus,
        queries=queries,
        encoder=encoder,
        top_k=max(EVALUATION_K_VALUES),
        query_batch_size=query_batch_size,
        corpus_batch_size=corpus_batch_size,
        cache_queries_on_device=bool(
            evaluation.get("cache_queries_on_device", True)
        ),
        show_progress=show_progress,
    )
    _synchronize(encoder.device)
    retrieval_seconds = time.perf_counter() - retrieval_start

    evaluated = evaluate_with_beir(qrels, results)
    evaluated["latency_ms_per_query"] = retrieval_seconds * 1000 / len(queries)

    output_path = evaluation.get("output_path")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(evaluated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"#> Saved evaluation metrics to {path}.")
    return evaluated


def main() -> None:
    args = parse_args()
    metrics = run(
        load_config(str(args.config)),
        show_progress=args.show_progress,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
