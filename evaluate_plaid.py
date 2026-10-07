import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import re
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from beir import util  # type: ignore
from beir.datasets.data_loader import GenericDataLoader  # type: ignore
from beir.retrieval.evaluation import EvaluateRetrieval  # type: ignore
import torch
from torch import Tensor
from tqdm import tqdm

from models.encoder import Encoder
from retrieval.plaid.index import PLAIDIndex, prepare_embeddings
from retrieval.plaid.search import PLAIDSearcher
from utils.utils import load_config


EVALUATION_K_VALUES = (10, 1000)
BEIR_DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a multi-vector retriever with PLAID."
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
        help="Show indexing and retrieval progress bars.",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Hide indexing and retrieval progress bars.",
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
    title = str(document.get("title", "")).strip()
    text = str(document.get("text", "")).strip()
    return len(title) + len(text) + (len(" ") if title and text else 0)


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _tokenize_document_chunk(
    corpus: Mapping[str, Mapping[str, Any]],
    document_ids: list[str],
    encoder: Encoder,
) -> tuple[Tensor, Tensor]:
    texts = [_document_text(corpus[document_id]) for document_id in document_ids]
    return encoder.tensorize_doc(texts, pin_memory=True)


def _prefetch_document_chunks(
    corpus: Mapping[str, Mapping[str, Any]],
    document_ids: list[str],
    encoder: Encoder,
    batch_size: int,
) -> Iterator[tuple[int, tuple[Tensor, Tensor]]]:
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
            yield current_start, tokenized

            if next_start is None or next_future is None:
                break
            current_start = next_start
            current_future = next_future


def evaluate_with_beir(
    qrels: Mapping[str, Mapping[str, int]],
    results: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    ndcg, _, recall, _ = EvaluateRetrieval.evaluate(
        qrels,
        results,
        list(EVALUATION_K_VALUES),
    )
    reciprocal_rank = 0.0
    for query_id, relevance in qrels.items():
        ranked = sorted(
            results.get(query_id, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        for rank, (document_id, _) in enumerate(ranked, start=1):
            if relevance.get(document_id, 0) > 0:
                reciprocal_rank += 1.0 / rank
                break

    return {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@1000": recall["Recall@1000"],
        "MRR@10": reciprocal_rank / len(qrels),
    }


def _default_index_name(config: Mapping[str, Any], dataset_path: Path) -> str:
    model = config["model"]
    parts = [str(model["model_name"]), dataset_path.name]
    checkpoint = model.get("ckpt_path")
    if checkpoint is not None:
        parts.append(Path(checkpoint).parent.name)
    name = "-".join(parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def create_index(
    config: Mapping[str, Any],
    dataset_path: Path,
    encoder: Encoder,
    show_progress: bool,
) -> PLAIDIndex:
    plaid = config.get("plaid", {})
    index_folder = plaid.get(
        "index_folder",
        str(Path(__file__).resolve().parent / "indexes" / "plaid"),
    )
    index_name = plaid.get(
        "index_name",
        _default_index_name(config, dataset_path),
    )
    return PLAIDIndex(
        index_folder=str(Path(index_folder).expanduser()),
        index_name=str(index_name),
        override=bool(plaid.get("override", False)),
        nbits=int(plaid.get("nbits", 4)),
        kmeans_niters=int(plaid.get("kmeans_niters", 4)),
        max_points_per_centroid=int(
            plaid.get("max_points_per_centroid", 256)
        ),
        n_samples_kmeans=plaid.get("n_samples_kmeans"),
        seed=int(plaid.get("seed", 42)),
        use_triton=plaid.get("use_triton"),
        n_ivf_probe=int(plaid.get("n_ivf_probe", 8)),
        n_full_scores=int(plaid.get("n_full_scores", 8192)),
        batch_size=int(plaid.get("batch_size", 1 << 18)),
        num_threads=plaid.get("num_threads"),
        show_progress=show_progress,
        device=plaid.get("device", encoder.device),
        low_memory=bool(plaid.get("low_memory", False)),
    )


def build_index(
    corpus: Mapping[str, Mapping[str, Any]],
    encoder: Encoder,
    index: PLAIDIndex,
    corpus_batch_size: int,
    show_progress: bool,
) -> None:
    document_ids = list(corpus)
    document_ids.sort(
        key=lambda document_id: _document_length(corpus[document_id])
    )
    document_embeddings = []
    chunks = _prefetch_document_chunks(
        corpus,
        document_ids,
        encoder,
        corpus_batch_size,
    )

    with tqdm(
        total=len(document_ids),
        desc="Encoding corpus",
        unit="doc",
        disable=not show_progress,
    ) as progress:
        for start, tokenized in chunks:
            chunk_ids = document_ids[start : start + corpus_batch_size]
            document_repr = encoder.encode_tokenized_doc(
                tokenized,
                bsize=len(chunk_ids),
                to_cpu=True,
            )
            document_embeddings.extend(
                prepare_embeddings(
                    document_repr["mv_repr"],
                    document_repr["mv_mask"],
                )
            )
            progress.update(len(chunk_ids))

    index.add_documents(document_ids, document_embeddings)


def plaid_retrieve(
    corpus: Mapping[str, Mapping[str, Any]],
    queries: Mapping[str, str],
    encoder: Encoder,
    index: PLAIDIndex,
    top_k: int,
    query_batch_size: int,
) -> dict[str, dict[str, float]]:
    query_ids = list(queries)
    query_repr = encoder.encode_qry(
        [queries[query_id] for query_id in query_ids],
        bsize=query_batch_size,
        to_cpu=True,
    )
    search_k = min(top_k + 1, len(corpus))
    search_results = PLAIDSearcher(index).search_batch(
        query_repr["mv_repr"],
        k=search_k,
        query_mask=query_repr.get("mv_mask"),
    )

    results: dict[str, dict[str, float]] = {}
    for query_id, query_result in zip(query_ids, search_results):
        ranked = {}
        for document_id, score in zip(
            query_result.document_ids,
            query_result.scores.tolist(),
        ):
            if document_id != query_id:
                ranked[document_id] = score
            if len(ranked) == top_k:
                break
        results[query_id] = ranked
    return results


def run(
    config: dict[str, Any],
    show_progress: bool | None = None,
) -> dict[str, float]:
    data = config.get("data", config)
    evaluation = config.get("evaluation", config.get("eval", {}))
    if show_progress is None:
        show_progress = bool(evaluation.get("show_progress", True))

    dataset_path = Path(data["dataset_path"]).expanduser()
    split = str(data.get("split", "test"))
    corpus, queries, qrels = load_beir_dataset(dataset_path, split)
    encoder = Encoder(config["model"])
    index = create_index(config, dataset_path, encoder, show_progress)

    indexing_seconds = 0.0
    if index.is_indexed:
        print("#> Reusing existing PLAID index.")
    else:
        _synchronize(encoder.device)
        indexing_start = time.perf_counter()
        build_index(
            corpus,
            encoder,
            index,
            corpus_batch_size=int(evaluation.get("corpus_batch_size", 32)),
            show_progress=show_progress,
        )
        _synchronize(encoder.device)
        indexing_seconds = time.perf_counter() - indexing_start

    _synchronize(encoder.device)
    retrieval_start = time.perf_counter()
    results = plaid_retrieve(
        corpus,
        queries,
        encoder,
        index,
        top_k=max(EVALUATION_K_VALUES),
        query_batch_size=int(evaluation.get("query_batch_size", 32)),
    )
    _synchronize(encoder.device)
    retrieval_seconds = time.perf_counter() - retrieval_start

    evaluated = evaluate_with_beir(qrels, results)
    evaluated["indexing_seconds"] = indexing_seconds
    evaluated["retrieval_seconds"] = retrieval_seconds
    evaluated["latency_ms_per_query"] = (
        retrieval_seconds * 1000 / len(queries)
    )

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
