import random
from typing import Any
from collections import namedtuple

from dataset.collection import Collection
from utils.loader import load_qrels


Triple = namedtuple(
    "triple",
    ["query_id", "positive_id", "negative_ids"]
)

RerankTriple = namedtuple(
    "rerank_triple",
    ["query_id", "doc_ids", "scores"]
)


class RetrievalEvalDataset:
    """Queries, corpus, and relevance judgments for full-corpus evaluation."""

    def __init__(
        self,
        queries: str | Collection,
        corpus: str | Collection,
        qrels: str | dict[str, dict[str, int]],
    ) -> None:
        if isinstance(queries, str):
            self.queries = Collection(queries, is_query=True)
        elif isinstance(queries, Collection):
            self.queries = queries
        else:
            raise TypeError("queries should be `str` or `Collection`")

        if isinstance(corpus, str):
            self.corpus = Collection(corpus, is_query=False)
        elif isinstance(corpus, Collection):
            self.corpus = corpus
        else:
            raise TypeError("corpus should be `str` or `Collection`")

        if isinstance(qrels, str):
            self.qrels = load_qrels(qrels)
        elif isinstance(qrels, dict):
            self.qrels = qrels
        else:
            raise TypeError("qrels should be a path or a nested dictionary")

        self.query_ids = list(self.qrels)
        missing_queries = set(self.query_ids).difference(self.queries.keys())
        if missing_queries:
            raise ValueError(
                f"Qrels contain {len(missing_queries)} query ids absent from queries."
            )

    def __len__(self) -> int:
        return len(self.query_ids)


class TriplesDataset:
    def __init__(
        self,
        triples: str,
        queries: str | Collection,
        corpus: str | Collection,
        nway: int,
        random_sample: bool = False,
        seed: int | None = None,
    ) -> None:
        self.nway = nway
        self.random = random.Random(seed) if random_sample else None

        if isinstance(queries, str):
            self.queries = Collection(queries, is_query=True)
        elif isinstance(queries, Collection):
            self.queries = queries
        else:
            raise TypeError("queries should be `str` or `Collection`")

        if isinstance(corpus, str):
            self.corpus = Collection(corpus, is_query=False)
        elif isinstance(corpus, Collection):
            self.corpus = corpus
        else:
            raise TypeError("corpus should be `str` or `Collection`")

        self.triples = self._load_triples(triples)

    def _load_triples(self, path: str) -> list[Triple]:
        triples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                qid, pos_id, *neg_ids = line.rstrip("\n\r").split("\t")
                if len(neg_ids) < self.nway - 1:
                    raise ValueError(f"Line {line_no}: need at least {self.nway - 1} negatives.")
                triples.append(Triple(qid, pos_id, neg_ids))
        return triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        triple = self.triples[index]

        if self.random is not None:
            negative_ids = self._sample_negatives(triple.negative_ids)
        else:
            negative_ids = triple.negative_ids[:self.nway - 1]

        document_ids = [triple.positive_id, *negative_ids]

        return {
            "query": self.queries[triple.query_id],
            "document": [self.corpus[id] for id in document_ids],
        }

    def _sample_negatives(self, negative_ids: list[str]) -> list[str]:
        sample_size = self.nway - 1
        if len(negative_ids) < sample_size:
            raise ValueError(f"Need at least {sample_size} negatives, got {len(negative_ids)}.")

        if len(negative_ids) == sample_size:
            return list(negative_ids)

        return self.random.sample(negative_ids, sample_size)


class RerankDataset:
    def __init__(
        self,
        triples: str,
        queries: str | Collection,
        corpus: str | Collection,
        nway: int
    ) -> None:
        if nway < 2:
            raise ValueError("nway must be at least 2.")

        self.nway = nway

        if isinstance(queries, str):
            self.queries = Collection(queries, is_query=True)
        elif isinstance(queries, Collection):
            self.queries = queries
        else:
            raise TypeError("queries should be `str` or `Collection`")

        if isinstance(corpus, str):
            self.corpus = Collection(corpus, is_query=False)
        elif isinstance(corpus, Collection):
            self.corpus = corpus
        else:
            raise TypeError("corpus should be `str` or `Collection`")

        self.triples = self._load_triples(triples)

    def _load_triples(self, path: str) -> list[Triple]:
        triples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                qid, *pairs = line.rstrip("\n\r").split("\t")

                if len(pairs) % 2 != 0:
                    raise ValueError(f"Line {line_no}: did-score pairs are incomplete.")

                if len(pairs) // 2 < self.nway:
                    raise ValueError(f"Line {line_no}: need at least {self.nway} pairs.")

                doc_ids = pairs[0::2][:self.nway]
                scores = [float(score) for score in pairs[1::2][:self.nway]]
                triples.append(RerankTriple(qid, doc_ids, scores))

        return triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        triple = self.triples[index]

        return {
            "query": self.queries[triple.query_id],
            "document": [self.corpus[id] for id in triple.doc_ids],
            "label": triple.scores
        }
