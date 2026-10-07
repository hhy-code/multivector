import random
from typing import Any
from collections import namedtuple

from dataset.collection import Collection
from utils.loader import load_qrels


Triple = namedtuple(
    "triple",
    ["query_id", "positive_id", "negative_ids"]
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
    ) -> None:
        self.nway = nway
        if nway < 1:
            raise ValueError("nway must be at least 1.")

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

        # negative_ids = triple.negative_ids[:self.nway - 1]
        negative_ids = random.sample(triple.negative_ids, k=self.nway - 1)

        document_ids = [triple.positive_id, *negative_ids]

        return {
            "query": self.queries[triple.query_id],
            "document": [self.corpus[id] for id in document_ids],
        }
