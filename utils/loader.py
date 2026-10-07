import json


def _load_from_tsv(record: str) -> list[str]:
    return record.strip().split("\t")


def _load_from_json(record: str) -> dict:
    return json.loads(record)


def load_queries(path: str) -> dict[str, str]:
    print(f"#> Loading the queries from {path} ...")

    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if path.endswith(".tsv"):
                qid, query, *_ = _load_from_tsv(line)
            elif path.endswith(".jsonl"):
                record = _load_from_json(line)
                qid, query = record["_id"], record["text"]
            else:
                raise ValueError("Invalid file type. Only `tsv` and `jsonl` are supported.")
            queries[qid] = query

    print(f"#> Got {len(queries)} queries.")

    return queries


def load_corpus(path: str) -> dict[str, str]:
    print(f"#> Loading the documents from {path} ...")

    documents = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if path.endswith(".tsv"):
                did, document, *rest = _load_from_tsv(line)
                if len(rest) >= 1:
                    document = (rest[0] + " " + document).strip()  # combine the title and text.
            elif path.endswith(".jsonl"):
                record = _load_from_json(line)
                did, document = record["_id"], record["text"]
                if "title" in record:
                    document = (record["title"] + " " + document).strip()
            else:
                raise ValueError("Invalid file type. Only `tsv` and `jsonl` are supported.")
            documents[did] = document

    print(f"#> Got {len(documents)} documents.")

    return documents


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    print(f"#> Loading qrels from {path} ...")

    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items = _load_from_tsv(line)
            if len(items) == 3:
                qid, did, score = items
            elif len(items) == 4:
                qid, _, did, score = items
            else:
                raise ValueError("Invalid input format: too many elements to parse into qid, did, and score.")

            if score == "score":  # skip header
                continue

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = int(score)

    print(f"#> Loaded qrels for {len(qrels)} unique queries.")

    return qrels