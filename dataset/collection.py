from utils.loader import load_queries, load_corpus


class Collection:
    def __init__(self, path: str, is_query: bool = False) -> None:
        self.path = path
        self.data: dict[str, str] = self._load_file(path, is_query)

    def _load_file(self, path: str, is_query: bool = False) -> dict[str, str]:
        return load_queries(path) if is_query else load_corpus(path)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, id: str) -> str:
        return self.data[id]

    def __len__(self) -> int:
        return len(self.data)