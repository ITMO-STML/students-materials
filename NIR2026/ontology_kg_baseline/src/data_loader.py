import json


class DatasetLoader:
    def __init__(self, path: str, limit: int | None = None):
        self.path = path
        self.limit = limit

    def load(self) -> list[dict]:
        items = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                items.append(json.loads(line))

                if self.limit is not None and len(items) >= self.limit:
                    break

        return items