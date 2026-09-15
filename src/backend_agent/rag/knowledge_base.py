import asyncio
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    document_id: str
    title: str
    content: str
    score: float
    metadata: dict[str, str]


class SQLiteKnowledgeBase:
    def __init__(self, database_path: Path, source_dir: Path) -> None:
        self._database_path = database_path
        self._source_dir = source_dir

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)
        await self.index_source_directory()

    async def add_document(
        self,
        *,
        title: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._add_document_sync,
            title,
            content,
            metadata or {},
        )

    async def index_source_directory(self) -> int:
        if not self._source_dir.exists():
            return 0
        indexed = 0
        for path in sorted(self._source_dir.glob("*.md")):
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            await self.add_document(
                title=path.stem,
                content=content,
                metadata={"source": str(path)},
            )
            indexed += 1
        return indexed

    async def search(self, query: str, *, limit: int = 4) -> list[RetrievedDocument]:
        rows = await asyncio.to_thread(self._load_documents_sync)
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        document_tokens = [_tokenize(row[2]) for row in rows]
        document_frequency: Counter[str] = Counter()
        for tokens in document_tokens:
            document_frequency.update(set(tokens))

        total_documents = max(len(rows), 1)
        scored: list[RetrievedDocument] = []
        for row, tokens in zip(rows, document_tokens, strict=True):
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = counts[token]
                if not frequency:
                    continue
                inverse_frequency = math.log((total_documents + 1) / (document_frequency[token] + 1)) + 1
                score += (1 + math.log(frequency)) * inverse_frequency
            if score <= 0:
                continue
            scored.append(
                RetrievedDocument(
                    document_id=row[0],
                    title=row[1],
                    content=_best_excerpt(row[2], query),
                    score=round(score, 4),
                    metadata=json.loads(row[3]),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _initialize_sync(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._database_path, timeout=5) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _add_document_sync(self, title: str, content: str, metadata: dict[str, str]) -> str:
        identity: object = (
            {"source": metadata["source"]}
            if metadata.get("source")
            else {"title": title, "content": content, "metadata": metadata}
        )
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        with sqlite3.connect(self._database_path, timeout=5) as connection:
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                INSERT INTO documents(document_id, title, content, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(document_id) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (digest, title, content, json.dumps(metadata, ensure_ascii=False)),
            )
        return digest

    def _load_documents_sync(self) -> list[tuple[str, str, str, str]]:
        with sqlite3.connect(self._database_path, timeout=5) as connection:
            connection.execute("PRAGMA busy_timeout=5000")
            cursor = connection.execute(
                "SELECT document_id, title, content, metadata_json FROM documents LIMIT 5000"
            )
            return list(cursor.fetchall())


def _tokenize(text: str) -> list[str]:
    normalized = text.lower()
    words = re.findall(r"[a-z0-9_\-]+", normalized)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    bigrams = [chinese[index : index + 2] for index in range(max(len(chinese) - 1, 0))]
    single_chars = list(chinese) if len(chinese) == 1 else []
    return words + bigrams + single_chars


def _best_excerpt(content: str, query: str, *, max_chars: int = 1_500) -> str:
    if len(content) <= max_chars:
        return content
    lowered = content.lower()
    candidates = [token for token in _tokenize(query) if len(token) >= 2]
    positions = [lowered.find(token) for token in candidates]
    positions = [position for position in positions if position >= 0]
    center = min(positions) if positions else 0
    start = max(center - max_chars // 4, 0)
    end = min(start + max_chars, len(content))
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{content[start:end]}{suffix}"
