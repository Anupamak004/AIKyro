from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .srt import Cue, group_cues, parse_srt


@dataclass(frozen=True)
class Passage:
    movie: str
    start: str
    end: str
    text: str
    characters: tuple[str, ...]
    cue_start: int
    cue_end: int

    @property
    def citation(self) -> str:
        return f"{self.movie} — {self.start}–{self.end}"


class SubtitleIndex:
    """A sparse-vector index that retains cue boundaries and movie metadata."""

    def __init__(self) -> None:
        self.passages: list[Passage] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    @staticmethod
    def _movie_title(filename: str) -> str:
        title = Path(filename).stem
        title = re.sub(r"[._]+", " ", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title or "Untitled movie"

    def add_srt(self, filename: str, content: str) -> int:
        title = self._movie_title(filename)
        cues = parse_srt(content)
        if not cues:
            return 0
        for group in group_cues(cues):
            text = " ".join(cue.text for cue in group)
            # Speaker labels are common in SRTs; keep them in the passage and extract them for context.
            speakers = tuple(dict.fromkeys(re.findall(r"(?:^|\s)([A-Z][A-Z0-9 _'-]{1,28}):(?=\s)", text)))
            self.passages.append(
                Passage(title, group[0].start, group[-1].end, text, speakers,
                        group[0].index, group[-1].index)
            )
        self._rebuild()
        return len(cues)

    def add_files(self, files: Iterable[tuple[str, str]]) -> dict[str, int]:
        result = {}
        for filename, content in files:
            result[filename] = self.add_srt(filename, content)
        return result

    def _rebuild(self) -> None:
        if not self.passages:
            self.vectorizer = self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            sublinear_tf=True, stop_words="english", max_features=120_000,
        )
        self.matrix = self.vectorizer.fit_transform(
            f"{p.movie} {p.text} {' '.join(p.characters)}" for p in self.passages
        )

    @property
    def movie_titles(self) -> list[str]:
        return sorted({p.movie for p in self.passages})

    def resolve_movie(self, query: str) -> tuple[str | None, list[str]]:
        """Return a unique title mentioned in a query or likely title candidates."""
        q = query.casefold()
        exact = [title for title in self.movie_titles if title.casefold() in q]
        if len(exact) == 1:
            return exact[0], []
        if len(exact) > 1:
            return None, exact
        words = re.findall(r"[\w'-]+", q)
        # Match title phrases even when punctuation differs.
        normalized_query = " ".join(words)
        normalized_titles = {
            title: " ".join(re.findall(r"[\w'-]+", title.casefold())) for title in self.movie_titles
        }
        partial = [title for title, normalized in normalized_titles.items()
                   if normalized and (normalized in normalized_query or normalized_query in normalized)]
        if len(partial) == 1 and len(normalized_titles[partial[0]]) > 3:
            return partial[0], []
        return None, partial if len(partial) > 1 else []

    def retrieve(self, query: str, k: int = 5, movie: str | None = None) -> list[tuple[Passage, float]]:
        if not self.passages or self.vectorizer is None or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        candidates = [(i, score) for i, score in enumerate(scores)
                      if score > 0 and (movie is None or self.passages[i].movie == movie)]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return [(self.passages[i], float(score)) for i, score in candidates[:k]]

    def movies_with_cue_count(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for passage in self.passages:
            counts[passage.movie] = counts.get(passage.movie, 0) + (passage.cue_end - passage.cue_start + 1)
        return sorted(counts.items())
