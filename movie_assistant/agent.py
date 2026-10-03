from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests

from .rag import Passage, SubtitleIndex


EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
ACTION_RE = re.compile(r"\b(send|email|e-mail|share|forward)\b", re.I)


@dataclass
class Result:
    kind: str
    text: str
    citations: list[str]
    passages: list[Passage]
    recipient: str | None = None
    subject: str | None = None
    email_body: str | None = None


def _llm_answer(question: str, passages: list[Passage]) -> str | None:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    context = "\n\n".join(f"[{p.citation}]\n{p.text}" for p in passages)
    try:
        response = requests.post(
            f"{os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": os.getenv("LLM_MODEL", "gpt-4o-mini"), "temperature": 0.2,
                  "messages": [
                      {"role": "system", "content": "Answer only from the subtitle excerpts. State when they do not establish a fact. Cite every factual claim inline with the supplied [Movie — start–end] citation. Do not invent plot context."},
                      {"role": "user", "content": f"Question: {question}\n\nSubtitle excerpts:\n{context}"},
                  ]}, timeout=45,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError):
        return None


def answer_question(question: str, passages: list[tuple[Passage, float]]) -> Result:
    selected = [p for p, _ in passages]
    if not selected:
        return Result("answer", "I couldn't find a relevant passage in the subtitles currently loaded. Try a character name, quote fragment, or a more specific question.", [], [])
    response = _llm_answer(question, selected)
    if response is None:
        excerpts = "\n\n".join(f"“{p.text}”\n— {p.citation}" for p in selected[:3])
        response = ("The closest subtitle passages I found are:\n\n" + excerpts +
                    "\n\nThese excerpts provide dialogue evidence; subtitle text alone may not establish the full plot context.")
    citations = list(dict.fromkeys(p.citation for p in selected))
    # Keep citations visible in the answer itself even if an optional model omits them.
    if citations and not any(citation in response for citation in citations):
        response += "\n\nSources: " + "; ".join(citations)
    return Result("answer", response, citations, selected)


def route_request(question: str, index: SubtitleIndex, top_k: int = 5) -> Result:
    """Route info queries to retrieval and email requests through clarification first."""
    if not index.passages:
        return Result("clarify", "Upload one or more .srt files first so I can search the dialogue.", [], [])
    if ACTION_RE.search(question):
        recipient_matches = EMAIL_RE.findall(question)
        if not recipient_matches:
            # Retrieve the requested information now, but do not guess a recipient.
            movie, ambiguity = index.resolve_movie(question)
            if ambiguity:
                return Result("clarify", "I found multiple possible movies: " + ", ".join(ambiguity) + ". Which one should I use?", [], [])
            ranked = index.retrieve(question, k=top_k, movie=movie)
            base = answer_question(question, ranked)
            if not base.passages:
                return Result("clarify", "Which movie or scene should I include, and what email address should receive it?", [], [])
            return Result("clarify", "I found the relevant dialogue. What email address should receive this summary? Reply with the address (and optionally a different movie/scene request).\n\n" + base.text,
                          base.citations, base.passages)
        recipient = recipient_matches[-1]
        movie, ambiguity = index.resolve_movie(question)
        if ambiguity:
            return Result("clarify", "I found multiple possible movies: " + ", ".join(ambiguity) + ". Which one should I use?", [], [], recipient)
        ranked = index.retrieve(question, k=top_k, movie=movie)
        base = answer_question(question, ranked)
        if not base.passages:
            return Result("clarify", "I couldn't identify a relevant scene to email. Which movie, character, or quote should I look for?", [], [], recipient)
        title = f"Movie dialogue summary: {base.passages[0].movie}"
        body = base.text + "\n\nSources:\n" + "\n".join(f"- {p.citation}" for p in base.passages)
        return Result("email", "I prepared an email with the retrieved scene summary and timestamp citations. Review the content below; sending now.",
                      base.citations, base.passages, recipient, title, body)
    movie, ambiguity = index.resolve_movie(question)
    if ambiguity:
        return Result("clarify", "Which movie did you mean? Possible matches: " + ", ".join(ambiguity), [], [])
    return answer_question(question, index.retrieve(question, k=top_k, movie=movie))
