from __future__ import annotations

import hashlib
import os
import re

import streamlit as st
from dotenv import load_dotenv

from movie_assistant.agent import EMAIL_RE, Result, route_request
from movie_assistant.mcp_client import send_email
from movie_assistant.rag import SubtitleIndex

load_dotenv()
st.set_page_config(page_title="Movie Intelligence Assistant", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Building subtitle search index…")
def build_index(files: tuple[tuple[str, str], ...]) -> SubtitleIndex:
    index = SubtitleIndex()
    index.add_files(files)
    return index


def display_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Sources"):
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")


st.title("🎬 Movie Intelligence & Follow-up Assistant")
st.caption("Upload subtitle files, ask about dialogue or scenes, and email sourced summaries through the MCP email tool.")

with st.sidebar:
    st.header("Subtitle library")
    uploads = st.file_uploader(
        "Upload .srt files (up to 100 movies)", type=["srt"], accept_multiple_files=True,
        help="Name each file after its movie, for example The Matrix.srt.",
    )
    if uploads and len(uploads) > 100:
        st.error("Please upload no more than 100 subtitle files.")
        uploads = uploads[:100]
    upload_data: tuple[tuple[str, str], ...] = tuple(
        (item.name, item.getvalue().decode("utf-8-sig", errors="replace")) for item in (uploads or [])
    )
    signature = hashlib.sha256("".join(name + content for name, content in upload_data).encode("utf-8")).hexdigest()
    index = build_index(upload_data) if upload_data else SubtitleIndex()
    if index.passages:
        st.success(f"Indexed {len(index.movie_titles)} movie(s) · {len(index.passages)} dialogue passages")
        with st.expander("Movies in this session"):
            for title, cues in index.movies_with_cue_count():
                st.write(f"{title} · {cues} cues")
    else:
        st.info("Upload subtitle files to begin. Files are processed in this session.")

    if os.getenv("LLM_API_KEY"):
        st.caption(f"Answer generation: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
    else:
        st.caption("Answer mode: retrieved dialogue excerpts. Configure an optional LLM API key for grounded synthesis.")
    smtp_ready = all(os.getenv(key) for key in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD"))
    st.caption("MCP email: configured" if smtp_ready else "MCP email: configure SMTP in .env before sending")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload subtitles, then ask me about a movie, character interaction, or quote. You can also ask me to email a scene summary."}]
if "pending_email" not in st.session_state:
    st.session_state.pending_email = None

for message in st.session_state.messages:
    display_message(message)

prompt = st.chat_input("Ask about a movie or request an email")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message(st.session_state.messages[-1])
    with st.chat_message("assistant"):
        with st.spinner("Searching subtitle dialogue…"):
            pending = st.session_state.pending_email
            address = EMAIL_RE.search(prompt)
            if pending and address:
                result = Result("email", "Sending the prepared scene summary through the MCP email tool.",
                                pending["citations"], pending["passages"], address.group(0),
                                pending["subject"], pending["body"])
                st.session_state.pending_email = None
            else:
                if pending:
                    st.session_state.pending_email = None
                result = route_request(prompt, index)

            if result.kind == "email":
                try:
                    sent = send_email(result.recipient or "", result.subject or "Movie dialogue summary", result.email_body or result.text)
                    response = f"{sent}\n\n{result.text}"
                except Exception as exc:
                    response = f"I couldn't send the email: {exc}\n\nThe prepared content is below so you can retry after fixing the MCP/SMTP configuration:\n\n{result.email_body or result.text}"
            else:
                response = result.text
                if result.kind == "clarify" and "What email address" in result.text and result.passages:
                    body = result.text.split("\n\n", 1)[-1]
                    body += "\n\nSources:\n" + "\n".join(f"- {p.movie} — {p.start}–{p.end}" for p in result.passages)
                    st.session_state.pending_email = {
                        "subject": f"Movie dialogue summary: {result.passages[0].movie}",
                        "body": body, "citations": result.citations, "passages": result.passages,
                    }
        st.markdown(response)
        if result.citations:
            with st.expander("Sources"):
                for citation in result.citations:
                    st.markdown(f"- {citation}")
    st.session_state.messages.append({"role": "assistant", "content": response, "citations": result.citations})
