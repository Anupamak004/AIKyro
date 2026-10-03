# Movie Intelligence & Follow-up Assistant

A Streamlit app for searching dialogue in up to 100 SRT subtitle files, answering questions with timestamp citations, and sending scene summaries through an MCP email tool.

## Features

- Parses SRT cues including comma/dot millisecond timecodes, cue settings, HTML tags, and multiline dialogue.
- Groups nearby cues into passages while preserving each passage's movie, exact timestamp range, and cue range.
- Uses a sparse TF-IDF vector index with unigram/bigram matching and cosine similarity.
- Routes informational questions to retrieval; email requests retrieve evidence first, ask for missing recipient/movie details, and call an MCP stdio tool when ready.
- Optional OpenAI-compatible grounded answer generation. Without an API key, it returns the closest retrieved dialogue with citations.
- The email MCP server dispatches with SMTP and reports configuration or delivery errors.

## Run locally

Python 3.10+ is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

Upload `.srt` files in the sidebar. Filenames become movie titles; use recognizable names such as `The Matrix.srt`. The uploaded files are kept in the Streamlit session and are not written to disk.

### Optional grounded answer generation

Set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in `.env` for an OpenAI-compatible Chat Completions endpoint. The assistant supplies only retrieved passages to the model and requests timestamp citations. Dialogue-only fallback works without an LLM.

### MCP email setup

Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, and `SMTP_USE_TLS` in `.env`. Credentials should be an app password or other SMTP credential, not a mailbox's normal login password. The Streamlit app starts `movie_assistant.email_mcp_server` over the official MCP Python SDK stdio transport and invokes its `send_email` tool. The SMTP tool validates the recipient and fails closed if required settings are absent.

The MCP server can also be run directly for an MCP host:

```powershell
python -m movie_assistant.email_mcp_server
```

## Project layout

```text
app.py                         Streamlit UI and request handling
movie_assistant/srt.py         SRT parsing, cleaning, passage grouping
movie_assistant/rag.py         Metadata-aware sparse-vector retrieval
movie_assistant/agent.py       Intent routing, ambiguity checks, grounded answer generation
movie_assistant/mcp_client.py  MCP stdio client used by the UI
movie_assistant/email_mcp_server.py  MCP SMTP email tool
```

## Notes

- Citations refer to subtitle timecodes and can be displayed as `Movie Title — HH:MM:SS.mmm–HH:MM:SS.mmm`.
- Subtitle-only evidence may not contain enough information to answer broad plot questions. The assistant should say when the retrieved passages do not establish an answer.
- Email requests require an unambiguous movie/scene and a recipient address. If the address is omitted, the retrieved summary is held in the current session while the assistant asks for the address.
- The current index is in memory; re-upload files after starting a new session.
