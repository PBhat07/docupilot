from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import streamlit as st

API_URL = os.getenv("DOCUPILOT_API_URL", "http://api:8000").rstrip("/")
TIMEOUT = httpx.Timeout(120.0, connect=5.0)
PAGE_SIZE = 100

st.set_page_config(
    page_title="DocuPilot | Documentation Workspace",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1320px; padding-top: 1.6rem; padding-bottom: 3rem;}
      .dp-hero {padding: 1.25rem 1.5rem; border: 1px solid rgba(128,128,128,.28);
        border-radius: 16px; margin-bottom: 1rem;}
      .dp-muted {color: rgba(128,128,128,.95); font-size: .92rem;}
      a {overflow-wrap: anywhere;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_submitted_job" not in st.session_state:
    st.session_state.last_submitted_job = None
if "active_view" not in st.session_state:
    st.session_state.active_view = "Documentation Assistant"


def api(method: str, path: str, **kwargs: Any) -> Any | None:
    """Call the DocuPilot API and show a useful, non-sensitive error."""
    try:
        with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
            detail = body.get("detail", body)
        except ValueError:
            detail = exc.response.text[:1000]
        st.error(f"API request failed ({exc.response.status_code}): {detail}")
    except httpx.RequestError as exc:
        st.error(
            f"Cannot reach DocuPilot API at {API_URL}. "
            f"Check that the API and its dependencies are running. ({exc})"
        )
    return None


def as_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def pretty_time(value: Any) -> str:
    if not value:
        return "—"
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo:
                parsed = parsed.astimezone()
            return parsed.strftime("%Y-%m-%d %H:%M %Z").strip()
        except ValueError:
            return value
    return str(value)


def status_label(value: Any) -> str:
    value = str(value or "unknown").lower()
    icons = {
        "completed": "✅",
        "finished": "✅",
        "success": "✅",
        "succeeded": "✅",
        "running": "🔄",
        "started": "🔄",
        "queued": "🕒",
        "deferred": "🕒",
        "failed": "❌",
        "stopped": "⏹️",
        "registered": "⚪",
    }
    return f"{icons.get(value, '•')} {value.replace('_', ' ').title()}"


def safe_external_link(url: Any) -> bool:
    if not isinstance(url, str):
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def render_citations(citations: Any) -> None:
    """Render only citation metadata returned by the answer API."""
    if not isinstance(citations, list) or not citations:
        st.caption("No page-level citations were returned for this answer.")
        return

    st.markdown(f"**Sources used for this answer · {len(citations)} reference(s)**")
    for index, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            continue
        ref = str(citation.get("ref") or f"Reference {index}")
        title = str(citation.get("title") or citation.get("url") or "Documentation page")
        url = citation.get("url")
        with st.container(border=True):
            st.markdown(f"**[{ref}] {title}**")
            if safe_external_link(url):
                st.markdown(f"[Open original documentation page ↗]({url})")
                st.caption(url)
            else:
                st.caption("The API did not return a valid HTTP(S) page URL for this reference.")
            details = []
            if citation.get("source_id"):
                details.append(f"Source ID: {citation['source_id']}")
            if citation.get("chunk_id"):
                details.append(f"Chunk ID: {citation['chunk_id']}")
            if citation.get("chunk_index") not in (None, ""):
                details.append(f"Chunk index: {citation['chunk_index']}")
            if citation.get("content_hash"):
                details.append(f"Content hash: {citation['content_hash']}")
            if details:
                st.caption(" · ".join(details))


def load_sources() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    payload = api("GET", "/sources", params={"limit": PAGE_SIZE, "offset": 0})
    if payload is None:
        return [], None
    return as_items(payload), payload if isinstance(payload, dict) else None


def render_knowledge_base(sources: list[dict[str, Any]], source_payload: dict[str, Any] | None) -> None:
    st.title("Knowledge Base")
    st.write("Register documentation, monitor indexing jobs, and inspect your source inventory.")

    total = source_payload.get("total", len(sources)) if source_payload else len(sources)
    active = sum(str(s.get("status", "")).lower() in {"queued", "running"} for s in sources)
    indexed = sum(str(s.get("status", "")).lower() in {"completed", "finished", "success", "succeeded"} for s in sources)
    failed = sum(str(s.get("status", "")).lower() == "failed" for s in sources)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Registered sources", total)
    metric_cols[1].metric("Indexed / successful", indexed)
    metric_cols[2].metric("In progress", active)
    metric_cols[3].metric("Failed", failed)

    st.subheader("Add a documentation source")
    with st.form("add_source_form", clear_on_submit=True, border=True):
        url = st.text_input(
            "Root documentation URL",
            placeholder="https://docs.python.org/3/library/argparse.html",
            help="Use the official documentation entry point you want DocuPilot to crawl.",
        )
        c1, c2 = st.columns(2)
        max_pages = c1.number_input("Maximum pages to crawl", min_value=1, max_value=500, value=10, step=1)
        max_depth = c2.number_input("Maximum crawl depth", min_value=0, max_value=10, value=1, step=1)
        submitted = st.form_submit_button("Add source and start indexing", type="primary", use_container_width=True)

    if submitted:
        if not url.strip() or not safe_external_link(url.strip()):
            st.warning("Enter a valid HTTP or HTTPS documentation URL.")
        else:
            result = api(
                "POST",
                "/sources",
                json={"url": url.strip(), "max_pages": int(max_pages), "max_depth": int(max_depth)},
            )
            if result:
                st.session_state.last_submitted_job = result.get("job_id")
                st.success(f"Indexing request accepted: {status_label(result.get('status', 'queued'))}")
                if result.get("job_id"):
                    st.caption(f"Job ID: {result['job_id']}")
                st.rerun()

    st.divider()
    st.subheader("Registered documentation sources")
    st.caption(
        "This inventory lists the root URLs registered with DocuPilot. "
        "Open a source below to inspect its indexed-page count and crawl details."
    )

    if not sources:
        st.info("No documentation sources found yet. Add a URL above to get started.")
        return

    for source in sources:
        source_id = source.get("source_id")
        source_url = source.get("url") or "(URL not returned by API)"
        status = source.get("status", "unknown")
        with st.container(border=True):
            top = st.columns([5, 1.4, 1.5])
            top[0].markdown(f"**{source_url}**")
            top[1].markdown(status_label(status))
            top[2].caption(f"Updated: {pretty_time(source.get('last_ingested_at'))}")

            if source.get("last_error"):
                st.error(str(source["last_error"]))
            action_cols = st.columns([1.2, 1.2, 4])
            inspect_clicked = action_cols[0].button("Inspect source", key=f"inspect-{source_id}", use_container_width=True)
            refresh_clicked = action_cols[1].button("Refresh source", key=f"refresh-{source_id}", use_container_width=True)
            if inspect_clicked and source_id:
                st.session_state.selected_source_detail = source_id
            if refresh_clicked and source_id:
                result = api("POST", f"/sources/{source_id}/refresh", json={})
                if result:
                    st.session_state.last_submitted_job = result.get("job_id")
                    st.success(f"Refresh request accepted: {status_label(result.get('status', 'queued'))}")
                    st.rerun()

            active_job_id = source.get("active_job_id")
            if active_job_id:
                st.caption(f"Active job: {active_job_id}")
                if st.button("Check indexing job", key=f"job-{source_id}"):
                    job = api("GET", f"/jobs/{active_job_id}")
                    if job:
                        st.write(f"**Job status:** {status_label(job.get('status'))}")
                        if job.get("error"):
                            st.error(str(job["error"]))
                        if job.get("result") is not None:
                            st.json(job["result"])

    selected_id = st.session_state.get("selected_source_detail")
    if selected_id:
        selected = next((s for s in sources if s.get("source_id") == selected_id), None)
        st.divider()
        st.subheader("Source details")
        detail = api("GET", f"/sources/{selected_id}")
        if detail:
            d1, d2, d3 = st.columns(3)
            d1.metric("Indexed pages", detail.get("page_count", "—"))
            d2.metric("Page versions", detail.get("version_count", "—"))
            crawl = detail.get("latest_crawl") or {}
            d3.metric("Pages seen in latest crawl", crawl.get("pages_seen", "—"))
            st.markdown(f"**Root URL:** {detail.get('url') or (selected or {}).get('url', '—')}")
            st.markdown(f"**Source status:** {status_label(detail.get('status'))}")
            st.markdown(f"**Last successful ingestion:** {pretty_time(detail.get('last_ingested_at'))}")
            if crawl:
                st.markdown("**Latest crawl**")
                st.json({
                    "status": crawl.get("status"),
                    "started_at": pretty_time(crawl.get("started_at")),
                    "finished_at": pretty_time(crawl.get("finished_at")),
                    "pages_seen": crawl.get("pages_seen"),
                    "errors_seen": crawl.get("errors_seen"),
                })
            st.info(
                "The current API exposes the indexed-page count and crawl summary, but not a full endpoint "
                "listing every indexed page URL. Page-level URLs retrieved for individual answers appear "
                "under each answer's citations."
            )


def render_assistant(sources: list[dict[str, Any]]) -> None:
    st.title("Documentation Assistant")
    st.write("Ask questions across your indexed documentation and verify answers against the original pages.")

    if not sources:
        st.info("Add and index at least one documentation source in **Knowledge Base** before asking a question.")
        return

    source_options: dict[str, str] = {}
    for source in sources:
        source_id = source.get("source_id")
        if not source_id:
            continue
        label = f"{source.get('url') or source_id}  ·  {source.get('status', 'unknown')}"
        source_options[label] = source_id

    selected_labels = st.multiselect(
        "Search within these sources",
        options=list(source_options),
        default=[],
        help="Choose one or more sources to scope retrieval. Leave empty to search across all registered sources.",
    )
    selected_ids = [source_options[label] for label in selected_labels]
    if not selected_ids:
        st.caption("Scope: all registered sources")
    else:
        st.caption(f"Scope: {len(selected_ids)} selected source(s)")

    controls = st.columns([1, 5])
    if controls[0].button("New conversation", use_container_width=True):
        st.session_state.conversation_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()
    controls[1].caption("Follow-up questions share the same conversation ID so the backend can resolve context.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("abstained"):
                st.info("The retrieved evidence was insufficient to provide a verified answer.")
            citations = message.get("citations") or []
            if citations:
                with st.expander(f"Verify answer · {len(citations)} cited page(s)", expanded=False):
                    render_citations(citations)
            timings = message.get("timings_ms") or {}
            if timings:
                st.caption(" · ".join(f"{key.replace('_', ' ').title()}: {value} ms" for key, value in timings.items()))

    question = st.chat_input("Ask about the indexed documentation, or ask a follow-up…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching indexed pages, checking evidence, and preparing an answer…"):
                result = api(
                    "POST",
                    "/questions",
                    json={
                        "question": question,
                        "conversation_id": st.session_state.conversation_id,
                        "source_ids": selected_ids or None,
                    },
                )

            if result:
                answer = str(result.get("answer") or "The API returned no answer.")
                citations = result.get("citations") or []
                abstained = bool(result.get("abstained", False))
                st.markdown(answer)
                if abstained:
                    st.info("DocuPilot could not verify a reliable answer from the retrieved evidence.")
                    reason = result.get("abstention_reason")
                    if reason:
                        st.caption(f"Reason: {str(reason).replace('_', ' ')}")
                if citations:
                    with st.expander(f"Verify answer · {len(citations)} cited page(s)", expanded=True):
                        render_citations(citations)
                else:
                    st.caption("No page-level citations were returned for this response.")
                timings = result.get("timings_ms") or {}
                if timings:
                    st.caption(" · ".join(f"{key.replace('_', ' ').title()}: {value} ms" for key, value in timings.items()))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "abstained": abstained,
                    "timings_ms": timings,
                })
            else:
                st.warning("The question request failed. Your question is still visible above; check API health and try again.")


st.markdown(
    """
    <div class="dp-hero">
      <h1>📚 DocuPilot</h1>
      <p>Evidence-grounded technical documentation workspace</p>
      <div class="dp-muted">Manage documentation sources · Ask follow-up questions · Verify citations against original pages</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Workspace")
    view = st.radio(
        "Choose a workspace",
        ["Knowledge Base", "Documentation Assistant"],
        key="active_view",
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("Check API health", use_container_width=True):
        health = api("GET", "/health")
        if health and health.get("status") == "ok":
            st.success("DocuPilot API is healthy.")
        elif health:
            st.warning(f"API health response: {health}")
    st.caption(f"API: {API_URL}")
    st.caption(f"Conversation: {st.session_state.conversation_id[:8]}…")

sources, source_payload = load_sources()
if view == "Knowledge Base":
    render_knowledge_base(sources, source_payload)
else:
    render_assistant(sources)
