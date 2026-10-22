from __future__ import annotations
import os, uuid
from typing import Any
import httpx
import streamlit as st

API_URL = os.getenv("DOCUPILOT_API_URL", "http://api:8000").rstrip("/")
TIMEOUT = httpx.Timeout(120.0, connect=5.0)
st.set_page_config(page_title="DocuPilot", page_icon="📚", layout="wide")
st.markdown("""<style>.block-container{max-width:1120px;padding-top:2rem}.hero{padding:1.2rem 1.5rem;border:1px solid rgba(128,128,128,.3);border-radius:14px;margin-bottom:1rem}</style>""", unsafe_allow_html=True)
if "conversation_id" not in st.session_state: st.session_state.conversation_id = uuid.uuid4().hex
if "messages" not in st.session_state: st.session_state.messages = []

def api(method: str, path: str, **kwargs: Any):
    try:
        with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
    except httpx.HTTPStatusError as exc:
        try: detail = exc.response.json()
        except ValueError: detail = exc.response.text
        st.error(f"API error {exc.response.status_code}: {detail}")
    except httpx.RequestError as exc:
        st.error(f"Cannot reach API at {API_URL}. Check Docker services. ({exc})")
    return None

st.markdown("""<div class="hero"><h1>📚 DocuPilot</h1><p>Ask technical questions and get answers grounded in indexed documentation.</p><small>Source-aware retrieval · Verifiable citations · Background indexing</small></div>""", unsafe_allow_html=True)
with st.sidebar:
    st.subheader("Backend")
    if st.button("Check API health", use_container_width=True):
        health=api("GET","/health")
        if health and health.get("status")=="ok": st.success("API healthy")
    st.divider()
    st.subheader("Add documentation")
    with st.form("source_form", clear_on_submit=True):
        url=st.text_input("Documentation URL", placeholder="https://docs.example.com/")
        c1,c2=st.columns(2)
        pages=c1.number_input("Maximum pages",1,500,10)
        depth=c2.number_input("Crawl depth",0,10,1)
        submit=st.form_submit_button("Add and index",type="primary",use_container_width=True)
    if submit:
        if not url.strip(): st.warning("Enter a URL.")
        else:
            result=api("POST","/sources",json={"url":url.strip(),"max_pages":int(pages),"max_depth":int(depth)})
            if result: st.success("Ingestion job submitted"); st.json(result)
    st.divider()
    st.subheader("Sources")
    source_data=api("GET","/sources",params={"limit":100,"offset":0})
    sources=source_data.get("items",[]) if isinstance(source_data,dict) else []
    options={f"{s.get('status','unknown')} · {s.get('url') or s.get('source_id')}":s["source_id"] for s in sources if s.get("source_id")}
    selected=st.multiselect("Restrict answers to",list(options),help="Leave empty to search all indexed sources.")
    with st.expander("Source status",expanded=False):
        if not sources: st.caption("No sources registered yet.")
        for s in sources:
            st.markdown(f"**{s.get('status','unknown')}**")
            st.caption(s.get("url") or s.get("source_id",""))
            if s.get("last_error"): st.error(str(s["last_error"]))
    if st.button("New conversation",use_container_width=True):
        st.session_state.conversation_id=uuid.uuid4().hex
        st.session_state.messages=[]
        st.rerun()

st.caption(f"API endpoint: {API_URL}")
st.subheader("Ask your documentation")
st.caption("Index a documentation URL first. DocuPilot cites retrieved evidence and may abstain when evidence is insufficient.")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Citations"):
                for c in msg["citations"]:
                    title=c.get("title") or c.get("ref") or "Source"
                    if c.get("url"): st.markdown(f"- [{title}]({c['url']})")
                    else: st.markdown(f"- {title}")
                    if c.get("chunk_id"): st.caption(f"Chunk: {c['chunk_id']}")
question=st.chat_input("Ask a question about your indexed documentation…")
if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating an answer…"):
            result=api("POST","/questions",json={"question":question,"conversation_id":st.session_state.conversation_id,"source_ids":[options[x] for x in selected] or None})
        if result:
            answer=str(result.get("answer") or "No answer returned.")
            citations=result.get("citations") or []
            st.markdown(answer)
            if result.get("abstained"): st.info("DocuPilot abstained because evidence was insufficient or citations could not be verified.")
            if citations:
                with st.expander(f"Evidence and citations ({len(citations)})",expanded=True):
                    for c in citations:
                        title=c.get("title") or c.get("ref") or "Source"
                        if c.get("url"): st.markdown(f"- [{title}]({c['url']})")
                        else: st.markdown(f"- {title}")
                        if c.get("chunk_id"): st.caption(f"Chunk: {c['chunk_id']}")
            timings=result.get("timings_ms") or {}
            if timings: st.caption(" · ".join(f"{k}: {v} ms" for k,v in timings.items()))
            st.session_state.messages.append({"role":"assistant","content":answer,"citations":citations})
        else: st.warning("Request failed. Check API health and logs.")
elif not st.session_state.messages:
    st.info("Add a documentation URL in the sidebar, wait for indexing, then ask a question.")
