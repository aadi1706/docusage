"""DocuSage — Streamlit chat frontend."""
import os
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuSage", page_icon="📄", layout="wide")


def _render_response(data: dict):
    st.write(data["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{data['latency_ms']:.0f} ms")
    col2.metric(
        "Confidence",
        f"{data['confidence_score']:.0%}" if data["confidence_score"] is not None else "—",
    )
    col3.markdown(f"**Type:** `{data['query_type']}`")

    if data.get("citations"):
        with st.expander(f"📎 Citations ({len(data['citations'])})"):
            for c in data["citations"]:
                st.markdown(f"- {c}")

    if data.get("hallucination_flags"):
        st.warning("⚠️ Hallucination flags: " + ", ".join(data["hallucination_flags"]))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL)

    if st.button("🔌 Ping /health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                st.success(f"● Online — {r.json().get('service', 'ok')}")
            else:
                st.error(f"● HTTP {r.status_code}")
        except Exception as e:
            st.error(f"● Unreachable: {e}")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 DocuSage")
st.caption("Agentic RAG for Indian financial documents — RBI, SEBI, NSE/BSE")

# ── Replay history ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            _render_response(msg["data"])

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about RBI circulars, SEBI regulations, financial reports…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                payload = {"query": prompt}
                if st.session_state.session_id:
                    payload["session_id"] = st.session_state.session_id

                r = requests.post(
                    f"{api_url}/query",
                    json=payload,
                    timeout=120,
                )

                if r.status_code != 200:
                    st.error(f"API error {r.status_code}: {r.text}")
                else:
                    data = r.json()
                    st.session_state.session_id = data.get("session_id")
                    _render_response(data)
                    st.session_state.messages.append({"role": "assistant", "data": data})

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at {api_url}. Is the server running?")
            except requests.exceptions.Timeout:
                st.error("Request timed out after 120s.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
