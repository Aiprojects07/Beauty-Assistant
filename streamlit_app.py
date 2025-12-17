import os
from pathlib import Path
import uuid

import streamlit as st
from dotenv import load_dotenv

# Load .env BEFORE importing modules that read environment at import time
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# Map Streamlit Cloud secrets into environment vars (if provided) BEFORE imports that read os.getenv
for key in (
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "MEMORY_SESSION_ID",
):
    try:
        if key in st.secrets and not os.getenv(key):
            val = st.secrets.get(key)
            if val is not None:
                os.environ[key] = str(val)
    except Exception:
        # Safe on local runs without st.secrets
        pass

# Local imports (now env vars are available)
from product_tools_optimized_updated import general_product_qna, SessionState

APP_TITLE = "Beauty Assistant"

# --- Sidebar ---
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title(APP_TITLE)

# Generate unique session ID per user (first load)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"user_{uuid.uuid4().hex[:8]}"
session_id = st.session_state["session_id"]

with st.sidebar:
    st.header("Session")
    st.caption(f"Your session: `{session_id}`")

    sess = SessionState(session_id)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Context"):
            summary = sess.get_summary()
            st.session_state["context_summary"] = summary
        
    with col2:
        if st.button("Reset Session"):
            # Clear python state and memory files (similar to CLI reset)
            sess.clear()
            for f in sess.memory_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    try:
                        f.unlink()
                    except Exception:
                        pass
            st.session_state["messages"] = []
            st.session_state["context_summary"] = "(cleared)"
            st.success("Session and memory cleared.")
            st.rerun()

    # Context display
    st.divider()
    st.subheader("Context Preview")
    st.markdown(
        f"""
        <pre style='white-space: pre-wrap;'>
{st.session_state.get('context_summary', '(none)')}
        </pre>
        """,
        unsafe_allow_html=True,
    )

# --- Chat Area ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts: {role: "user"|"assistant", content: str}

# 1) Render existing conversation history first
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# 2) Chat input docked at the bottom
user_query = st.chat_input("Ask something...", key="chat_input")

# 3) On submit, append the user turn and stream the assistant reply live
if user_query and user_query.strip():
    user_text = user_query.strip()
    # Append and render the new user message immediately
    st.session_state["messages"].append({"role": "user", "content": user_text})
    st.chat_message("user").markdown(user_text)

    # Prepare a streaming placeholder for the assistant
    assistant_box = st.chat_message("assistant")
    stream_placeholder = assistant_box.empty()
    streamed_parts: list[str] = []

    def _on_chunk(chunk: str):
        if not isinstance(chunk, str) or not chunk:
            return
        streamed_parts.append(chunk)
        stream_placeholder.markdown("".join(streamed_parts))

    # Call backend with streaming callback
    try:
        final_answer = general_product_qna(
            query=user_text,
            session_id=session_id,
            stream_callback=_on_chunk,
        )
    except Exception as e:
        final_answer = f"[ERROR] {e}"

    # Ensure final text is displayed and persist it in history
    stream_placeholder.markdown(final_answer)
    st.session_state["messages"].append({"role": "assistant", "content": final_answer})
