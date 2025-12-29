import json
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/generate_stream"

st.set_page_config(page_title="Local SLM Chat", page_icon="💬", layout="centered")
st.title("💬 Local Chat (Your SLM)")

# Keep conversation in session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me a TinyStories-style question."}
    ]

# Render existing messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def build_prompt(messages):
    # Simple chat formatting; you can tune this
    lines = []
    for m in messages:
        if m["role"] == "user":
            lines.append(f"User: {m['content']}")
        else:
            lines.append(f"Assistant: {m['content']}")
    return "\n".join(lines).strip() + "\nAssistant:"

def stream_sse_tokens(response):
    """
    Parse SSE from FastAPI endpoint:
    event: token
    data:  hello
    """
    event = None
    for raw in response.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if raw.startswith("event:"):
            event = raw.split(":", 1)[1].strip()
        elif raw.startswith("data:"):
            data = raw.split(":", 1)[1]  # keep leading spaces
            if event == "token":
                yield data

# Input box (ChatGPT style)
user_input = st.chat_input("Type your message...")
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add placeholder assistant message
    with st.chat_message("assistant"):
        placeholder = st.empty()
        assistant_text = ""

        # Build prompt from conversation so far
        prompt = build_prompt(st.session_state.messages)

        payload = {
            "prompt": prompt,
            "max_new_tokens": 200,
            "temperature": 0.8,
            "top_k": 200,
        }

        with requests.post(
            BACKEND_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            for tok in stream_sse_tokens(r):
                assistant_text += tok
                placeholder.markdown(assistant_text)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

# Sidebar controls
with st.sidebar:
    st.subheader("Controls")
    if st.button("New chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me a TinyStories-style question."}
        ]
        st.rerun()
