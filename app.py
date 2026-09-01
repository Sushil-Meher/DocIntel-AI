import os
import sys
import tempfile

import streamlit as st

from src.ingestion import ingest_pdf, ingest_url
from src.rag import answer_question


st.set_page_config(
    page_title="RAGForge AI",
    page_icon="📄",
    layout="centered"
)

# Streamlit's default top padding leaves a lot of empty space above the
# title; this is the only CSS in the app, purely to tighten that.
st.markdown(
    "<style>.block-container { padding-top: 2.5rem; }</style>",
    unsafe_allow_html=True
)

st.title("RAGForge AI")
st.caption("Document intelligence for PDFs and websites.")


def format_source(source):
    if source["source_type"] == "Website":
        return source["source"]

    if "page" in source:
        return f"Page {source['page']} · {source['source']}"

    return source["source"]


if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "source" not in st.session_state:
    st.session_state.source = None

if "source_type" not in st.session_state:
    st.session_state.source_type = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:

    st.subheader("Source")

    source_type = st.segmented_control(
        "Source type",
        ["PDF", "Website"],
        default="PDF",
        required=True,
        label_visibility="collapsed"
    )

    if source_type == "PDF":

        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"]
        )

        if uploaded_file is not None:

            if st.button("Process PDF"):

                with st.spinner("Processing PDF..."):

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getbuffer())
                        temp_path = temp_file.name

                    try:

                        index, chunks = ingest_pdf(temp_path)

                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.source = uploaded_file.name
                        st.session_state.source_type = "PDF"
                        st.session_state.chat_history = []

                        st.success("PDF processed.")

                    except Exception as e:

                        st.error(f"Error: {e}")

                    finally:

                        # Best-effort cleanup - on Windows the temp file can
                        # briefly stay locked, which shouldn't fail an
                        # otherwise-successful upload.
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except OSError:
                            pass

    else:

        url = st.text_input(
            "Website URL",
            placeholder="https://example.com"
        )

        if st.button("Process website"):

            if not url:
                st.error("Enter a website URL.")

            else:

                with st.spinner("Reading website..."):

                    try:

                        index, chunks = ingest_url(url)

                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.source = url
                        st.session_state.source_type = "Website"
                        st.session_state.chat_history = []

                        st.success("Website processed.")

                    except Exception as e:

                        st.error(f"Error: {e}")

    if st.session_state.index is not None:

        st.divider()
        st.caption("CURRENT SOURCE")
        st.markdown(
            f"**{st.session_state.source_type} · {st.session_state.source}**"
        )
        st.caption("Ready")
        st.caption("Loading a new source starts a fresh conversation.")


if st.session_state.index is not None:

    for turn in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            st.write(turn["answer"])

            if turn.get("sources"):
                st.markdown("**Sources**")
                for source in turn["sources"]:
                    st.caption(format_source(source))

    query = st.chat_input("Ask a follow-up question...")

    if query:

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Thinking..."):

            answer = answer_question(
                query,
                st.session_state.index,
                st.session_state.chunks,
                history=st.session_state.chat_history
            )

        with st.chat_message("assistant"):
            st.write(answer.text)

            if answer.sources:
                st.markdown("**Sources**")
                for source in answer.sources:
                    st.caption(format_source(source))

        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": answer.text,
                "sources": answer.sources
            }
        )

else:

    st.info(
        "**No document loaded**\n\n"
        "Upload a PDF or enter a website URL in the sidebar to start "
        "asking questions."
    )
