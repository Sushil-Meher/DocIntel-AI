import os
import sys
import tempfile

import streamlit as st

from src.ingestion import ingest_pdf, ingest_url
from src.rag import answer_question


st.set_page_config(
    page_title="DocIntel AI",
    page_icon="📚",
    layout="wide"
)


st.title("📚 DocIntel AI")
st.caption("Ask questions about your PDFs and websites.")


def format_source(source):
    if source["source_type"] == "Website":
        return source["source"]

    if "page" in source:
        return f"Page {source['page']} — {source['source']}"

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


st.sidebar.header("Document Source")

source_type = st.sidebar.radio(
    "Choose source",
    ["Upload PDF", "Company Website"]
)


if source_type == "Upload PDF":

    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.sidebar.button("Process PDF"):

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

                    st.sidebar.success("PDF processed successfully!")

                except Exception as e:

                    st.sidebar.error(f"Error: {e}")

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

    url = st.sidebar.text_input(
        "Company website URL",
        placeholder="https://example.com"
    )

    if st.sidebar.button("Process Website"):

        if not url:
            st.sidebar.error("Please enter a website URL.")

        else:

            with st.spinner("Loading website..."):

                try:

                    index, chunks = ingest_url(url)

                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.source = url
                    st.session_state.source_type = "Website"
                    st.session_state.chat_history = []

                    st.sidebar.success("Website processed successfully!")

                except Exception as e:

                    st.sidebar.error(f"Error: {e}")


st.divider()


if st.session_state.index is not None:

    with st.container(border=True):
        st.markdown("**Current source**")
        st.markdown(f"{st.session_state.source_type}: {st.session_state.source}")
        st.markdown("Status: Ready")

    st.caption(
        "Conversation history is scoped to this document - "
        "processing a new PDF or website starts a fresh conversation."
    )

    for turn in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            st.write(turn["answer"])

            if turn.get("sources"):
                st.markdown("**Sources**")
                for source in turn["sources"]:
                    st.markdown(f"- {format_source(source)}")

    query = st.chat_input("Ask a question about the current document")

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
                    st.markdown(f"- {format_source(source)}")

        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": answer.text,
                "sources": answer.sources
            }
        )

else:

    st.info(
        "Upload a PDF or enter a company website URL in the sidebar "
        "to get started."
    )
