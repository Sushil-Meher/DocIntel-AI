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
st.subheader("Ask questions about your documents or company website")


if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "source" not in st.session_state:
    st.session_state.source = None


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

                    st.sidebar.success("PDF processed successfully!")

                finally:

                    if os.path.exists(temp_path):
                        os.remove(temp_path)


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

                    st.sidebar.success("Website processed successfully!")

                except Exception as e:

                    st.sidebar.error(f"Error: {e}")


st.divider()


if st.session_state.index is not None:

    st.success(
        f"Currently using: {st.session_state.source}"
    )

    query = st.text_input(
        "Ask a question",
        placeholder="What are the most important points in the company policy?"
    )

    if st.button("Ask"):

        if not query.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Thinking..."):

                answer = answer_question(
                    query,
                    st.session_state.index,
                    st.session_state.chunks
                )

            st.markdown("### Answer")
            st.write(answer)

else:

    st.info(
        "Upload a PDF or enter a company website to get started."
    )