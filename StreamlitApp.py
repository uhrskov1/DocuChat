from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer)
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from htmlTemplate import css, bot_template, user_template
from dotenv import load_dotenv
import faiss
import logging
import streamlit as st
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt template
llmtemplate = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not prompt to select answers. do not ask me questions
{question}
[/INST]
"""


def prepare_pdf_file():
    # load the local data directory and chunk the data for further processing
    docs = SimpleDirectoryReader(input_dir="data", required_exts=[".pdf"]).load_data(show_progress=True)
    text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    text_chunks = []
    doc_ids = []
    nodes = []

    # Create a local Faiss vector store
    d = 1024  # dimensions of mxbai-embed-large
    faiss_index = faiss.IndexFlatL2(d)
    logger.info("initializing the vector store related objects")
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # local vector embeddings model
    logger.info("initializing the OllamaEmbedding")
    embed_model = OllamaEmbedding(model_name='mxbai-embed-large', base_url='http://localhost:11434')
    logger.info("initializing the global settings")
    Settings.embed_model = embed_model
    Settings.llm = Ollama(model="llama3", base_url='http://localhost:11434')
    Settings.transformations = [text_parser]

    logger.info("enumerating docs")
    for doc_idx, doc in enumerate(docs):
        curr_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(curr_text_chunks)
        doc_ids.extend([doc_idx] * len(curr_text_chunks))

    logger.info("enumerating text_chunks")
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = docs[doc_ids[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    logger.info("enumerating nodes")
    try:
        for node in nodes:
            node_embedding = embed_model.get_text_embedding(
                node.get_content(metadata_mode=MetadataMode.ALL)
            )
            node.embedding = node_embedding
    except:
        print('Is Ollama running?')

    logger.info("initializing the storage context")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("indexing the nodes in VectorStoreIndex")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        transformations=Settings.transformations,
    )

    logger.info("initializing the VectorIndexRetriever with top_k as 5")
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    response_synthesizer = get_response_synthesizer()
    logger.info("creating the RetrieverQueryEngine instance")
    vector_query_engine = RetrieverQueryEngine(
        retriever=vector_retriever,
        response_synthesizer=response_synthesizer,
    )
    logger.info("creating the HyDEQueryTransform instance")
    hyde = HyDEQueryTransform(include_original=True)
    hyde_query_engine = TransformQueryEngine(vector_query_engine, hyde)
    return hyde_query_engine


def handle_userinput(user_question):
    response = st.session_state.conversation.query(str_or_query_bundle=user_question)
    st.session_state.chat_history.append(response)

    response_text = response.response
    print(response_text)
    st.write(bot_template.replace("{{MSG}}", response_text), unsafe_allow_html=True)


def main():
    load_dotenv()

    # Directory where to save the uploaded files, Make sure this directory exists or create it
    # Saving file to drive as llama_index only has function to read from directory (SimpleDirectoryReader)
    upload_folder = 'data'
    os.makedirs(upload_folder, exist_ok=True)

    # Delete all files in the directory - to be sure to chat with the uploaded document
    for entry in os.listdir('data'):
        file_path = os.path.join('data', entry)
        if os.path.isfile(file_path):
            os.unlink(file_path)  # Remove the file

    st.set_page_config(page_title="Chat with your PDF",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with your PDF :books:")
    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your document")
        uploaded_file = st.file_uploader(
            "Upload your PDF here and click on 'Process'", accept_multiple_files=False)

        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # Path to save the file
            save_path = os.path.join(upload_folder, uploaded_file.name)

            # Write the file to the specified folder
            with open(save_path, "wb") as f:
                f.write(bytes_data)

        if st.button("Process"):
            with st.spinner("Processing"):
                st.session_state.conversation = prepare_pdf_file()


if __name__ == '__main__':
    main()
