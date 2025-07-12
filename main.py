import os
import shutil
import time
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from chromadb.config import Settings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === Load environment variables ===
load_dotenv()

# # === Session-specific Chroma DB path ===
# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())
import shutil

# === Session-specific Chroma DB path ===
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

CHROMA_SESSION_DIR = f"chroma_sessions/{st.session_state.session_id}"

try:
    shutil.rmtree(CHROMA_SESSION_DIR, ignore_errors=True)
except Exception as e:
    print(f"Error deleting Chroma sessions directory: {e}")

os.makedirs(CHROMA_SESSION_DIR, exist_ok=True)

CHROMA_SESSION_DIR = f"chroma_sessions/{st.session_state.session_id}"
os.makedirs(CHROMA_SESSION_DIR, exist_ok=True)

# === State ===
class State(TypedDict):
    messages: Annotated[list, add_messages]

# === Tools list ===
tools = []

# === Embeddings ===
embedding_fn = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# === Dictionary to store DBs per PDF ===
per_file_dbs = {}
if "per_file_dbs" not in st.session_state:
    st.session_state.per_file_dbs = {}
per_file_dbs = st.session_state.per_file_dbs

# === RAG Tool ===
# def document_retrieval(query: str) -> str:
#     if not per_file_dbs:
#         return "No documents indexed yet. Please upload PDFs first."
#     selected_file = st.sidebar.selectbox("Select PDF to search:", list(per_file_dbs.keys()))
#     db = per_file_dbs[selected_file]
#     results = db.similarity_search(query)
#     if not results:
#         return "Sorry, I couldn't find relevant info in the selected PDF."
#     return "\n\n".join([doc.page_content for doc in results])
# === RAG Tool ===
def document_retrieval(query: str) -> str:
    if not per_file_dbs:
        return "No documents indexed yet. Please upload PDFs first."
    all_results = []
    for db_name, db in per_file_dbs.items():
        results = db.similarity_search(query)
        if results:
            all_results.extend(results)
    if not all_results:
        return "Sorry, I couldn't find relevant info in the documents."
    return "\n\n".join([doc.page_content for doc in all_results])

# === Static Tools ===
def multiply(a: int, b: int) -> int:
    return a * b

def add(a: int, b: int) -> int:
    return a + b

# === Add tools ===
tools.extend([
    Tool(name="duckduckgo_search", func=DuckDuckGoSearchRun().run, description="Search the web using DuckDuckGo."),
    Tool(name="add", func=add, description="Add two numbers."),
    Tool(name="multiply", func=multiply, description="Multiply two numbers."),
    Tool(name="document_retrieval", func=document_retrieval, description="Retrieves relevant information from uploaded PDF documents using semantic search. Use this tool to answer user questions based on the content of the uploaded files.")
])

# === LLM Setup ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)
llm_with_tools = llm.bind_tools(tools)

# === LangGraph ===
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", lambda state: {"messages": [llm_with_tools.invoke(state["messages"])]})
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# === Streamlit UI ===
if __name__ == "__main__":
    st.set_page_config(page_title="Sameer AI ")
    st.title(" Sameer AI ")
    st.sidebar.header("Upload PDFs")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "user-001"
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_uploaded_files" not in st.session_state:
        st.session_state.last_uploaded_files = set()
    st.warning(
        "📱 If you're on a mobile device, tap the `>>` menu at the top-left to upload PDFs.",
        icon="⚠️"
    )
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # === Handle new uploads ===
    if uploaded_files:
        current_uploaded = set(file.name for file in uploaded_files)

        if current_uploaded != st.session_state.last_uploaded_files:
            # st.info("New PDFs detected. Indexing individually...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            for uploaded_file in uploaded_files:
                path = f"temp_{uploaded_file.name}"
                with open(path, "wb") as f:
                    f.write(uploaded_file.read())
                loader = PyPDFLoader(path)
                chunks = loader.load_and_split(splitter)

                file_db_path = os.path.join(CHROMA_SESSION_DIR, uploaded_file.name)
                # file_db = Chroma.from_documents(
                #     documents=chunks,
                #     embedding=embedding_fn,
                #     persist_directory=file_db_path
                # )
                # file_db.persist()
                # per_file_dbs[uploaded_file.name] = file_db
                chroma_settings = Settings(
                    persist_directory=file_db_path,
                    anonymized_telemetry=False,
                    is_persistent=True
                )
                file_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_fn,
                    persist_directory=file_db_path,
                    client_settings=chroma_settings
                )
                file_db.persist()
                per_file_dbs[uploaded_file.name] = file_db

            st.session_state.last_uploaded_files = current_uploaded
            # st.success("PDFs indexed separately and saved!")
            st.success("PDFs indexed separately and saved!")

            # Recompile graph with tools
            llm_with_tools = llm.bind_tools(tools)
            st.session_state.graph = graph_builder.compile(checkpointer=st.session_state.checkpointer)

    if "graph" not in st.session_state:
        st.session_state.graph = graph_builder.compile(checkpointer=st.session_state.checkpointer)

    graph = st.session_state.graph

    # Chat History Display
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # Chat Input
    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        config = {
            "thread_id": st.session_state.thread_id,
            "checkpoint": st.session_state.checkpointer
        }

        try:
            response = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            ai_message = next(
                (m.content.strip() for m in reversed(response["messages"]) if isinstance(m, (AIMessage, ToolMessage)) and m.content),
                None
            )
            if ai_message:
                st.session_state.chat_history.append(("assistant", ai_message))
                with st.chat_message("assistant"):
                    st.markdown(ai_message)
        except Exception as e:
            st.error(f"❌ Error: {e}")