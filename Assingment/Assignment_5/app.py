from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from functions import embed_file,save_message,send_message,paint_history,format_docs
import streamlit as st

st.set_page_config(
    page_title="Assignment5",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            다음 문맥만을 사용하여 질문에 답하세요. 답을 모르면 그냥 모른다고 하세요. 아무것도 지어내지 마세요. 다만 "고마워"와 같이 문서를 참고하지 않아도 할 수 있는 간단한 질문에 대해서는 짧게 대답해도 됩니다.
            반드시 한국어로 대답해야하며, 되도록 경어를 사용하세요.
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

### 디자인 ###
st.title("Assignment 5")
st.markdown("""어서 오세요!!
            
챗봇을 사용하여, 당신이 업로드한 문서에 대한 질문을 해보세요!!!""")
### 디자인 ###

with st.sidebar:
    key = st.text_input("여기에 당신의 api 키를 입력해주세요.")
    if key:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )
    st.header("깃허브 레포지토리 링크: https://github.com/ghostclog/nomad-gpt-ch7-Assignment")
    st.header("- function.py 내용 >>>")
    st.markdown("""
@st.cache_data(show_spinner="파일을 임베딩 중입니다...")
def embed_file(file,key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_type=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
""")
    st.header("- app.py 내용 >>")
    st.markdown(
"""
st.set_page_config(
    page_title="Assignment5",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            다음 문맥만을 사용하여 질문에 답하세요. 답을 모르면 그냥 모른다고 하세요. 아무것도 지어내지 마세요. 다만 "고마워"와 같이 문서를 참고하지 않아도 할 수 있는 간단한 질문에 대해서는 짧게 대답해도 됩니다.
            반드시 한국어로 대답해야하며, 되도록 경어를 사용하세요.
            Context: {context}
            ''',
        ),
        ("human", "{question}"),
    ]
)

### 디자인 ###
st.title("Assignment 5")
st.markdown('''어서 오세요!!
            
챗봇을 사용하여, 당신이 업로드한 문서에 대한 질문을 해보세요!!!''')
### 디자인 ###

with st.sidebar:
    key = st.text_input("여기에 당신의 api 키를 입력해주세요.")
    if key:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )

if key:
    llm = ChatOpenAI(
        temperature=0.5,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        openai_api_key=key
    )

    if file:
        retriever = embed_file(file,key)
        send_message("준비됬어요! 질문해주세요!", "ai", save=False)
        paint_history()
        message = st.chat_input("당신이 업로드한 문서에 대한 질문을 주세요.")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                response = chain.invoke(message)
    else:
        st.session_state["messages"] = []

    """)
if key:
    llm = ChatOpenAI(
        temperature=0.5,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        openai_api_key=key
    )

    if file:
        retriever = embed_file(file,key)
        send_message("준비됬어요! 질문해주세요!", "ai", save=False)
        paint_history()
        message = st.chat_input("당신이 업로드한 문서에 대한 질문을 주세요.")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                response = chain.invoke(message)
    else:
        st.session_state["messages"] = []