from typing import Any
from uuid import UUID
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

# 웹 페이지 설정
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# 챗팅이 생성되는걸 실시간으로 보여주기 위한 클래스.
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)



# 데이터 캐싱하기. 해당 어노테이션?이 존재하면 이미 존재하는 파일에 대해선 임베딩을 다시 하진 않음.
@st.cache_data(show_spinner="파일을 임베딩 중입니다...")
def embed_file(file):
    # 파일 내용 읽기
    file_content = file.read()
    # 파일 저장 경로 생성
    file_path = f"./.cache/files/{file.name}"
    # 이진 파일로 생성 및 내용 작성. 내용은 업로드된 파일 내용으로... 저장해야 임베딩 가능
    with open(file_path, "wb") as f:
        f.write(file_content)
    # 캐시 파일 경로 지정
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    # 문서 쪼개기 툴
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    # 저장한 문서 가져오기
    loader = UnstructuredFileLoader(file_path)
    # 툴로 문서 쪼개기
    docs = loader.load_and_split(text_splitter=splitter)
    #임베딩하고, 캐싱 저장하기
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # 백터 스토어 생성
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    # 리트리버 생성. 반환하여 체인에 사용
    retriever = vectorstore.as_retriever()
    return retriever

# 내용 저장
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

# 메세지 보내기. 만약에 신규 메세지면, session에 저장함.
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 기존 대화 로그 출력해주는 부분 / 이미 저장된 대화 로그이기에 때문에 save는 false로 지정
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


# ai한테 제공할 문서 제작 및 반환
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

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
st.title("DocumentGPT")
st.markdown("""어서 오세요!!
            
챗봇을 사용하여, 당신이 업로드한 문서에 대한 질문을 해보세요!!!""")
### 디자인 ###

with st.sidebar:
    api=st.text_input("api키를 입력해주세요.")
    if api:
        # 모델
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
            opne_api_key = api
        )
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )

# 파일이 들어온 경우
if file:
    retriever = embed_file(file)
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
        # send_message(response.content, "ai") 상단의 st.chat_message("ai") 때문에 알아서 ai가 쓴 글로 판정됨.
        # 다만, 저장이 안되기에 저장하기 위해선 llm이 종료되는 시점에 해당 메세지를 전달해줘야함.
else:
    st.session_state["messages"] = []