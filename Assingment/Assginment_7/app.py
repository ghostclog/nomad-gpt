from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

from functions import make_llm,make_retriever

docs = None
choice = None
correct_pt = 0
num_of_question = 0
st.set_page_config(
    page_title="Assignment 7",
    page_icon="❓",
)

st.title("Assignment 7")

st.markdown("""
            GPT 과제 7번입니다!
            
            1. 사이드탭에 api키를 입력해주세요!

            2. GPT가 사이트를 다 확인하면, 해당 사이트(cloudflare)에 대한 질문을 해주세요!
            """)


with st.sidebar:
    api_key = st.text_input("하단에 api 키를 입력해주세요.")
    if api_key:
        make_llm(api_key)
        
    #분할선
    st.divider()
    st.write("레포지토리 링크: https://github.com/ghostclog/nomad-gpt-ch10-Assignment")
    st.divider()
    st.header("app.py 내용")
    st.markdown("""
 docs = None
choice = None
correct_pt = 0
num_of_question = 0
st.set_page_config(
    page_title="Assignment 7",
    page_icon="❓",
)

st.title("Assignment 7")

st.markdown('''
            GPT 과제 7번입니다!
            
            1. 사이드탭에 api키를 입력해주세요!

            2. GPT가 사이트를 다 확인하면, 해당 사이트(cloudflare)에 대한 질문을 해주세요!
            ''')


with st.sidebar:
    api_key = st.text_input("하단에 api 키를 입력해주세요.")
    if api_key:
        make_llm(api_key)               

if api_key:
    st.write("")
    st.write("")
    make_retriever()
    """)
    st.divider()
    st.header("function.py 내용입니다.")
    st.markdown("""
    from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

llm = None

def make_llm(key):
    global llm
    llm = ChatOpenAI(
        temperature=0.5,
        streaming=True,
        openai_api_key=key
        )

answers_prompt = ChatPromptTemplate.from_template(
    '''
    다음 컨텍스트만 사용하면 사용자의 질문에 답할 수 있습니다. 모른다고만 말할 수 없다면 아무것도 지어내지 마세요.
                                                    
    그런 다음 0에서 5 사이의 답에 점수를 부여합니다.

    답변이 사용자에게 높은 점수를 요구하면 낮은 점수를 받아야 합니다.

    0인 경우에도 항상 정답의 점수를 포함해야 합니다.

    Context: {context}
                                                  
    Examples:
                                                  
    질문: 달은 얼마나 멀리 있나요?
    답변: 달은 384,400km 떨어져 있습니다.
    점수: 5
                                                  
    질문: 태양은 얼마나 멀리 있나요?
    답변: 모르겠습니다
    점수: 0
                                                  
    네 차례!

    Question: {question}
'''
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            다음 기존 답변만 사용하여 사용자의 질문에 답하세요.

            가장 높은 점수(더 도움이 되는)를 받고 가장 최근의 점수를 선호하는 답을 사용합니다.

            출처를 인용하고 답변의 출처를 변경하지 말고 그대로 반환합니다.

            Answers: {answers}
            ''',
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="웹사이트 로딩 중...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

def make_retriever():
    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")
    query = st.text_input("해당 웹사이트에 대해 물어보세요.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$"))
    """)


if api_key:
    st.write("")
    st.write("")
    make_retriever()