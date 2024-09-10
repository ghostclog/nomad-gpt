import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

@st.cache_data(show_spinner="파일 로딩중...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 교사로서 역할을 수행하는 데 도움이 되는 어시스턴트입니다.
         
    다음 맥락에만 근거하여 텍스트에 대한 사용자의 지식을 테스트하기 위해 10개의 질문을 작성합니다.
    
    각 질문에는 4개의 답이 있어야 하고, 그 중 3개는 틀려야 하며, 1개는 틀려야 합니다.
         
    (o)를 사용하여 정답을 알립니다.
         
    질문 예제:
         
    질문: 바다의 색은 무엇인가요?
    답변: 빨간색|노란색|녹색|파란색(o)
         
    질문: 수도 또는 조지아는 어디인가요?
    답변: 바쿠|트빌리시(o)|마닐라|베이루트
         
    질문: 아바타는 언제 출시되었나요?
    답변: 2007|2001|2009(o)|1998
         
    질문: 줄리어스 시저는 누구였나요?
    답변: 로마 황제(o)|화가|배우|모델
         
    네 차례!
         
    Context: {context}
""",
        )
    ]
)

questions_chain  = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    당신은 강력한 포맷 알고리즘입니다.

    시험 문제를 JSON 형식으로 포맷합니다.
    (o)로 답하는 것이 맞습니다.
     
    예시 입력:

    Question: 바다의 색상으로 가장 적절한 것은?
    Answers: 빨강|노랑|초록|파랑(o)
         
    Question: 조지아의 수도는?
    Answers: 바쿠|트빌리시(o)|마닐라|베이루트
         
    Question: 영화 '아바타'의 개봉 연도는?
    Answers: 2007|2001|2009(o)|1998
         
    Question: 시저는 누구인가?
    Answers: 로마 황제(o)|화가|배우|모델
    
     
    예시 출력:
     
    ```json
    {{ "questions": [
            {{
                "question": "바다의 색상으로 가장 적절한 것은?",
                "answers": [
                        {{
                            "answer": "빨강",
                            "correct": false
                        }},
                        {{
                            "answer": "노랑",
                            "correct": false
                        }},
                        {{
                            "answer": "초록",
                            "correct": false
                        }},
                        {{
                            "answer": "파랑",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "조지아의 수도는?",
                "answers": [
                        {{
                            "answer": "바쿠",
                            "correct": false
                        }},
                        {{
                            "answer": "트빌리시",
                            "correct": true
                        }},
                        {{
                            "answer": "마닐라",
                            "correct": false
                        }},
                        {{
                            "answer": "베이루트",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "영화 '아바타'의 개봉 연도는?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "시저는 누구인가?",
                "answers": [
                        {{
                            "answer": "로마 황제",
                            "correct": true
                        }},
                        {{
                            "answer": "화가",
                            "correct": false
                        }},
                        {{
                            "answer": "배우",
                            "correct": false
                        }},
                        {{
                            "answer": "모델",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="퀴즈 만드는중...")
def run_quiz_chain(_docs,topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="위키피디아 검색 중...")
def wikipedia_search(term):
    retriver = WikipediaRetriever(top_k_results=3, lang="ko")
    return retriver.get_relevant_documents(term)

with st.sidebar:
    docs = None
    choice = st.selectbox("당신이 사용할 것을 선택하세요.",(
        "파일","위키피디아"
    ),)
    if choice == "파일":
        file = st.file_uploader("파일을 업로드하세요.",type=["pdf","txt","docx"])
        if file:
            docs = split_file(file)
    elif choice == "위키피디아":
        topic = st.text_input("검색하실 주제를 입력해주세요.")
        if topic:
            docs = wikipedia_search(topic)

if not docs:
    st.markdown("""
    QuizGPT에 오신 것을 환영합니다.
                
    업로드한 위키피디아나 파일에서 퀴즈를 만들어 지식을 테스트하고 공부하는 데 도움을 드리겠습니다.
                
    사이드바에서 파일을 업로드하거나 위키피디아에서 검색하는 것으로 시작하세요.    
    """)
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    st.write(response)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()