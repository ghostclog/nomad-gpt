import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser

from functions import wikipedia_search,split_file,run_quiz_chain,make_llm

docs = None
choice = None
correct_pt = 0
num_of_question = 0
st.set_page_config(
    page_title="Assignment 6",
    page_icon="❓",
)

st.title("Assignment 6")

st.markdown("""
            GPT 과제 6번입니다!
            
            1. 사이드탭에 api키를 입력해주세요!

            2. 사용하실 문서를 선택해주세요!

            3. GPT가 문서를 다 읽고나서, 난이도를 선택해주세요!
            """)


with st.sidebar:
    api_key = st.text_input("하단에 api 키를 입력해주세요.")
    if api_key:
        make_llm(api_key)
        choice = st.selectbox("사용할 문서의 종류를 선택하세요.",(
        "파일","위키피디아"
    ),)
    
    if choice=="파일":
        file = st.file_uploader("파일을 업로드하세요.",type=["pdf","txt","docx"])
        if file:
            docs = split_file(file)
    elif choice=="위키피디아":
        topic = st.text_input("주제를 선택하세요.")
        if topic:
            docs = wikipedia_search(topic)
    #분할선
    st.divider()
    st.write("레포지토리 링크: https://github.com/ghostclog/nomad-gpt-ch9-Assignment")
    st.divider()
    st.header("app.py 내용")
    st.markdown("""
    from functions import wikipedia_search,split_file,run_quiz_chain,make_llm

    docs = None
    choice = None
    correct_pt = 0
    num_of_question = 0
    st.set_page_config(
        page_title="Assignment 6",
        page_icon="❓",
    )

    st.title("Assignment 6")

    st.markdown('''
                GPT 과제 6번입니다!
                
                1. 사이드탭에 api키를 입력해주세요!

                2. 사용하실 문서를 선택해주세요!

                3. GPT가 문서를 다 읽고나서, 난이도를 선택해주세요!
                ''')


    with st.sidebar:
        api_key = st.text_input("하단에 api 키를 입력해주세요.")
        if api_key:
            make_llm(api_key)
            choice = st.selectbox("사용할 문서의 종류를 선택하세요.",(
            "파일","위키피디아"
        ),)
        
        if choice=="파일":
            file = st.file_uploader("파일을 업로드하세요.",type=["pdf","txt","docx"])
            if file:
                docs = split_file(file)
        elif choice=="위키피디아":
            topic = st.text_input("주제를 선택하세요.")
            if topic:
                docs = wikipedia_search(topic)

    if docs:
        st.write("")
        st.write("")
        nan_e_do = st.radio("난이도를 선택하세요!",["쉬움","어려움"])
        if nan_e_do:
            rs = run_quiz_chain(docs,nan_e_do)
            st.write(rs)
            with st.form("questions_form"):
                num_of_question = len(rs["questions"])
                for question in rs["questions"]:
                    st.write(question["question"])
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                    )
                    if {"answer": value, "correct": True} in question["answers"]:
                        st.success("Correct!")
                        correct_pt += 1
                    elif value is not None:
                        st.error("Wrong!")
                button = st.form_submit_button()
        if button:
            if(correct_pt == num_of_question):
                st.balloons()
                st.write("만점입니다!!!")
    """)
    st.divider()
    st.header("function.py 내용입니다.")
    st.markdown("""
    easy_questions_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    '''
        당신은 문제를 만들어주는 ai입니다. 사용자가 데이터를 당신에게 제공한다면, 당신은 반드시 4개 이상의 문제를 만들어내야 합니다.

        난이도의 경우 반드시 "쉬움"로 맞춰야하며, 각 질문에는 4개의 답이 있어야 하고, 그 중 3개는 틀려야 하며, 1개는 정답이여야합니다.

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
    ''',
            )
        ]
    )

    hard_questions_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    '''
        당신은 문제를 만들어주는 ai입니다. 사용자가 데이터를 당신에게 제공한다면, 당신은 반드시 4개 이상의 문제를 만들어내야 합니다.

        난이도의 경우 반드시 "어려움"로 맞춰야하며, 각 질문에는 4개의 답이 있어야 하고, 그 중 3개는 틀려야 하며, 1개는 정답이여야합니다.

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
    ''',
            )
        ]
    )

    formatting_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''
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
    ''',
            )
        ]
    )

    llm = None

    def make_llm(key):
        global llm
        llm = ChatOpenAI(
            temperature=0.5,
            streaming=True,
            openai_api_key=key
            )

    class JsonOutputParser(BaseOutputParser):
        def parse(self, text):
            text = text.replace("```", "").replace("json", "")
            return json.loads(text)
        
    output_parser = JsonOutputParser()

    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)

    @st.cache_data(show_spinner="위키피디아 검색 중...")
    def wikipedia_search(term):
        retriver = WikipediaRetriever(top_k_results=3, lang="ko")
        return retriver.get_relevant_documents(term)

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

    @st.cache_data(show_spinner="퀴즈 만드는중...")
    def run_quiz_chain(_docs,nan_e_do):
        formatting_chain = formatting_prompt | llm
        if nan_e_do == "쉬움":
            questions_chain  = {"context": format_docs} | easy_questions_prompt | llm
        elif nan_e_do == "어려움":
            questions_chain  = {"context": format_docs} | hard_questions_prompt | llm

        chain = {"context": questions_chain} | formatting_chain | output_parser
        return chain.invoke(_docs)
    """)


if docs:
    st.write("")
    st.write("")
    nan_e_do = st.radio("난이도를 선택하세요!",["쉬움","어려움"])
    if nan_e_do:
        rs = run_quiz_chain(docs,nan_e_do)
        st.write(rs)
        with st.form("questions_form"):
            num_of_question = len(rs["questions"])
            for question in rs["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    correct_pt += 1
                elif value is not None:
                    st.error("Wrong!")
            button = st.form_submit_button()
    if button:
        if(correct_pt == num_of_question):
            st.balloons()
            st.write("만점입니다!!!")