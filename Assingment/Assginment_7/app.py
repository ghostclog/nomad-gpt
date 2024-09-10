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
    page_title="Assignment 7",
    page_icon="❓",
)

st.title("Assignment 7")

st.markdown("""
            GPT 과제 7번입니다!
            
            1. 사이드탭에 api키를 입력해주세요!

            2. 확인하실 사이트 링크를 알려주세요!

            3. GPT가 사이트를 다 확인하면, 해당 사이트에 대한 질문을 해주세요!
            """)


with st.sidebar:
    api_key = st.text_input("하단에 api 키를 입력해주세요.")
    if api_key:
        make_llm(api_key)
        
    #분할선
    st.divider()
    st.write("레포지토리 링크: ")
    st.divider()
    st.header("app.py 내용")
    st.markdown("""
    """)
    st.divider()
    st.header("function.py 내용입니다.")
    st.markdown("""
    """)


if docs:
    st.write("")
    st.write("")
    