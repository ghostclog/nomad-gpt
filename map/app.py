import streamlit as st
import folium
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOllama
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler


@st.cache_data(show_spinner="질문에 대한 답을 찾는 중...")
def find_data(history):
    st.write("하하",history)

st.title("Map with GPT")

st.markdown("하단에 특정한 역사적 사건 사고에 대해 질문하시면 해당 물음에 대해 답해드립니다.")

history = st.text_input("")
if history:
    find_data(history)