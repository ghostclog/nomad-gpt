from typing import Any, Type
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import DuckDuckGoSearchResults

# 덕덕고 검색 함수
def duckduckgo(inputs):
    ddg = DuckDuckGoSearchResults()
    keyword = inputs["keyword"]
    return ddg.run(f"what is {keyword}")
    
# 위키피디아 검색 함수
def Wikipedia(inputs):
    retriver = WikipediaRetriever(top_k_results=3, lang="ko")
    keyword = inputs["keyword"]
    rs = "Wikipedia\n\n"
    data_list = retriver.invoke(keyword)
    for page_content in data_list:
        rs += f"{page_content.page_content} \n\n"
    return rs

# 파일 저장 함수... 인데 사용 안되는거 같습니다. 
# 우선 assistant 생성 프롬프트를 조정해야 할 거 같은데, 솔직히 말해서 그거까지 못할거같습니다.
def save_the_file(inputs):
    docs = inputs["docs"]
    f = open("./file.txt","w",encoding="utf-8")
    f.write(docs)

# 어시스턴트가 사용 할 함수맵
functions_map = {
    "duckduckgo": duckduckgo,
    "Wikipedia": Wikipedia,
    "save_the_file": save_the_file,
}

# 함수 내용 정의
functions = [
    {
        "type": "function",
        "function": {
            "name": "duckduckgo",
            "description": "If you provide a 'keyword', it is a function that uses DuckDuckGoSearchResults to return the search result as a string based on that keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "This is the 'keyword' used to search on duckduckgo.",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Wikipedia",
            "description": "If you provide a 'keyword', it is a function that uses Wikipedia Retriever to return the search result as a string based on that keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "This is the 'keyword' used to search on Wikipedia.",
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_the_file",
            "description": "Whatever tool you use, if you provide a document about your search results, it's a function that stores them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs": {
                        "type": "string",
                        "description": "Search Results",
                    },
                },
                "required": ["docs"],
            },
        },
    },
]