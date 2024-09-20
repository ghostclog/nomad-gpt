import streamlit as st
import assistant
import time
import utility

# 웹 기본 설정
st.set_page_config(
    page_title="Assignment 6",
    page_icon="❓",
)

# 웹 메인 페이지 타이틀
st.title("Nomad GPT Graduate")

# assistant 객체를 세션에 사용하기 위한 사전 정의
if 'assistant_obj' not in st.session_state:
    st.session_state.assistant_obj = None
    st.session_state["messages"] = []
with st.sidebar:
    api_key = st.text_input("하단에 open api 키를 입력해주세요.") # API 키 받아오기.
    st.session_state.assistant_obj = assistant.create_assistant(api_key) # 받아온 키로 assistant 생성
    # 링크 및 코드 확인하는 부분
    option = st.radio(
        '확인하러는 링크 및 문서를 선택하세요. 주석은 제외되어 있습니다.',
        ('repository link', 'app.py', 'assistant.py','utility.py','function_file.py')
    )
    if option == "repository link":
        st.markdown("""
            https://github.com/ghostclog/nomad-gpt-graduate
        """)
    elif option == "app.py":
        st.markdown("""
            import streamlit as st
import assistant
import time
import utility

st.set_page_config(
    page_title="Assignment 6",
    page_icon="❓",
)

st.title("Nomad GPT Graduate")

if 'assistant_obj' not in st.session_state:
    st.session_state.assistant_obj = None
with st.sidebar:
    api_key = st.text_input("하단에 open api 키를 입력해주세요.")
    st.session_state.assistant_obj = assistant.create_assistant(api_key)
                    
if(st.session_state.assistant_obj == "error"):
    st.warning("어시스턴트 생성되지 않았습니다! 키를 입력하지 않으셨거나, 잘못된 키를 입력하셨습니다.")
    st.session_state["messages"] = []
else:
    if st.session_state.assistant_obj:
        utility.paint_history(st.session_state["messages"])
        msg = st.chat_input("메세지를 입력해주세요!")
        if msg:
            utility.send_message(msg,"human",st.session_state["messages"])
            thread = assistant.create_thread(msg)
            if thread:  
                with st.spinner('맞춤 런 생성 중...'):
                    run = assistant.create_run(thread, st.session_state.assistant_obj)
                if run:
                    with st.spinner('처리중...'):
                        while(assistant.get_run(run, thread).status != "completed"):
                            time.sleep(1)
                            if(assistant.get_run(run, thread).status == "requires_action"):
                                assistant.submit_tool_outputs(run, thread)
                                time.sleep(1)
                    if assistant.get_run(run, thread).status == "completed":
                        messages = assistant.get_messages(thread)  # 메시지를 가져옴
                        utility.send_message(messages,"ai",st.session_state["messages"])         
        """)
    elif option == "assistant.py":
        st.markdown("""
            import openai as client
import function_file
import json
import streamlit as st

@st.cache_data(show_spinner="어시스턴트 생성 중...")
def create_assistant(api_key):
    try:
        client.api_key = api_key
        assistant = client.beta.assistants.create(
            name="Search and Information Assistant",
            instructions='''
            You are an assistant to search and provide information.
            Use the functions and tools that are given or available to you to provide you with the appropriate answers to the questions you ask.
            ''',
            model="gpt-4-1106-preview",
            tools=function_file.functions,
        )
        return assistant
    except:
        return "error"
    
@st.cache_data(show_spinner="맞춤 스레드 생성 중...")
def create_thread(msg):
    thread = client.beta.threads.create(
    messages=[
            {
                "role": "user",
                "content": f"{msg}",
            }
        ]
    )
    return thread

def create_run(thread,assistant):
    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    )   
    return run
    
def get_run(run, thread):
    return client.beta.threads.runs.retrieve(
        run_id=run.id,
        thread_id=thread.id,
    )


def send_message(thread, content):
    return client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=content
    )


def get_messages(thread):
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    messages = list(messages)
    messages.reverse()
    
    result = ""
    
    for message in messages:
        result += f"{message.content[0].text.value}\n"
    
    return result  # 메시지를 반환


def get_tool_outputs(run, thread):
    run = get_run(run, thread)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": function_file.functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run, thread):
    outpus = get_tool_outputs(run, thread)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run.id, thread_id=thread.id, tool_outputs=outpus
    )
        """)
    elif option == "utility.py":
        st.markdown("""
            import streamlit as st

def save_message(message, role,message_obj):
    message_obj.append({"message": message, "role": role})

def send_message(message, role,message_obj,save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role,message_obj)

def paint_history(message_obj):
    for message in message_obj:
        send_message(
            message["message"],
            message["role"],
            message_obj,
            save=False,
        )


        """)
    elif option == "function_file.py":
        st.markdown("""
            from typing import Any, Type
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import DuckDuckGoSearchResults


def duckduckgo(inputs):
    ddg = DuckDuckGoSearchResults()
    keyword = inputs["keyword"]
    return ddg.run(f"what is {keyword}")
    

def Wikipedia(inputs):
    retriver = WikipediaRetriever(top_k_results=3, lang="ko")
    keyword = inputs["keyword"]
    rs = "Wikipedia\n\n"
    data_list = retriver.invoke(keyword)
    for page_content in data_list:
        rs += f"{page_content.page_content} \n\n"
    return rs

def save_the_file(inputs):
    docs = inputs["docs"]
    f = open("./file.txt","w",encoding="utf-8")
    f.write(docs)


functions_map = {
    "duckduckgo": duckduckgo,
    "Wikipedia": Wikipedia,
    "save_the_file": save_the_file,
}


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
        """)

# assistant가 생성되지 않은 경우. 혹은 잘못된 API키를 입력하여 생성 과정에서 에러가 발생할 경우 생성되는 문구
if(st.session_state.assistant_obj == "error"):
    st.warning("어시스턴트 생성되지 않았습니다! 키를 입력하지 않으셨거나, 잘못된 키를 입력하셨습니다.")
    # 기본적으로 작동하는 구간이기때문에 동시에 메세지를 저장해줄 메세지 세션 또한 같이 생성. 
    # 해당 코드는 DocumentGPT 부분에서 사용한 코드 복붙
else:
    # assistant 생성 성공시.
    if st.session_state.assistant_obj:
        # 세션에 저장되어 있던 메세지 화면에 출력 및 사용자에게 메세지를 받음
        utility.paint_history(st.session_state["messages"])
        msg = st.chat_input("메세지를 입력해주세요!")
        if msg:
            # 메세지를 세션에 저장하고, 동시에 해당 메세지를 처리해줄 thread 생성.
            utility.send_message(msg,"human",st.session_state["messages"])
            thread = assistant.create_thread(msg)
            if thread:
                # run 생성
                with st.spinner('맞춤 런 생성 중...'):
                    run = assistant.create_run(thread, st.session_state.assistant_obj)
                if run:
                    # assistant.get_run(run, thread).status를 자주 사용 할 경우 에러가 발생 할 수 있어서 sleep을 사용했습니다.
                    # run의 상태가 completed이 될때까지 기다리며, 만약 requires_action인 경우.
                    # submit_tool_outputs를 사용하여 필요한 함수를 제공합니다.
                    with st.spinner('처리중...'):
                        while(assistant.get_run(run, thread).status != "completed"):
                            time.sleep(1)
                            if(assistant.get_run(run, thread).status == "requires_action"):
                                assistant.submit_tool_outputs(run, thread)
                                time.sleep(1)
                    # run의 상태가 completed인 경우 while문을 빠져나오고, 동시에 assistant가 생성한 메세지를 저장함과 동시에 띄워줍니다.
                    if assistant.get_run(run, thread).status == "completed":
                        messages = assistant.get_messages(thread)  # 메시지를 가져옴
                        utility.send_message(messages,"ai",st.session_state["messages"])
