import openai as client
import function_file
import json
import streamlit as st

# api 키 필요
# 키가 들어온 경우 assistant를 생성하고 반환합니다. 만약 api 키가 잘못되어 생성에 실패 할 경우 error라는 문자열을 반환합니다.
@st.cache_data(show_spinner="어시스턴트 생성 중...")
def create_assistant(api_key):
    try:
        client.api_key = api_key
        assistant = client.beta.assistants.create(
            name="Search and Information Assistant",
            instructions="""
            You are an assistant to search and provide information.
            Use the functions and tools that are given or available to you to provide you with the appropriate answers to the questions you ask.
            """,
            model="gpt-4-1106-preview",
            tools=function_file.functions,
        )
        return assistant
    except:
        return "error"
    
# 메세지만 필요
# 메세지를 처리할 thread를 생성합니다.
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

# 어시스턴스, 스레드 필요
# run을 생성합니다.
def create_run(thread,assistant):
    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    )   
    return run
    
# 진행 여부 확인 코드
def get_run(run, thread):
    return client.beta.threads.runs.retrieve(
        run_id=run.id,
        thread_id=thread.id,
    )

# 어시스턴트에게 메세지를 보냅니다. utility의 send_message와는 다릅니다.
def send_message(thread, content):
    return client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=content
    )

# 어시스트가 생성한 메세지를 반환받습니다.
def get_messages(thread):
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    messages = list(messages)
    messages.reverse()
    
    result = ""  # 메시지를 저장할 문자열 변수
    
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

# 어시스턴트가 필요로 하는 툴을 제공하는 함수
def submit_tool_outputs(run, thread):
    outpus = get_tool_outputs(run, thread)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run.id, thread_id=thread.id, tool_outputs=outpus
    )