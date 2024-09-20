import streamlit as st

# 내용 저장
def save_message(message, role,message_obj):
    message_obj.append({"message": message, "role": role})

# 메세지 보내기. 만약에 신규 메세지면, session에 저장함.
def send_message(message, role,message_obj,save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role,message_obj)

# 기존 대화 로그 출력해주는 부분 / 이미 저장된 대화 로그이기에 때문에 save는 false로 지정
def paint_history(message_obj):
    for message in message_obj:
        send_message(
            message["message"],
            message["role"],
            message_obj,
            save=False,
        )

