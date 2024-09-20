# 노마드 코더 GPT 스터디 내용 기록 레포지토리입니다.

## 스터디 개요
- 챌린지 및 스터디 진행 기간: 24.08.26~24.09.15
- 스터디 기간 중 사용한 핵심 라이브러리: langcahin(llm 모델 구축 및 챗봇 구현), streamlit(웹페이지 형태의 gui 구현), subprocess(파이썬 프로그램 내부에서 cmd 조작), pydub(AudioSegment를 사용하여 음성 데이터에 정보를 추출 및 음성 데이터 분할)
- 사용 외부 프로그램: ffmpeg(영상 데이터에서 음성 추출)


## 사용 기술 및 구현사항
### ch1~6
- langchain LECL: PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, FewShotChatMessagePromptTemplate을 이용하여 다양한 chain 구축. 이 중 PromptTemplate, ChatPromptTemplate를 이용한 chain을 주력으로 구축하였습니다.
- memory: ConversationBufferMemory, ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory,ConversationKGMemory, 등을 사용하여 대화를 기록 및 저장하는 chatbot을 구현하였습니다.
- retriver: 기본적으로 제공하는 WikipediaRetriver 외에도 사용자에게 문서를 받고, 임베딩하였고. map_rank, map_rerank, refine 등 다양한 기법을 이미 구현된 라이브러리에서 사용하고, 상황에 따라 체인 기법을 커스텀하여 retriver를 구현하였습니다.
### ch7~
- 오프라인 LLM: ollama를 사용하여 오프라인에서도 사용 가능한 llm을 구축(privateGPT)
- streamlit을 이용한 chat 사이트 구현
- agent와 assistant 기능을 활용한 개선된 챗봇 구현(agentGPT 및 졸업 과제)
- openai와 pydub. 그리고, ffmpeg를 활용하여 사용자가 업로드한 영상에서 음성 데이터를 추출. 및 음성 내역을 정리 및 요약하는 GPT 구축(meetingGPT)
- 이전에 배운 retriver를 활용하여 특정 사이트에 대한 데이터를 추출 후, 해당 데이터를 요약하는 챗봇 구현(QuizGPT, 8번 과제) / wikipediaRetriver와 duckduckgo 검색 엔진을 사용한 웹 스크랩핑 라이브러리를 사용하여 구축
