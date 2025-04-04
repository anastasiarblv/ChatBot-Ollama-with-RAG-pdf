# Терминал VSCODE:
#(conda_env_rae) D:\ollama_chatbot>pip install langchain
#(conda_env_rae) D:\ollama_chatbot>pip install -qU langchain-ollama 
#(conda_env_rae) D:\ollama_chatbot>pip install streamlit 
#(conda_env_rae) D:\ollama_chatbot\2_chatbot_with_history>streamlit run 2_chatbot_with_history.py

# вопрос - "What is my name?"
# ответ  - "I don't know your real name."

# вопрос - "What is the capital of Russia?"
# ответ  - "Moscow is Russia's main city capital."

# вопрос - "My name is Anastasia"
# ответ  - "Nice to meet you Anastasia!"

# вопрос - "What is my name?"
# ответ  - "Your name is Anastasia, I remember!"

import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# SystemMessagePromptTemplate - создаем свой собственный шаблон-подсказку для переписки с ИИ,
# т.е. в этом вводном предожении (в напутствии для нашей Модели ИИ) сказано, 
# что ответ должен быть расписан по пунктам, если Модель ИИ не знаете ответа, то пусть так и скажет.
# HumanMessagePromptTemplate - наши сообщения (вопросы, question) в чат.
# AIMessagePromptTemplate - ответы от ИИ на наши сообщения (answer) в чат.
# Затем нам нужно все это объединить в один шаблон чата, поэтому нужен еще ChatPromptTemplate.


system_prompt = SystemMessagePromptTemplate.from_template("You are a helpful AI Assistant.You work as teacher for 5th grade students." 
                                                           "You explain things in 6 words.")

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [] # при первом запуске нашего приложения Streamlit,
# у нас будет пустая история чата, т.к. мы еще никакой запрос туда не отправили. 

# Но если мы постоянно будем вести с наши чат-ботом переписку, то история нашего чата сформируется.
# И вот код, чтобы получить эту историю чата:
def get_history():
    # При первом запуске будет только system_prompt 
    chat_history = [system_prompt] # помещаем сюда system_prompt 
    for chat in st.session_state['chat_history']: # смотрим историю моего чата, если там что-то есть уже:
        # сначала вопросы от нас
        question = HumanMessagePromptTemplate.from_template(chat['user'])
        chat_history.append(question)
        # потом ответы от ИИ
        answer = AIMessagePromptTemplate.from_template(chat['assistant'])
        chat_history.append(answer)
    return chat_history


def generate_answer(chat_history):
    base_url = "http://localhost:11434/"
    model_name = "llama3.2:latest"
    model = ChatOllama(model= model_name, base_url=base_url)
    chat_template = ChatPromptTemplate.from_messages(chat_history)
    chain = chat_template|model|StrOutputParser()
    answer = chain.invoke({})
    return answer

st.title("Приложение: чат-бот, дающий ответ в т.ч. на основе предыдущего с ним общения")
with st.form("llm-form"):
    question_text = st.text_area("Введите свой вопрос")  
    submit = st.form_submit_button("Отправить вопрос")

if question_text and submit:
    with st.spinner("Вопрос в обработке..."):
        question = HumanMessagePromptTemplate.from_template(question_text)
        chat_history = get_history() # получаем историю чата
        chat_history.append(question) # в нашу уже имеющуюся историю чата добавляем новый наш запрос в этот чат
        answer = generate_answer(chat_history) # ответ основан на контексте всей нашей истории чата
        st.session_state['chat_history'].append({'user': question_text, 'assistant': answer})


st.write('## История чата')
for chat in reversed(st.session_state['chat_history']):
       st.write(f"**:adult: Вопрос пользователя**: {chat['user']}")
       st.write(f"**:brain: Ответ ассистента**: {chat['assistant']}")
       st.write("---")