# Терминал VSCODE:
#(base) D:\ollama_chatbot>conda activate conda_env_rae
#(conda_env_rae) D:\ollama_chatbot>ollama list
#(conda_env_rae) D:\ollama_chatbot>ollama run llama3.2:latest
#(conda_env_rae) D:\ollama_chatbot>ollama
# Создаем новый файл 1_simple_chatbot.py
#(conda_env_rae) D:\ollama_chatbot>pip install langchain
#(conda_env_rae) D:\ollama_chatbot>pip install -qU langchain-ollama 
#(conda_env_rae) D:\ollama_chatbot>pip install streamlit 

# Запускаем код:
# (conda_env_rae) D:\ollama_chatbot\1_simple_chatbot>streamlit run 1_simple_chatbot.py

import streamlit as st
from langchain_ollama import ChatOllama

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []


st.title("Приложение: чат-бот, дающий ответ без учета предыдущего с ним общения")
with st.form("llm-form"):
    question_text = st.text_area("Введите свой вопрос")
    submit = st.form_submit_button("Отправить вопрос")

def generate_answer(question_text):
    base_url = "http://localhost:11434/"
    model_name = "llama3.2:latest"
    model = ChatOllama(model= model_name, base_url=base_url)
    answer = model.invoke(question_text)
    return answer.content

if question_text and submit:
    with st.spinner("Вопрос в обработке..."):
        answer = generate_answer(question_text)
        st.session_state['chat_history'].append({"user": question_text, "ollama": answer})
        st.write(answer)

st.write("## История чата")
for chat in reversed(st.session_state['chat_history']):
    st.write(f"**🧑 Вопрос пользователя**: {chat['user']}")
    st.write(f"**🧠 Ответ ассистента**: {chat['ollama']}")
    st.write("---")