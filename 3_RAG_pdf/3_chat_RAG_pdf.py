# Создаем папку 3_RAG_pdf
# В папке 3_RAG_pdf создаем папку rag-dataset - в нее загружаем наши исходные pdf-файлы
# Внутри папки rag-dataset создаем еще две папки:
# finance (загружаем pdf-файлы соотвествующей тематики)
# sport (загружаем pdf-файлы соотвествующей тематики)
# В 3_RAG_pdf cоздаем новый файл .env
# В папке 3_RAG_pdf cоздаем новый файл 3_chat_RAG_pdf.py

# Переходим на сайт LangSmith, автризируемся через GitHub,
# на сайте LangSmith: Personal -> Settings -> API Keys -> Create API Key -> 
# Description = rag_pdf_token -> Key Type = Personal Access Token -> Create API Key,
# .env: LANGCHAIN_API_KEY = "rag_pdf_token"

# Терминал VSCODE:
#(base) D:\ollama_chatbot>conda activate conda_env_rae
#(conda_env_rae) D:\ollama_chatbot>ollama list
#(conda_env_rae) D:\ollama_chatbot>ollama run llama3.2:latest # запускаем ранее установленную Модель ИИ (llama3.2:3b)
#(conda_env_rae) D:\ollama_chatbot>ollama pull nomic-embed-text # скачиваем Модель для создания эмбедингов (nomic-embed-text)
#(conda_env_rae) D:\ollama_chatbot>ollama list # список существующих на компе Моделей ИИ (llama3.2:3b, nomic-embed-text)
#(conda_env_rae) D:\ollama_chatbot>pip install -U langchain-community faiss-cpu langchain-huggingface pymupdf tiktoken langchain-ollama python-dotenv


import os
import warnings
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()
os.environ['LANGCHAIN_PROJECT']

##################################################################
# Из папки "rag-dataset" подгружаем все пути наших pdf-файлы:
import os

pdfs = []
for root, dirs, files in os.walk('rag-dataset'):
    for file in files:
        if file.endswith('.pdf'):
            pdfs.append(os.path.join(root, file))

##################################################################
# Берем все наши pdfs-файлы и прогоняем их через PyMuPDFLoader,
# в итоге, теперь все содержимое наших pdfs-файлов постранично находится в текстовом виде в переменной docs
from langchain_community.document_loaders import PyMuPDFLoader
docs = [] # 
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    docs.extend(pages)

#print(len(docs)) # - суммарное кол-во страниц во всех наших pdf-файлах

##################################################################
# Теперь разбиваем весь текст, находящийся в переменной docs на фрагменты (chunks),
# использя для этого Рекурсивный Разделитель Текста (Recursive Text Splitter LangChain)
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# chunk_size=1000 - кол-во символов внутри одного фрагмента (chunk),
# chunk_overlap=100 - будет происходить это наложение двух фрагментов друг на друга, чтобы не было потери контекста
chunks = text_splitter.split_documents(docs) # все фрагменты
#print(len(chunks)) # - суммарное кол-во фрагментов, каждый из которых состоит примерно из 1000 символов

##################################################################
# Итого: Текущая архитектура нашего приложения:
# - мы взяли 3 документа в pdf-формате,
# - далее с помощью pyMuPDF мы все эти pdf-документы превратили в один большой текстовый документ (docs), 
# состоящий из len(docs) страниц,
# - затем этот большой текстовый документ (docs) мы с помощью Рекурсивного Разделения Текста 
# поделили на фрагменты (chunks).

##################################################################
# Теперь из фрагментов (chunks) нужно сделать эмбединги, используя спец. модель = "nomic-embed-text",
# предварительно нужно (conda_env_rae) D:\ollama_chatbot>ollama pull nomic-embed-text
# Находим длину (len_single_vector) одного из эмбедингов = 768, 
# т.е. в каждом векторе будет 768 элементов.
# Cоздаем индекс с нормой L2 (index).
# Создаем Хранилище Векторов (vector_store)
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

base_url="http://localhost:11434" # порт, который используется для работы в локальной системе
embeddings_model_name = 'nomic-embed-text' # спец. модель = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=embeddings_model_name, base_url=base_url)
single_vector = embeddings.embed_query(chunks[0].page_content)
len_single_vector =  len(single_vector)
index = faiss.IndexFlatL2(len(single_vector)) # создаем индекс с нормой L2
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(), # мы будем хранить в памяти
    index_to_docstore_id={} # пустой словарь,
    # но как только мы добавим наш документ в Хранилище Векторов,
    # все идентификаторы документов будут находиться в этом словаре
)

vector_store.add_documents(documents=chunks)
#ids = vector_store.add_documents(documents=chunks) # идентификатор для каждого фрагмента (chunk) 
# в нашем Вектороном Хранилище (vector_store). 
# Кол-во идентфикаторов (len(ids)) равно кол-ву фрагментов (len(chunks))

##################################################################
# Задаем вопрос (question), и этот вопрос будет использоваться для
# извлечения соответствующего фрагмента (chunk) из общего кол-ва фрагментов (len(chunks)).
# Какой бы вопрос ни был (question), он будет превращен с числовое вектороное предствление также с помощью 'nomic-embed-text',
# и далее с этот вопрос, представленный уже числовым векторным видом будет сопоставлен 
# с нашими фрагментами (chunks) из нашей Векторного Хранилища (vector_store),
# и будут искаться сходства используя косинусное сходство,
# если будут найдены подходящие фрагменты (chunks), то они будут нам возвращены.

# Процесса Извлечения (Retrieval):
# Делаем из нашего Векторного Хранилища поисковик:
# - алгоритм поиска (search_type), например 'mmr' или 'similarity'
# а далее мы уже в  search_kwargs укажем важные настройки:
# - 'fetch_k': 100 - будет выбрано 100 релевантных документов, которые будет переданы алгоритму поиска для ранжирования,
# - 'k': 3 - итоговое кол-во выходных самых релевантных фрагментов, которые нам выдадут
# - 'lambda_mult': 1 - вернет только те выходные данные, которые имеют отношение к нашему запросу.
# (если 'lambda_mult': 0 - вернет разнообразный результат).
# docs_relevantnye_for_my_question - и теперь нам выдадут только 3 наиболее подходящих под наш вопрос (question) фрагмента
# Проходим по содержимому этих фрагментов (relevantnyi_chunk.page_content), которые больше всего 
# могут помочь в формулировании ответа на наш вопрос (question), 
# и объединяем их в один файл one_file_for_relevantnye_chunks_for_my_question


#question = "What are the regulations for modern fuel?" # sport: Formula 1
#question = "Real Madrid is tier 1 club?" # sport: football
question = "What is the impact on global trade and the economy?" # finance
retriever_search = vector_store.as_retriever(search_type="similarity", search_kwargs = {'k': 3, 
                                                                          'fetch_k': 100,
                                                                          'lambda_mult': 1})
docs_relevantnye_for_my_question = retriever_search.invoke(question)
# len(docs_relevantnye_for_my_question) = 3 - получаем итоговое выходное общее кол-во документов, 
# которые нам нужно вернуть (k=3).


def func_format_docs(relevantnye_chunks_for_my_question):
    one_file_for_relevantnye_chunks_for_my_question = "\n\n".join([relevantnyi_chunk.page_content for relevantnyi_chunk in relevantnye_chunks_for_my_question])
    return one_file_for_relevantnye_chunks_for_my_question

##################################################################
# Теперь нужно передать этот единый файл (one_file_for_relevantnye_chunks_for_my_question) с наиболее
# подходящими для ответа на вопрос (question) фрагментами, 
# а также и непосредственно сам наш вопрос (question) уже непосредственно в саму Модель в ИИ (model_name = "llama3.2:latest"), 
# чтобы уже Модель ИИ могла обработать наш вопрос (question)
# и дать нам уже четкий ответ, основываясь на имеющихся у нее помощниках-фрагментах (one_file_for_relevantnye_chunks_for_my_question).
# system_prompt - создаем свой собственный шаблон-подсказку для переписки с ИИ,
# т.е. в этом вводном предожении (в напутствии для нашей Модели ИИ) сказано, 
# что ответ должен быть расписан по пунктам, если Модель ИИ не знаете ответа, то пусть так и скажет.

from langchain import hub
from langchain_core.output_parsers import StrOutputParser # чтобы получить итоговый строчной вывод результата от Модели ИИ
from langchain_core.runnables import RunnablePassthrough # чтобы передать вопрос напрямую нашей Модели ИИ
from langchain_core.prompts import ChatPromptTemplate 

from langchain_ollama import ChatOllama

model_name = "llama3.2:latest"

AI_model = ChatOllama(model= model_name, base_url=base_url)

system_prompt = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
"""
system_prompt = ChatPromptTemplate.from_template(system_prompt)

# Формируем Цепочку Запросов (RAG Chain):
# - предоставляем файл с релевантным контектом = one_file_for_relevantnye_chunks_for_my_question ввиде функции func_format_docs,
# - "question": RunnablePassthrough() - предаем вопрос в Модель ИИ
# - предаем шаблон-подсказку (system_prompt) для переписки с Моделью ИИ,
# - предаем нашу Модель ИИ (AI_model)
# - StrOutputParser() - чтобы получить итоговый строчной вывод результата от Модели ИИ

rag_chain = (
    {"context": retriever_search|func_format_docs, "question": RunnablePassthrough()}
    | system_prompt
    | AI_model
    | StrOutputParser()
)

output = rag_chain.invoke(question)
print("Question: ", question)
print("Answer: ")
print(output) # получаем итоговый строчной вывод результата от Модели ИИ

