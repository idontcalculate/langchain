import gradio as gr
import random
import time
import os
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")
persist_directory = 'rtdocs/gpt-index-readthedocs.io/'
#persist_directory = '.chroma\\index'
db = Chroma(persist_directory=persist_directory)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")
embeddings= OpenAIEmbeddings()

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_message=False
)
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0, max_tokens=-1),
    chain_type="stuff",
    retriver=db.as_retriever(),
    memory=memory,
    get_chat_history= lambda h: h,
    verbose=True,
)
res = qa(
    {
        "question": "what is llama-index?",
        "chat_history": [],
    }
)
print(res)

