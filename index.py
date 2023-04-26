from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import os
os.environ["OPENAI_API_KEY"] = "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"

with open("/home/cloudsuperadmin/scrape-chain/LLM.txt", encoding="ISO-8859-1") as f:
    document = f.read()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(document)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))], persist_directory="db")

docsearch.persist()
docsearch = None