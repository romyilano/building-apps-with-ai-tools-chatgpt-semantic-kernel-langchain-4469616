from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
# download all text from wikipeida 

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Tea")
documents = loader.load()

## chunk size is number of characters, overlap is between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

docsearch = FAISS.from_documents(texts, embeddings)
# a q & a retrieval chatbot using 
qa = RetrievalQA.from_chain_type(llm = ChatOpenAI(model_name="gpt-4o"), 
                                 chain_type = "stuff",
                                retriever = docsearch.as_retriever())
while True:
  query = input("ask a question about tea\n")
  if input == "quit":
    break
  print(qa.run(query))
# print(qa.run("Where did tea originate?"))