from langchain import OpenAI, FAISS
from langchain.chains import VectorDBQAWithSourcesChain, ConversationalRetrievalChain
import pickle
import argparse

from langchain.embeddings import OpenAIEmbeddings
import config
from utils import get_chat_history

config.load_dotenv()
parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

db = FAISS.load_local("faiss_index", OpenAIEmbeddings())

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(
        temperature=0
    ),
    retriever=db.as_retriever(),
    get_chat_history=get_chat_history,
)
chat_history = []
result = chain({"question": args.question, "chat_history": chat_history})
print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")
