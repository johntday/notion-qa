from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter
import config

config.load_dotenv()
loader = NotionDirectoryLoader("Notion_DB")
documents = loader.load()

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
markdown_splitter = MarkdownTextSplitter(chunk_size=1536, chunk_overlap=0)
docs = markdown_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# Store the embeddings vectors using FAISS
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")


# test search
query = "List of groceries"
print(f"TEST SEARCH: {query}")
new_db = FAISS.load_local("faiss_index", embeddings)
docs = new_db.similarity_search(query)
print("similarity_search:")
print(docs[0])
print('\n')

docs_and_scores = new_db.similarity_search_with_score(query)
print("similarity_search_with_score:")
print(docs_and_scores[0])
