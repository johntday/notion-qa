import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from streamlit_chat import message
from langchain import OpenAI, FAISS
from langchain.chains import VectorDBQAWithSourcesChain, ConversationalRetrievalChain

import config

config.load_dotenv()
# index = faiss.read_index("docs.index")
# with open("faiss_store.pkl", "rb") as f:
#     store = pickle.load(f)
db = FAISS.load_local("faiss_index", OpenAIEmbeddings())

chain = ConversationalRetrievalChain.from_llm(
    OpenAI(
        temperature=0
    ),
    db.as_retriever(),
)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Blendle Notion QA Bot", page_icon=":robot:")
st.header("Blendle Notion QA Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    # result = chain({"question": user_input})
    result = chain({"question": user_input, "chat_history": st.session_state['history']})

    output = f"Answer: {result['answer']}\nSources: {result['sources']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

    # Add the user's query and the chatbot's response to the chat history
    # st.session_state['history'].append((user_input, result["answer"]))

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
