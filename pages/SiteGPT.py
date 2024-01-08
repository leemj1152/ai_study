import asyncio
import sys
import streamlit as st

from typing import Any
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_transformers import Html2TextTransformer
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document

if "win32" in sys.platform:
    # Windows specific event-loop policy & cmd
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    cmds = [["C:/Windows/system32/HOSTNAME.EXE"]]
else:
    # Unix default event-loop policy & cmds
    cmds = [
        ["du", "-sh", "/Users/fredrik/Desktop"],
        ["du", "-sh", "/Users/fredrik"],
        ["du", "-sh", "/Users/fredrik/Pictures"],
    ]


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs: Any):
        self.message += token
        self.message_box.markdown(self.message)


answers_llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
)

choose_llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
    model="gpt-3.5-turbo-1106",
)

memory = ConversationBufferWindowMemory(
    return_messages=True, k=6, memory_key="chat_history"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def find_history(query):
    histories = load_memory("")
    temp = []
    for idx in range(len(histories) // 2):
        temp.append(
            {
                "input": histories[idx * 2].content,
                "output": histories[idx * 2 + 1].content,
            }
        )

    docs = [
        Document(page_content=f"input:{item['input']}\noutput:{item['output']}")
        for item in temp
    ]
    try:
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        found_docs = vector_store.similarity_search(query)
        candidate = found_docs[0].page_content.split("\n")[1]
        return candidate.replace("output:", "")
    except IndexError:
        return None


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you con't
    just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5. 0 being not helpful to
    the user and 5 being helpful to the user.

    context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    
    Question: {question}
    """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | answers_llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                # "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's
            question.

            Use the answers that have the highest score (more helpful) and
            favor the most recent ones.

            Return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | choose_llm
    condensed = "\n\n".join(
        f"Answer:{answer['answer']}\nSource:{answer['source']}\n" for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", " ")
    )


@st.cache_data(show_spinner="Loading... website")
def load_sitemap(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


@st.cache_data(show_spinner="Loading... website")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = AsyncChromiumLoader([url])
    docs = loader.load_and_split(text_splitter=splitter)
    html2text_transformer.transform_documents(docs)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


def add_memory_message(input, output):
    memory.save_context({"input": input}, {"output": output})


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def session_reset():
    st.session_state["messages"] = []


st.set_page_config(page_title="SiteGPT", page_icon="❓")

st.title("Site GPT")

html2text_transformer = Html2TextTransformer()

content = None
url_type = None

with st.sidebar:
    choice_type = st.selectbox(
        "Choose what you want to use.",
        ("sitemap", "originalUrl"),
    )
    if choice_type == "sitemap":
        url = st.text_input(
            "Write down a Sitemap URL", placeholder="https://example.com/sitemap.xml"
        )
        if url:
            if ".xml" not in url:
                with st.sidebar:
                    st.error("Please write down a Sitemap URL")
            else:
                url_type = "sitemap"

    else:
        url = st.text_input("Write down a URL", placeholder="https://example.com")
        if url:
            url_type = "html"

if url_type == "sitemap":
    session_reset()
    retriever = load_sitemap(url)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about this page...")
    if message:
        send_message(message, "human")
        found = find_history(message)
        if found:
            send_message(found, "ai", save=False)
        else:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            with st.chat_message("ai"):
                result = chain.invoke(message)
                content = result.content.replace("$", "\$")
                add_memory_message(message, content)

elif url_type == "html":
    session_reset()
    retriever = load_website(url)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about this page...")
    if message:
        send_message(message, "human")
        found = find_history(message)
        if found:
            send_message(found, "ai", save=False)
        else:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            with st.chat_message("ai"):
                result = chain.invoke(message)
                content = result.content
                add_memory_message(message, content)

else:
    session_reset()
