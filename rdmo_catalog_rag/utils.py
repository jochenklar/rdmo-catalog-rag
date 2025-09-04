import logging
import os
import xml.etree.ElementTree as et
from pathlib import Path
from types import SimpleNamespace

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

SYSTEM_TEMPLATE = """
You are a helpful assistant that answers based on provided XML documents.
Each document describes an RDMO element.
For every element you find, give some information and the exact uri.
"""

CONTEXT_TEMPLATE = """
Use the following documents to answer the question:

{context}
"""

USER_TEMPLATE = """
Question: {content}
"""

LLM = {
    "model": "gpt-4.1-mini"
}


def get_settings():
    try:
        from chatbot import settings
        return settings
    except ModuleNotFoundError:
        return SimpleNamespace(
            SYSTEM_TEMPLATE=SYSTEM_TEMPLATE,
            CONTEXT_TEMPLATE=CONTEXT_TEMPLATE,
            USER_TEMPLATE=USER_TEMPLATE,
            LLM=LLM
        )


def create_documents():
    xml_base_path = Path(os.getenv("XML_BASE_PATH")).expanduser() / "rdmorganiser"

    xml_paths = [
        path
        for path in xml_base_path.rglob("*")
        if path.suffix == ".xml"
    ]

    documents = []
    for xml_path in xml_paths:
        logging.info("Processing xml_path=%s", xml_path)

        tree = et.parse(xml_path)
        root = tree.getroot()
        documents += get_documents(root)

    return documents


def get_tag(node):
    return node.tag.split("}")[-1]


def get_uri(node):
    return node.attrib.get(r"{http://purl.org/dc/elements/1.1/}uri")


def get_text(node):
    if node.text and node.text.strip():
        return node.text.strip()


def get_documents(node):
    documents = []

    # everything with an uri is a document
    if get_uri(node):
        logging.debug(f"tag={get_tag(node)} uri={get_uri(node)}")

        content = ""
        for child in node:
            child_tag = get_tag(child)
            child_uri = get_uri(child)
            child_text = get_text(child)
            if child_uri:
                content += f"{child_tag}: {child_uri}\n\n"
            elif child_text:
                content += f"{child_tag}: {child_text}\n\n"

        if content.strip():
            documents.append(Document(page_content=content, metadata={"tag": get_tag(node), "uri": get_uri(node)}))

    for child in node:
        documents.extend(get_documents(child))

    return documents


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large")


def get_vector_store(embedding_function):
    persist_directory = Path(os.getenv("VECTOR_STORE_PATH")).expanduser()
    return Chroma(embedding_function=embedding_function, persist_directory=persist_directory)


def retrieve_documents(retriever, message_content):
    documents = retriever.invoke(message_content)
    return "\n\n".join([document.page_content for document in documents])


def invoke_query(retriever, query):
    settings = get_settings()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", settings.SYSTEM_TEMPLATE),
            ("system", settings.CONTEXT_TEMPLATE),
            ("user", settings.USER_TEMPLATE)
        ]
    )

    llm = ChatOpenAI(**settings.LLM)

    chain = prompt | llm

    context = retrieve_documents(retriever, query)

    inputs = {
        "context": context,
        "content": query
    }

    return chain.invoke(inputs)
