import chainlit as cl
from chatbot.adapter import OpenAILangChainAdapter

from .utils import get_embeddings, get_vector_store, retrieve_documents


class Adapter(OpenAILangChainAdapter):

    def init_retriever(self):
        embeddings = get_embeddings()
        vector_store = get_vector_store(embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})

    async def on_chat_start(self):
        retriever = self.init_retriever()
        chain = self.init_chain()

        cl.user_session.set("retriever", retriever)
        cl.user_session.set("chain", chain)

    async def fetch_context(self, message):
        retriever = cl.user_session.get("retriever")
        return retrieve_documents(retriever, message.content)
