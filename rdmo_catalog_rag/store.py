from .utils import create_documents, get_embeddings, get_vector_store, invoke_query


def create_vector_store(args):
    documents = create_documents()
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    vector_store.reset_collection()
    vector_store.add_documents(documents)


def query_vector_store(args):
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    response = invoke_query(retriever, args.query)

    print(f"\nQuestion: {args.query}\n\n{response.content}\n")
