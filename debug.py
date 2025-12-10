from src.rag.retriever_direct import retrieve_docs

docs = retrieve_docs("merchant categories fraud", top_k=5, source="Bhatla")

print(docs[0])

