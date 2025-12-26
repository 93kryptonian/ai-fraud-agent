
from src.rag.retriever import retrieve_docs

query = "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?"
docs = retrieve_docs(query)

print("Retrieved:", len(docs))
for d in docs[:3]:
    print("-" * 40)
    print("Page:", d["page"])
    print("Source:", d["source_name"])
    print("Snippet:", d["content"][:300])
