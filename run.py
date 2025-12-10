# """
# Unified backend entrypoint for the Mekari Fraud Intelligence Agent.
# Supports:
# - CLI testing
# - RAG-only mode
# - Analytics-only mode
# - Full agent mode
# """

# import argparse
# import os
# import sys

# from dotenv import load_dotenv

# from src.agents.multilingual_agent import handle_query
# from src.rag.rag_chain import run_rag
# from src.analytics.fraud_analytics import run_analytics
# from src.db.supabase_client import DB
# from src.utils.logger import get_logger

# load_dotenv()
# logger = get_logger(__name__)


# # ======================================================
# # Health check for environment & services
# # ======================================================
# def health_check():
#     print("\n🔍 Checking system…")

#     # OpenAI
#     if os.getenv("OPENAI_API_KEY"):
#         print("✔ OpenAI key loaded.")
#     else:
#         print("✘ OPENAI_API_KEY missing!")

#     # Supabase
#     try:
#         DB.ping()
#         print("✔ Supabase connection OK.")
#     except Exception as e:
#         print(f"✘ Supabase error: {e}")

#     print("System check complete.\n")


# # ======================================================
# # CLI REPL
# # ======================================================
# def interactive_shell(mode="agent"):
#     print(f"\n🧠 Starting interactive shell in **{mode}** mode.")
#     print("Type 'exit' to quit.\n")

#     while True:
#         try:
#             q = input("You: ")
#         except KeyboardInterrupt:
#             print("\nExiting.")
#             break

#         if q.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break

#         if mode == "rag":
#             result = run_rag(q)
#             print("Assistant:", result["answer"])
#         elif mode == "analytics":
#             result = run_analytics(q)
#             print("Assistant:", result["answer"])
#         else:
#             result = handle_query(q)
#             if isinstance(result, dict) and "answer" in result:
#                 print("Assistant:", result["answer"])
#             else:
#                 print("Assistant:", result)
#         print()


# # ======================================================
# # MAIN ENTRYPOINT
# # ======================================================
# def main():
#     parser = argparse.ArgumentParser(
#         description="Mekari Fraud Intelligence Agent Backend"
#     )

#     parser.add_argument(
#         "--mode",
#         choices=["agent", "rag", "analytics", "repl", "health"],
#         default="agent",
#         help="Choose operational mode."
#     )

#     parser.add_argument(
#         "--query",
#         type=str,
#         help="Single query to run (non-interactive mode)"
#     )

#     args = parser.parse_args()

#     # --------------------------------------
#     # HEALTH CHECK MODE
#     # --------------------------------------
#     if args.mode == "health":
#         health_check()
#         return

#     # --------------------------------------
#     # INTERACTIVE SHELL
#     # --------------------------------------
#     if args.mode == "repl":
#         interactive_shell("agent")
#         return

#     # --------------------------------------
#     # SINGLE QUERY MODE
#     # --------------------------------------
#     if args.query:
#         q = args.query
#         if args.mode == "rag":
#             print(run_rag(q)["answer"])
#         elif args.mode == "analytics":
#             print(run_analytics(q)["answer"])
#         else:
#             print(handle_query(q))
#         return

#     # --------------------------------------
#     # DEFAULT: LAUNCH INTERACTIVE AGENT
#     # --------------------------------------
#     interactive_shell(mode=args.mode)


# if __name__ == "__main__":
# #     main()
# from src.embeddings.embedder import embedding_model
# v = embedding_model.embed_one("test")
# print(len(v))
from src.rag.retriever import retrieve_docs

query = "Which merchants or merchant categories exhibit the highest incidence of fraudulent transactions?"
docs = retrieve_docs(query)

print("Retrieved:", len(docs))
for d in docs[:3]:
    print("-" * 40)
    print("Page:", d["page"])
    print("Source:", d["source_name"])
    print("Snippet:", d["content"][:300])
