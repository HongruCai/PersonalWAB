from typing import Any, Dict, List
from pyserini.search.lucene import LuceneSearcher
import json
import sys
import os

FOLDER_PATH = os.path.dirname(__file__)
def load_searcher():
    print('BM25 searcher loaded.')
    return LuceneSearcher(os.path.join(FOLDER_PATH, 'search', 'indexes'))

def search_product_by_query(data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:

    hits = bm25_searcher.search(query)
    results = []
    for i in range(0, len(hits)):
        docid = hits[i].docid  
        doc = bm25_searcher.doc(docid) 
        item = doc.raw()
        #item = hits[i].raw()
        item = json.loads(item)
        p = item['contents']
        results.append('Product ' + str(i) + ': \n' + p + '\n')

    if results:
        return results
    return []


search_product_by_query.__info__ = {
    "type": "function",
    "function": {
        "name": "search_product_by_query",
        "description": "Search for products by a query string. The information of the top 10 products will be returned.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query string to search for products, such as 'laptop' or 'phone'.",
                },
            },
            "required": ["query"],
        },
    },
}

bm25_searcher = load_searcher()

# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json
# import os
# from typing import Any, Dict, List

# FOLDER_PATH = os.path.dirname(__file__)

# class DenseRetriever:
#     def __init__(self, index_path: str, corpus_path: str, model_name: str = "all-MiniLM-L6-v2"):

#         self.index_path = index_path
#         self.corpus_path = corpus_path
#         self.model = SentenceTransformer(model_name)
#         self.index = self._load_or_build_index()

#     def _load_or_build_index(self):

#         if os.path.exists(self.index_path):
#             print("Loading existing FAISS index...")
#             index = faiss.read_index(self.index_path)
#         else:
#             print("Building FAISS index from corpus...")
#             with open(self.corpus_path, "r", encoding="utf-8") as f:
#                 corpus = [json.loads(line.strip()) for line in f]
#             documents = [doc["contents"] for doc in corpus]

#             embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
#             dimension = embeddings.shape[1]

#             index = faiss.IndexFlatL2(dimension)
#             index.add(embeddings)

#             faiss.write_index(index, self.index_path)
#         return index

#     def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:

#         query_embedding = self.model.encode([query], convert_to_numpy=True)
#         distances, indices = self.index.search(query_embedding, top_k)

#         # 加载语料库
#         with open(self.corpus_path, "r", encoding="utf-8") as f:
#             corpus = [json.loads(line.strip()) for line in f]

#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx != -1:
#                 doc = corpus[idx]
#                 results.append('Product'+ str(idx) + ': \n' + doc["contents"] + '\n',
#                     )
#         return results


# dense_retriever = DenseRetriever(
#     index_path=os.path.join(FOLDER_PATH, "search", "faiss_index"),
#     corpus_path="data/Products/all_products.jsonl",
# )

# def search_product_by_query(data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
#     results = dense_retriever.search(query, top_k=10)
#     if results:
#         return results
#     return []

# search_product_by_query.__info__ = {
#     "type": "function",
#     "function": {
#         "name": "search_product_by_query",
#         "description": "Search for products by a query string using dense retrieval. The information of the top 10 products will be returned.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": "The query string to search for products, such as 'laptop' or 'phone'.",
#                 },
#             },
#             "required": ["query"],
#         },
#     },
# }
