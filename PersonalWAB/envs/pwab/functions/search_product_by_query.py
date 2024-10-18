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