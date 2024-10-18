from typing import Any, Dict, List
from pyserini.search.lucene import LuceneSearcher
import json
import sys

Product_Prompt = '''
Product Details:
- Title: <TITLE>
- Parent Asin: <PARENT_ASIN>
- Average Rating: <AVERAGE_RATING>
- Rating Number: <RATING_NUMBER>
- Price: <PRICE>
- Store: <STORE>
- Features: <FEATURES>
- Description: <DESCRIPTION>
- Details: <DETAILS>
- Category: <CATEGORY>
'''

def pretty_product(product: dict) -> str:
    res = Product_Prompt.replace('<CATEGORY>', str(product['main_category']))
    res = res.replace('<TITLE>', product['title'])
    res = res.replace('<AVERAGE_RATING>', str(product['average_rating']))
    res = res.replace('<RATING_NUMBER>', str(product['rating_number']))
    res = res.replace('<FEATURES>', str(product['features']))
    res = res.replace('<DESCRIPTION>', str(product['description']))
    res = res.replace('<PRICE>', str(product['price']))
    res = res.replace('<STORE>', str(product['store']))
    res = res.replace('<DETAILS>', json.dumps(product['details']))
    res = res.replace('<PARENT_ASIN>', product['parent_asin'])
    return res

def get_product_details_by_asin(data: Dict[str, Any], product_asin: str) -> str:
    product = data['all_products'].get(product_asin)
    if product:
        return pretty_product(product)
    return 'Product not found.'
    

get_product_details_by_asin.__info__ = {
    "type": "function",
    "function": {
        "name": "get_product_details_by_asin",
        "description": "Get the product details by product parent ASIN. The full details of the product will be returned.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_asin": {
                    "type": "string",
                    "description": "The parent ASIN of the product to look up.",
                },
            },
            "required": ["product_asin"],
        },
    },
}
