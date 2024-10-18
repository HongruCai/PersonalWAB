from typing import Any, Dict
from transformers import AutoTokenizer, AutoModel


def add_product_review(data: Dict[str, Any], review: str) -> Dict[str, Any]:
    
    if review:
        return {"review": review}
    return {"review": "No review provided."}


add_product_review.__info__ = {
    "type": "function",
    "function": {
        "name": "add_product_review",
        "description": "Add a full text review. The review should be a string. Do not include extra parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "review": {
                    "type": "string",
                    "description": "The full text content of the review.",
                },
            },
            "required": ["review"],
        },
    },
}
