from typing import Any, Dict, List


def stop(data: Dict[str, Any],) -> str:
    return '###STOP###'


stop.__info__ = {
    "type": "function",
    "function": {
        "name": "stop",
        "description": "This is a no-op function used to indicate that no further action is required.",
        "parameters": {}
    },
}
