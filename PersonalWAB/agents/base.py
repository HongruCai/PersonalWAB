
from typing import Dict, List


class BaseAgent:
    def __init__(self):
        pass

    def act(self, env, index, verbose=False, temperature=0.0, max_steps=30, memory='none', memory_length=10):
        return [0],[0], {}

    def get_messages(self) -> List[Dict[str, str]]:
        return []
