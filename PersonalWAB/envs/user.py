
import os
import abc
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Callable
import sys
import json

DIVERSITY = {
    "High": "A Highly Adventurous Explorer eager to discover diverse products across categories. You often seek recommendations, purchase a wide variety of items with varying ratings, and the your own ratings may often differ from the average. Your reviews are detailed and enthusiastic, reflecting unique tastes and enjoyment of variety.",

    "Medium": "A Balanced Seeker who enjoys trying new products but also values familiarity. You appreciate targeted recommendations, purchase a moderate number of items with solid ratings and a reasonable number of ratings, and your reviews balance detailed feedback with concise, practical comments.",

    "Low": "A Meticulously Selective Buyer who sticks to tried-and-true products, showing little interest in new options. You purchase fewer items, favoring those with high ratings and a large number of ratings. Your own ratings are often very close to or slightly above the average, and your reviews are thoughtful and focused on familiar products."
}

INTERACTION = {
    "High": "A Thorough Conversationalist who enjoys detailed discussions, exploring all aspects of a product or service. You provide extensive reviews and value comprehensive support, engaging in multiple rounds of communication.",

    "Medium": "A Moderate Engager who balances simplicity with detail. You prefer clear communication but can engage in detailed exchanges when necessary. You provide reviews that are a mix of concise observations and some detailed insights, especially if you have strong feelings about a product.",

    "Low": "A Minimalist Interactor who values simplicity and efficiency. You prefer quick, straightforward interactions and leave brief, to-the-point reviews, focusing only on essential product aspects."
}

PRICE = {
    "High": "A Price-Conscious Shopper who is very sensitive to cost and seeks the best deals.",
    "Medium": "A Balanced Buyer who considers price but also values quality and features.",
    "Low": "A Value-Driven Consumer who prioritizes quality and features over price."
}

PROFILE_PROMPT = '''
- Gender: <GENDER>
- Age: <AGE>
- Occupation: <OCCUPATION>
- You are <PRICE_SENSITIVITY>
- You are interested in <SHOPPING_INTEREST>
- You prefer brands like: <BRAND_PREFERENCE>
- You are <DIVERSITY_PREFERENCE>
- You are <INTERACTION_STYLE>
- Your tone and style are <TONE_AND_STYLE>
- You likes to mention: <ITEM_REFERENCE>
- You focus on: <FOCUS>
'''

PRODUCT_PROMPT = '''
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


def pretty_profile(profile: dict) -> str:
    res = PROFILE_PROMPT.replace('<GENDER>', profile['Gender'])
    res = res.replace('<AGE>', profile['Age'])
    res = res.replace('<OCCUPATION>', profile['Occupation'])
    res = res.replace('<PRICE_SENSITIVITY>', PRICE[profile['Price Sensitivity']])
    res = res.replace('<SHOPPING_INTEREST>', profile['Shopping Interest'])
    brands = profile['Brand Preference']
    res = res.replace('<BRAND_PREFERENCE>', brands)
    res = res.replace('<DIVERSITY_PREFERENCE>', DIVERSITY[profile['Diversity Preference']])
    res = res.replace('<INTERACTION_STYLE>', INTERACTION[profile['Interaction Complexity']])
    res = res.replace('<TONE_AND_STYLE>', profile['Tone and Style'])
    res = res.replace('<ITEM_REFERENCE>', profile['Item Reference'])
    res = res.replace('<FOCUS>', profile['Focus Aspect'])
    return res


def pretty_product(product: dict) -> str:
    # print(product)
    res = PRODUCT_PROMPT.replace('<CATEGORY>', str(product['main_category']))
    res = res.replace('<TITLE>', product['title'])
    res = res.replace('<AVERAGE_RATING>', str(product['average_rating']))
    res = res.replace('<RATING_NUMBER>', str(product['rating_number']))
    features = str(product['features'])

    res = res.replace('<FEATURES>', features)
    description = str(product['description'])
    res = res.replace('<DESCRIPTION>', description)
    res = res.replace('<PRICE>', str(product['price']))
    res = res.replace('<STORE>', str(product['store']))
    res = res.replace('<DETAILS>', json.dumps(product['details']))
    res = res.replace('<PARENT_ASIN>', product['parent_asin'])
    return res


class BaseUserSimulationEnv:
    metadata = {}

    def reset(self, instruction: dict, data, task_type) -> str:
        return ""

    def step(self, content: str) -> str:
        return ""

    def get_total_cost(self) -> float:
        return 0


class HumanUserSimulationEnv(BaseUserSimulationEnv):
    def reset(self, instruction: dict, data, task_type) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")


class NoUserSimulationEnv(BaseUserSimulationEnv):
    def reset(self, instruction: dict, data, task_type) -> str:
        obs = f"USER {instruction['user_id']}: {instruction['task']}"
        if instruction["type"] == "review":
            product_info = instruction["target"]["product_info"]
            obs += f"\nHere is the product information: {pretty_product(product_info)}"

        return obs

    def step(self, content: str) -> str:
        return "Error: ###STOP###"


USER_PROMPT = """You are an user interacting with an personalized shopping agent.

Your Profile:
<PROFILE>

You are looking for this product:
<PRODUCT>

The shopping agent will help you complete your shopping requests.

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not hallucinate information that is not provided.
- Do not give additional instructions or ask questions, only respond to the agent's questions.
- Do not provide any specific product details.
- Do not repeat the exact information in your profile or product. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in your profile.
- If the result is not satisfactory, you can express your dissatisfaction and provide clues to help the agent understand your preferences.
"""

USER_PROMPT_REVIEW = """You are an user interacting with an personalized shopping agent.

Your Profile:
<PROFILE>

You have purchased the following product:
<PRODUCT>

Your review is as follows:
<REVIEW>

The shopping agent will help you complete your shopping requests.

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not hallucinate information that is not provided.
- Do not give additional instructions or ask questions, only respond to the agent's questions.
- Do not provide any specific product details.
- Do not repeat the exact information in your profile or product. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in your profile.
- If the result is not satisfactory, you can express your dissatisfaction and provide clues to help the agent understand your preferences.
"""

class OpenAIChatFunc(object):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)

        self.prompt_price_per_million = {
            "gpt-4o-2024-08-06": 2.5,
            "gpt-4o-mini": 0.15,
            "gpt-4o-mini-2024-07-18": 0.15,
            "gpt-4o-2024-05-13": 5,
            "gpt-4o": 5,
            "gpt-4-turbo": 10,
            "gpt-4": 30,
            "gpt-4-32k-0613": 60,
            "gpt-3.5-turbo": 0.5,
        }
        self.completion_price_per_million = {
            "gpt-4o-2024-08-06": 10,
            "gpt-4o-mini": 0.6,
            "gpt-4o-mini-2024-07-18": 0.6,
            "gpt-4o-2024-05-13": 15,
            "gpt-4o": 15,
            "gpt-4-turbo": 30,
            "gpt-4": 60,
            "gpt-4-32k-0613": 120,
            "gpt-3.5-turbo": 1.5,
        }

        if (
            model not in self.prompt_price_per_million
            or model not in self.completion_price_per_million
        ):
            raise ValueError(f"Model {model} is not supported")
        self.model = model

    @ retry(
        wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10)
    )
    def chat_completion_request(
        self, messages: list[dict[str, str]], temperature: float = 1.0, max_tokens: int = 150
    ) -> tuple[str, float]:
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        cost = (
            self.prompt_price_per_million[self.model]
            * response.usage.prompt_tokens
            / 1e6
            + self.completion_price_per_million[self.model]
            * response.usage.completion_tokens
            / 1e6
        )
        return content, cost

    def __call__(self, messages: list[dict[str, str]]) -> tuple[str, float]:
        return self.chat_completion_request(messages, temperature=1.0, max_tokens=150)



def chat_func_factory(
    model: str,
) -> Callable[[list[dict[str, str]]], tuple[str, float]]:
    if model.startswith("gpt-4") or model.startswith("gpt-3.5"):
        return OpenAIChatFunc(model)
    else:
        raise ValueError(f"Unknown model {model}")


class LLMUserSimulationEnv(BaseUserSimulationEnv):
    def __init__(
        self, chat_func: Callable[[list[dict[str, str]]], tuple[str, float]]
    ) -> None:
        super().__init__()
        self.messages = []
        self.system_prompt = USER_PROMPT
        self.chat_func = chat_func
        self.total_cost = 0

    def reset(self, instruction, data, task_type) -> str:
        profile = pretty_profile(data['user_profile'][instruction['user_id']]['user_profile'])
        product = pretty_product(instruction['target']['product_info'])
        review = instruction['target']['review']['text']

        if task_type == "review":
            prompt = USER_PROMPT_REVIEW.replace('<PROFILE>', profile).replace('<PRODUCT>', product).replace('<REVIEW>', review)
        else:
            prompt = USER_PROMPT.replace('<PROFILE>', profile).replace('<PRODUCT>', product)
        
        self.total_cost = 0
        self.messages = [
            {
                "role": "system",
                "content": prompt
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
            {"role": "assistant", "content": instruction['task']},
        ]
        obs = f"USER {instruction['user_id']}: {instruction['task']}"
        if instruction["type"] == "review":
            product_info = instruction["target"]["product_info"]
            obs += f"\nHere is the product information: {pretty_product(product_info)}"
            
        return obs


    def step(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        content, cost = self.chat_func(self.messages)
        self.messages.append({"role": "assistant", "content": content})
        self.total_cost += cost
        return content

    def get_total_cost(self):
        return self.total_cost


def load_user(user_mode: str, model: str = "gpt-4o-mini") -> BaseUserSimulationEnv:
    if user_mode == "human":
        return HumanUserSimulationEnv()
    elif user_mode == "naive":
        chat_func = chat_func_factory(model)
        return LLMUserSimulationEnv(chat_func=chat_func)
    elif user_mode == "no":
        return NoUserSimulationEnv()
    else:
        raise ValueError(f"Unknown user mode {user_mode}")
