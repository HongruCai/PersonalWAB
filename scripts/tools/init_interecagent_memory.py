import os
import sys
import json
import random
import numpy as np
from time import time
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

PRODUCT_PROMPT = '''
Product:
- Title: <TITLE>
- Parent ASIN: <PARENT_ASIN>
- Categories: <CATEGORIES>
Review:
- Your Rating: <RATING>
- Your Review Text: <REVIEW_TEXT>
'''

USER_PROFILE_PROMPT = '''Your task is to extract a user profile from the user history.
The profile consists of two parts: like and dislike. Each part is a list. 
Distinguish between like and dislike by the rating and review text. Not all products need to be included in the profile.
Format the profile in JSON with two keys: "like" and "dislike".
Each item in the list should contain only the "parent_asin" to represent the product.
No other information or explanations are needed.

Extract user profiles from the following history:
History:
{History}
'''

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
def generate_user_profile_GPT(user_history, model, system_prompt, temperature):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}],
        max_tokens=5000,
        temperature=temperature,
    )
    profile = response.choices[0].message.content
    start_index = profile.find('{')
    return profile[start_index:]

def format_product_review(product: dict) -> str:
    """Format a product review based on PRODUCT_PROMPT template."""
    review_text = PRODUCT_PROMPT.replace('<TITLE>', product['title'])
    review_text = review_text.replace('<CATEGORIES>', str(product['categories']))
    review_text = review_text.replace('<PARENT_ASIN>', product['parent_asin'])
    return review_text

def get_user_history(file_path, user_id):
    """Extract the specified user's history reviews from JSON file."""
    with open(file_path, "r") as file:
        all_reviews = json.load(file)
    user_reviews = all_reviews.get(user_id, [])

    history_reviews = []
    for review in user_reviews:
        if review['split'] == 'history':
            formatted_review = format_product_review(review['product_info'])
            formatted_review = formatted_review.replace('<RATING>', str(review['review']['rating']))
            formatted_review = formatted_review.replace('<REVIEW_TEXT>', review['review']['text'])
            history_reviews.append(formatted_review)
    return history_reviews[-200:]

if __name__ == "__main__":
    output_data = {}
    with open("PUMA/data/user_history.json", "r") as file:
        users_data = json.load(file)

    sample_user_ids = list(users_data.keys())

    for user_id in tqdm(sample_user_ids):
        user_history = get_user_history("PUMA/data/user_history.json", user_id)
        system_prompt = USER_PROFILE_PROMPT.replace("{History}", "".join(user_history))
        
        temperature = 0.0
        profile_json = generate_user_profile_GPT(user_history, "gpt-4o-mini", system_prompt, temperature)
        
        profile_cleaned = profile_json.replace("```json", "").replace("```", "").replace("\n", "")
        profile_data = json.loads(profile_cleaned)
        profile_data['expect'] = [] 

        output_data[user_id] = profile_data

        with open("data/interecagent_memory.json", "w") as output_file:
            json.dump(output_data, output_file, indent=2)
