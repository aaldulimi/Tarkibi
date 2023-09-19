import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

def generate_search_query(person: str) -> str:
    """
    Generate a search query for a person
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Think step by step. Im building an audio dataset for one person. This person is Joe Rogan. I'm using youtube as the source for my audio clips (before pre-processing...). What should the youtube search query be? Return only the search query.  Dont add any other input. Just on string, which is the search query. Thank you."},
            {"role": "user", "content": "Joe Rogan"},
            {"role": "assistant", "content": "Joe Rogan podcast full episodes"},
            {"role": "user", "content": "Cristiano Ronaldo"},
            {"role": "assistant", "content": "Cristiano Ronaldo interview"},
            {"role": "user", "content": person},
        ]
    )

    return response['choices'][0]['message']['content']
