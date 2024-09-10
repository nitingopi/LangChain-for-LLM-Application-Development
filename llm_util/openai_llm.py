# from strategies.base_strategy import LLMStrategy
from llm_util.base_llm import LLMBase
from openai import OpenAI

from config.config import OPENAI_API_KEY
import textwrap


from IPython.display import display
from IPython.display import Markdown

class OpenAI_llm(LLMBase):
    def __init__(self):
        self.client = OpenAI()

    def get_response(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content
    
    def to_markdown(self, text: str) -> Markdown:
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

       