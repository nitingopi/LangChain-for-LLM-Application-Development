from llm_util.base_llm import LLMBase
import google.generativeai as genai
from config.config import GEMINI_API_KEY
import textwrap


from IPython.display import display
from IPython.display import Markdown

class Gemini_llm(LLMBase):
    def __init__(self) -> None:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API Key: ", GEMINI_API_KEY)


    def get_response(self, prompt: str) -> str:
        # Implement the Gemini API call here
        model = genai.GenerativeModel("gemini-1.5-flash")
        config = {
        "temperature": 0,
        "top_p": 1
        }
        output = model.generate_content(prompt, generation_config=config )
        return output.text
    
    def to_markdown(self, text: str) -> Markdown:
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

