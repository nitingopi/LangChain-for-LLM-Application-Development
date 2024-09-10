from llm_util.base_llm import LLMBase
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


from config.config import GEMINI_API_KEY
import textwrap


from IPython.display import display
from IPython.display import Markdown

class Gemini_llm(LLMBase):
    def __init__(self) -> None:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API Key: ", GEMINI_API_KEY)


    def get_llm(self) -> genai.GenerativeModel:
        return genai.GenerativeModel("gemini-1.5-flash")    
    
    def get_langchain_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=GEMINI_API_KEY
        )


    def get_response(self, prompt: str) -> str:
        # Implement the Gemini API call here
        model = self.get_llm()
        config = {
        "temperature": 0,
        "top_p": 1
        }
        output = model.generate_content(prompt, generation_config=config )
        return output.text
    
    def to_markdown(self, text: str) -> Markdown:
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
    

    def initialize_conversation(self, memory) -> ConversationChain:
        llm = self.get_langchain_llm()
        conversation = ConversationChain(
            llm=llm, 
            memory = memory,
            verbose=True
        )
        return conversation
    
    
  

