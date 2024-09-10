# from strategies.base_strategy import LLMStrategy
from config.config import CLAUDE_API_KEY
from llm_util.base_llm import LLMBase
import anthropic
from botocore.exceptions import ClientError
import textwrap

from IPython.display import display
from IPython.display import Markdown


class Claude_llm(LLMBase):
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        print("Claude API Key: ", CLAUDE_API_KEY)

    def get_response(self, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        # Set the model ID, e.g., Titan Text Premier.
        # model_id = "claude-3-opus-20240229"
        model_id = "claude-3-5-sonnet-20240620"
        try:
            # Send the message to the model, using a basic inference configuration.
            response = self.client.messages.create(
                model=model_id, max_tokens=1000, temperature=0, messages=conversation
            )
            return response.content
            # return "hi"

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)

    def to_markdown(self, text: str) -> Markdown:
        text = text.replace("•", "  *")
        return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))
