# from strategies.base_strategy import LLMStrategy
from llm_util.base_llm import LLMBase
# from openai import OpenAI

# from config.config import OPENAI_API_KEY
import textwrap
import boto3
from botocore.exceptions import ClientError

from IPython.display import display
from IPython.display import Markdown

class Bedrock_llm(LLMBase):
    def __init__(self):
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")

    def get_response(self, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]
        # Set the model ID, e.g., Titan Text Premier.
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        try:
            # Send the message to the model, using a basic inference configuration.
            response = self.client.converse(
                modelId=model_id,
                messages=conversation,
                inferenceConfig={"maxTokens":2048,"stopSequences":["\n\nHuman:"],"temperature":0.5,"topP":1},
                additionalModelRequestFields={"top_k":250}
            )

            # Extract and print the response text.
            response_text = response["output"]["message"]["content"][0]["text"]
            # print(response_text)
            return response_text
            # return "hi"

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)
    
    def to_markdown(self, text: str) -> Markdown:
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

       