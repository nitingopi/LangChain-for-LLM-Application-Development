from abc import ABC, abstractmethod
from IPython.display import Markdown
from IPython.display import display


class LLMBase(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def to_markdown(self, text: str) -> Markdown:
        pass





