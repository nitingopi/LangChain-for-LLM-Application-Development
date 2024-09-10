import os
from dotenv import load_dotenv

load_dotenv()

# Load API keys and other configuration settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
# JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN')
# JIRA_BASE_URL = os.getenv('JIRA_BASE_URL')