from dotenv import load_dotenv
import os

load_dotenv()

class OpenAIAPI:
    def __init__(self, api_key:str=None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        else:
            print("API key loaded successfully.")