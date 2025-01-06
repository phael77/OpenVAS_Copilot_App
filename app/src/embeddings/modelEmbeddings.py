import os
from typing import List, Any
from langchain_openai import OpenAIEmbeddings
from utils.api.openaiapi import OpenAIAPI


class ModelEmbeddings:
    def __init__(self, openai_api_key: str=None, model: str="text-embedding-3-small" ):
        