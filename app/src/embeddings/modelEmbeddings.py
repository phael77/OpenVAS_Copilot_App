import os
from typing import List, Any
from langchain_openai import OpenAIEmbeddings


# class ModelEmbeddings:
#     def __init__(self, openai_api_key: str, model: str="text-embedding-3-small"):
#         self.openai_api_key = openai_api_key  # Corrigido aqui
#         if not self.openai_api_key:
#             raise ValueError("OpenAI API key is required")

#         self.model_name = model
#         self.embeddings = self.load_model()

#     def load_model(self):
#         """
#         Inicializa o objeto OpenAIEmbeddings.
#         """
#         try:
#             print(f"Inicializando OpenAIEmbeddings com o modelo: {self.model_name}")
#             embeddings = OpenAIEmbeddings(model=self.model_name, openai_api_key=self.openai_api_key)
#             print("Modelo de embeddings carregado com sucesso!")
#             return embeddings
#         except Exception as e:
#             print(f"Erro ao carregar o modelo OpenAIEmbeddings: {e}")
#             raise

#     def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """
#         Gera embeddings para uma lista de textos.

#         Args:
#             texts (List[str]): Lista de textos para gerar embeddings.

#         Returns:
#             List[List[float]]: Lista de embeddings gerados.
#         """
#         if not self.embeddings:
#             raise ValueError("Modelo de embeddings não carregado. Verifique a inicialização.")

#         print(f"Gerando embeddings para {len(texts)} textos...")
#         embeddings = self.embeddings.embed_documents(texts)
#         return embeddings

#     def get_model_info(self) -> str:
#         """
#         Retorna informações sobre o modelo carregado.

#         Returns:
#             str: Nome do modelo de embeddings utilizado.
#         """
#         return f"Modelo de embeddings: {self.model_name}"

#     def get_openai_api_key(self) -> str:
#         """
#         Retorna a chave da API OpenAI utilizada.

#         Returns:
#             str: Chave da API OpenAI.
#         """
#         return self.openai_api_key

    
