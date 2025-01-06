from utils.api.openaiapi import OpenAIAPI
from src.embeddings.modelEmbeddings import ModelEmbeddings

# Inicialize a classe OpenAIAPI e pegue a chave da API
openai_api = OpenAIAPI()
api_key = openai_api.api_key  # Pegue a chave da API que foi carregada

# Inicialize o modelo de embeddings com a chave da API
embedding_model = ModelEmbeddings(openai_api_key=api_key)

# Textos para gerar embeddings
texts = ["OpenVAS é uma ferramenta de segurança.", "A LangChain é útil para RAG."]

# Gerar embeddings
embeddings = embedding_model.generate_embeddings(texts)

print("Embeddings gerados:", embeddings)

embedding_text = "\n".join([str(embedding) for embedding in embeddings])

# Exibir no console
print("Embeddings como texto:")
print(embedding_text)

# Ou salvar em um arquivo
with open("embeddings.txt", "w") as f:
    f.write(embedding_text)