from utils.api.openaiapi import OpenAIAPI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


api_key = OpenAIAPI()

chat = ChatOpenAI(
    model="gpt-4o-mini",
    max_completion_tokens='1000',
    temperature=0.5 
)

messages = [
    SystemMessage(content="Você é um assistente especializado em segurança cibernética, com foco em análises do OpenVAS. "
                          "Sua função é ajudar a interpretar relatórios, identificar vulnerabilidades e recomendar ações de mitigação."),
    HumanMessage(content="Quais são as vulnerabilidades críticas mais comuns encontradas nos relatórios do OpenVAS?"),
    AIMessage(content="As vulnerabilidades críticas mais comuns incluem falta de patches em sistemas, autenticação fraca, "
                      "exposição de serviços desnecessários, e uso de softwares desatualizados. Você pode mitigá-las "
                      "mantendo os sistemas atualizados, implementando autenticação forte e desativando serviços desnecessários."),
    HumanMessage(content="Como posso priorizar as ações de mitigação?"),
]

res = chat.invoke(messages)

print(res.content)
