from fastapi import FastAPI, Header, HTTPException
from ollama import Client

# ============================================================
# CONFIGURAÇÃO
# ============================================================

app = FastAPI()

# Ollama está rodando na mesma máquina do servidor
ollama_client = Client(
    host="http://localhost:11434"
)

# Chave para proteger nossa API
# NÃO coloque uma chave real diretamente no código em produção.
API_KEY = "minha-chave-secreta"


# ============================================================
# ENDPOINT PARA CONVERSAR COM O OLLAMA
# ============================================================

@app.post("/chat")
def chat(
    dados: dict,
    authorization: str = Header(None)
):

    # Verifica se a requisição possui a chave correta
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=401,
            detail="Não autorizado"
        )

    # Recebe as mensagens enviadas pelo Streamlit
    mensagens = dados["messages"]

    # Envia as mensagens para o Ollama
    resposta = ollama_client.chat(
        model="gemma4",
        messages=mensagens
    )

    # Retorna a resposta para o Streamlit
    return {
        "resposta": resposta.message.content
    }
