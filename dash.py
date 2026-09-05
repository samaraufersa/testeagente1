import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from ollama import chat


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard IBGE + LLM",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TÍTULO
# ============================================================

st.title("🤖 Dashboard IBGE + LLM")
st.subheader("Análise dos estados brasileiros com Inteligência Artificial")


# ============================================================
# 1. BUSCAR DADOS DO IBGE
# ============================================================

@st.cache_data
def carregar_dados():

    url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"

    resposta = requests.get(url, timeout=30)

    resposta.raise_for_status()

    dados = resposta.json()

    df = pd.json_normalize(dados)

    df = df[
        [
            "id",
            "sigla",
            "nome",
            "regiao.nome"
        ]
    ]

    df.columns = [
        "codigo",
        "sigla",
        "estado",
        "regiao"
    ]

    df = df.sort_values(
        by=["regiao", "estado"]
    ).reset_index(drop=True)

    return df


df = carregar_dados()


# ============================================================
# 2. FUNÇÕES QUE SERÃO USADAS COMO TOOLS PELA LLM
# ============================================================

def contar_estados():

    # O IBGE retorna 27 unidades federativas:
    # 26 estados + Distrito Federal.
    #
    # Como a pergunta é sobre ESTADOS,
    # retiramos o Distrito Federal.

    quantidade = len(
        df[df["sigla"] != "DF"]
    )

    return {
        "quantidade_estados": quantidade
    }


def regiao_com_mais_estados():

    dados = df[df["sigla"] != "DF"]

    resultado = (
        dados
        .groupby("regiao")
        .size()
        .sort_values(ascending=False)
    )

    return {
        "regiao": resultado.index[0],
        "quantidade": int(resultado.iloc[0])
    }


def regiao_com_menos_estados():

    dados = df[df["sigla"] != "DF"]

    resultado = (
        dados
        .groupby("regiao")
        .size()
        .sort_values()
    )

    return {
        "regiao": resultado.index[0],
        "quantidade": int(resultado.iloc[0])
    }


def estados_por_regiao(regiao: str):

    dados = df[
        (df["regiao"].str.lower() == regiao.lower()) &
        (df["sigla"] != "DF")
    ]

    return {
        "regiao": regiao,
        "estados": dados["estado"].tolist(),
        "quantidade": len(dados)
    }


def estados_por_sigla(sigla: str):

    resultado = df[
        df["sigla"].str.upper() == sigla.upper()
    ]

    if resultado.empty:

        return {
            "erro": f"Nenhum estado encontrado com a sigla {sigla}"
        }

    linha = resultado.iloc[0]

    return {
        "estado": linha["estado"],
        "sigla": linha["sigla"],
        "regiao": linha["regiao"]
    }


def listar_regioes():

    regioes = (
        df["regiao"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return {
        "regioes": regioes
    }


# ============================================================
# 3. DESCRIÇÃO DAS TOOLS PARA A LLM
# ============================================================

tools = [

    {
        "type": "function",
        "function": {
            "name": "contar_estados",
            "description": (
                "Retorna a quantidade de estados do Brasil. "
                "Não considera o Distrito Federal como estado."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "regiao_com_mais_estados",
            "description": (
                "Retorna a região brasileira que possui "
                "a maior quantidade de estados."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "regiao_com_menos_estados",
            "description": (
                "Retorna a região brasileira que possui "
                "a menor quantidade de estados."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "estados_por_regiao",
            "description": (
                "Retorna os estados pertencentes a uma determinada "
                "região brasileira."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "regiao": {
                        "type": "string",
                        "description": (
                            "Nome da região brasileira. "
                            "Exemplo: Nordeste."
                        )
                    }
                },
                "required": ["regiao"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "estados_por_sigla",
            "description": (
                "Retorna o nome e a região de um estado "
                "a partir da sua sigla."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sigla": {
                        "type": "string",
                        "description": (
                            "Sigla do estado. "
                            "Exemplo: PB."
                        )
                    }
                },
                "required": ["sigla"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "listar_regioes",
            "description": (
                "Retorna a lista das regiões brasileiras."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


# ============================================================
# 4. MAPEAR NOME DA TOOL PARA FUNÇÃO PYTHON
# ============================================================

funcoes = {

    "contar_estados":
        contar_estados,

    "regiao_com_mais_estados":
        regiao_com_mais_estados,

    "regiao_com_menos_estados":
        regiao_com_menos_estados,

    "estados_por_regiao":
        estados_por_regiao,

    "estados_por_sigla":
        estados_por_sigla,

    "listar_regioes":
        listar_regioes
}


# ============================================================
# 5. FUNÇÃO DO AGENTE
# ============================================================

def executar_agente(pergunta):

    mensagens = [

        {
            "role": "system",
            "content": """
Você é um assistente especializado nos dados do IBGE
disponíveis neste aplicativo.

Você deve responder perguntas sobre estados e regiões
brasileiras.

IMPORTANTE:

Quando a pergunta depender dos dados do DataFrame,
use uma das ferramentas disponíveis.

Não invente números.

Se a pergunta puder ser respondida usando uma ferramenta,
utilize a ferramenta.

Depois de receber o resultado da ferramenta,
responda ao usuário de forma clara e didática.
"""
        },

        {
            "role": "user",
            "content": pergunta
        }

    ]


    # ========================================================
    # LOOP DO AGENTE
    # ========================================================

    while True:

        resposta = chat(

            model="qwen3",

            messages=mensagens,

            tools=tools
        )


        # ====================================================
        # SE A LLM NÃO QUISER USAR TOOL
        # ====================================================

        if not resposta.message.tool_calls:

            return resposta.message.content


        # ====================================================
        # ADICIONA A RESPOSTA DA LLM AO HISTÓRICO
        # ====================================================

        mensagens.append(
            resposta.message
        )


        # ====================================================
        # EXECUTA AS TOOLS
        # ====================================================

        for chamada in resposta.message.tool_calls:

            nome_tool = chamada.function.name

            argumentos = chamada.function.arguments


            # -----------------------------------------------
            # Verifica se a ferramenta existe
            # -----------------------------------------------

            if nome_tool not in funcoes:

                resultado = {
                    "erro":
                    f"Ferramenta {nome_tool} não encontrada."
                }

            else:

                try:

                    funcao = funcoes[nome_tool]

                    resultado = funcao(
                        **argumentos
                    )

                except Exception as erro:

                    resultado = {
                        "erro": str(erro)
                    }


            # -----------------------------------------------
            # Envia o resultado para a LLM
            # -----------------------------------------------

            mensagens.append(
                {
                    "role": "tool",
                    "tool_name": nome_tool,
                    "content": str(resultado)
                }
            )


# ============================================================
# 6. DASHBOARD
# ============================================================

st.divider()

st.header("📊 Dados do IBGE")

col1, col2, col3 = st.columns(3)


# Quantidade de estados

quantidade_estados = contar_estados()["quantidade_estados"]

col1.metric(
    "Estados brasileiros",
    quantidade_estados
)


# Região com mais estados

mais = regiao_com_mais_estados()

col2.metric(
    "Região com mais estados",
    mais["regiao"],
    f'{mais["quantidade"]} estados'
)


# Região com menos estados

menos = regiao_com_menos_estados()

col3.metric(
    "Região com menos estados",
    menos["regiao"],
    f'{menos["quantidade"]} estados'
)


# ============================================================
# 7. MOSTRAR DATAFRAME
# ============================================================

st.subheader("Tabela de estados")

st.dataframe(
    df,
    use_container_width=True
)


# ============================================================
# 8. GRÁFICO 1
# ============================================================

st.subheader("Quantidade de estados por região")

df_estados = df[
    df["sigla"] != "DF"
]

df_regioes = (
    df_estados
    .groupby("regiao")
    .size()
    .reset_index(
        name="quantidade_estados"
    )
)

grafico1 = px.bar(
    df_regioes,
    x="regiao",
    y="quantidade_estados",
    title="Estados por região"
)

st.plotly_chart(
    grafico1,
    use_container_width=True
)


# ============================================================
# 9. GRÁFICO 2
# ============================================================

st.subheader("Distribuição dos estados por região")

grafico2 = px.pie(
    df_regioes,
    names="regiao",
    values="quantidade_estados",
    title="Distribuição dos estados brasileiros"
)

st.plotly_chart(
    grafico2,
    use_container_width=True
)


# ============================================================
# 10. GRÁFICO 3
# ============================================================

st.subheader("Estados de uma região")

regiao_selecionada = st.selectbox(
    "Escolha uma região:",
    sorted(
        df_estados["regiao"].unique()
    )
)

df_filtrado = df_estados[
    df_estados["regiao"] == regiao_selecionada
]

grafico3 = px.bar(
    df_filtrado,
    x="estado",
    y=[1] * len(df_filtrado),
    title=f"Estados da região {regiao_selecionada}"
)

grafico3.update_layout(
    yaxis_title="Quantidade"
)

st.plotly_chart(
    grafico3,
    use_container_width=True
)


# ============================================================
# 11. CHAT COM A LLM
# ============================================================

st.divider()

st.header("🤖 Pergunte sobre os dados")


st.write(
    "Faça perguntas sobre os estados e regiões brasileiras."
)


# Histórico da conversa

if "mensagens_chat" not in st.session_state:

    st.session_state.mensagens_chat = []


# Mostrar histórico

for mensagem in st.session_state.mensagens_chat:

    with st.chat_message(
        mensagem["role"]
    ):

        st.markdown(
            mensagem["content"]
        )


# Entrada do usuário

pergunta = st.chat_input(
    "Ex.: Qual região possui mais estados?"
)


if pergunta:

    # -----------------------------------------------
    # Mostra pergunta
    # -----------------------------------------------

    st.session_state.mensagens_chat.append(
        {
            "role": "user",
            "content": pergunta
        }
    )


    with st.chat_message("user"):

        st.markdown(pergunta)


    # -----------------------------------------------
    # Executa agente
    # -----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "A LLM está analisando os dados..."
        ):

            try:

                resposta = executar_agente(
                    pergunta
                )

            except Exception as erro:

                resposta = (
                    "Ocorreu um erro ao consultar a LLM.\n\n"
                    f"Detalhes: {erro}"
                )


        st.markdown(
            resposta
        )


    # -----------------------------------------------
    # Salva resposta
    # -----------------------------------------------

    st.session_state.mensagens_chat.append(
        {
            "role": "assistant",
            "content": resposta
        }
    )
