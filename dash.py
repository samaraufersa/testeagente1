import requests
import pandas as pd
import plotly.express as px
import streamlit as st
import ollama


# ============================================================
# 1. BUSCAR OS DADOS DO IBGE
# ============================================================

url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"

resposta = requests.get(url)
resposta.raise_for_status()

dados = resposta.json()


# ============================================================
# 2. TRANSFORMAR OS DADOS EM DATAFRAME
# ============================================================

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


# ============================================================
# 3. FUNÇÕES QUE SERÃO UTILIZADAS COMO TOOLS
# ============================================================

def contar_estados(df):
    """
    Retorna a quantidade total de estados.
    """
    return len(df)


def regiao_com_mais_estados(df):
    """
    Retorna a região que possui mais estados.
    """

    resultado = (
        df.groupby("regiao")
        .size()
        .sort_values(ascending=False)
    )

    return {
        "regiao": resultado.index[0],
        "quantidade": int(resultado.iloc[0])
    }


def regiao_com_menos_estados(df):
    """
    Retorna a região que possui menos estados.
    """

    resultado = (
        df.groupby("regiao")
        .size()
        .sort_values()
    )

    return {
        "regiao": resultado.index[0],
        "quantidade": int(resultado.iloc[0])
    }


def estados_por_regiao(df, regiao):
    """
    Retorna os estados pertencentes a uma região.
    """

    resultado = df[
        df["regiao"].str.lower() == regiao.lower()
    ]

    return resultado["estado"].tolist()


# ============================================================
# 4. DEFINIR AS TOOLS DO OLLAMA
# ============================================================

tools = [
    contar_estados,
    regiao_com_mais_estados,
    regiao_com_menos_estados,
    estados_por_regiao
]


# ============================================================
# 5. FUNÇÃO DO AGENTE
# ============================================================

def executar_agente(pergunta, df):

    mensagens = [
        {
            "role": "system",
            "content": """
            Você é um assistente especializado em analisar
            dados dos estados brasileiros.

            Utilize as ferramentas disponíveis sempre que
            a pergunta depender dos dados do DataFrame.

            Não invente informações.

            Responda em português de forma clara e objetiva.
            """
        },
        {
            "role": "user",
            "content": pergunta
        }
    ]

    # Primeira chamada ao modelo
    resposta = ollama.chat(
        model="gemma4",
        messages=mensagens,
        tools=tools
    )

    # Verifica se o modelo solicitou alguma ferramenta
    if resposta.message.tool_calls:

        # Adiciona a resposta do modelo ao histórico
        mensagens.append(resposta.message)

        # Percorre as ferramentas solicitadas
        for chamada in resposta.message.tool_calls:

            nome_funcao = chamada.function.name
            argumentos = chamada.function.arguments

            # -----------------------------------------------
            # Executa a ferramenta solicitada
            # -----------------------------------------------

            if nome_funcao == "contar_estados":

                resultado = contar_estados(df)

            elif nome_funcao == "regiao_com_mais_estados":

                resultado = regiao_com_mais_estados(df)

            elif nome_funcao == "regiao_com_menos_estados":

                resultado = regiao_com_menos_estados(df)

            elif nome_funcao == "estados_por_regiao":

                regiao = argumentos["regiao"]

                resultado = estados_por_regiao(
                    df,
                    regiao
                )

            else:

                resultado = "Ferramenta não encontrada."

            # -----------------------------------------------
            # Envia o resultado da ferramenta para o Ollama
            # -----------------------------------------------

            mensagens.append(
                {
                    "role": "tool",
                    "content": str(resultado),
                    "tool_name": nome_funcao
                }
            )

        # Segunda chamada ao modelo
        # Agora o modelo possui o resultado da função
        resposta_final = ollama.chat(
            model="gemma4",
            messages=mensagens
        )

        return resposta_final.message.content

    # Caso nenhuma ferramenta tenha sido utilizada
    return resposta.message.content


# ============================================================
# 6. DASHBOARD STREAMLIT
# ============================================================

st.title("🇧🇷 Dashboard dos Estados Brasileiros")


# ============================================================
# 7. MOSTRAR DATAFRAME
# ============================================================

st.subheader("Dados dos estados")

st.dataframe(df)


# ============================================================
# 8. GRÁFICO 1 — QUANTIDADE DE ESTADOS POR REGIÃO
# ============================================================

df_regioes = (
    df.groupby("regiao")
    .size()
    .reset_index(name="quantidade_estados")
)

grafico1 = px.bar(
    df_regioes,
    x="regiao",
    y="quantidade_estados",
    title="Quantidade de estados por região"
)

st.plotly_chart(
    grafico1,
    use_container_width=True
)


# ============================================================
# 9. GRÁFICO 2 — DISTRIBUIÇÃO DOS ESTADOS
# ============================================================

grafico2 = px.pie(
    df_regioes,
    names="regiao",
    values="quantidade_estados",
    title="Distribuição dos estados por região"
)

st.plotly_chart(
    grafico2,
    use_container_width=True
)


# ============================================================
# 10. GRÁFICO 3 — ESTADOS DE UMA REGIÃO
# ============================================================

regiao_selecionada = st.selectbox(
    "Escolha uma região:",
    sorted(df["regiao"].unique())
)

df_selecionado = df[
    df["regiao"] == regiao_selecionada
]

grafico3 = px.bar(
    df_selecionado,
    x="estado",
    y="codigo",
    title=f"Estados da região {regiao_selecionada}"
)

st.plotly_chart(
    grafico3,
    use_container_width=True
)


# ============================================================
# 11. CHAT COM O AGENTE
# ============================================================

st.subheader("🤖 Assistente de dados")

pergunta = st.chat_input(
    "Pergunte algo sobre os estados..."
)

if pergunta:

    # Mostra a pergunta do usuário
    with st.chat_message("user"):
        st.write(pergunta)

    # Executa o agente
    resposta = executar_agente(
        pergunta,
        df
    )

    # Mostra a resposta do agente
    with st.chat_message("assistant"):
        st.write(resposta)
