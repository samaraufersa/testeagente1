import streamlit as st
import requests
import pandas as pd
import plotly.express as px


# ========================================
# FUNÇÃO PARA BUSCAR DADOS DO IBGE
# ========================================

def buscar_dados_ibge():

    url = (
        "https://servicodados.ibge.gov.br/"
        "api/v1/localidades/estados"
    )

    resposta = requests.get(url)

    dados = resposta.json()

    return dados


# ========================================
# FUNÇÕES DO AGENTE
# ========================================

def contar_estados(df):

    return len(df)


def regiao_com_mais_estados(df):

    resultado = (
        df
        .groupby("regiao")
        .size()
        .idxmax()
    )

    return resultado


def estados_por_regiao(df, regiao):

    resultado = df[
        df["regiao"]
        .str.lower()
        == regiao.lower()
    ]

    return resultado["estado"].tolist()


# ========================================
# AGENTE
# ========================================

def agente(pergunta, df):

    pergunta = pergunta.lower()


    # QUANTIDADE DE ESTADOS
    if "quantos estados" in pergunta:

        total = contar_estados(df)

        return (
            f"O Brasil possui {total} "
            f"unidades federativas."
        )


    # REGIÃO COM MAIS ESTADOS
    elif (
        "região" in pergunta
        and "mais" in pergunta
        and "estados" in pergunta
    ):

        regiao = regiao_com_mais_estados(df)

        return (
            f"A região com mais estados é "
            f"{regiao}."
        )


    # REGIÃO NORDESTE
    elif "nordeste" in pergunta:

        estados = estados_por_regiao(
            df,
            "Nordeste"
        )

        return (
            "Os estados do Nordeste são: "
            + ", ".join(estados)
        )


    else:

        return (
            "Ainda não sei responder essa pergunta. "
            "Tente perguntar sobre estados "
            "ou regiões."
        )

# ========================================
# DASHBOARD
# ========================================

st.title(
    "Dashboard de Estados Brasileiros"
)

st.write(
    "Dados obtidos pela API do IBGE"
)


# ========================================
# BUSCAR DADOS
# ========================================

dados = buscar_dados_ibge()


# ========================================
# DATAFRAME
# ========================================

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


# ========================================
# TABELA
# ========================================

st.subheader(
    "Dados dos Estados"
)

st.dataframe(df)


# ========================================
# GRÁFICO 1
# ========================================

df_regioes = (
    df
    .groupby("regiao")
    .size()
    .reset_index(
        name="quantidade_estados"
    )
)

fig1 = px.bar(
    df_regioes,
    x="regiao",
    y="quantidade_estados",
    title=(
        "Quantidade de Estados "
        "por Região"
    )
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# ========================================
# GRÁFICO 2
# ========================================

fig2 = px.pie(
    df_regioes,
    names="regiao",
    values="quantidade_estados",
    title=(
        "Distribuição dos Estados "
        "por Região"
    )
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# ========================================
# GRÁFICO 3
# ========================================

regiao_selecionada = st.selectbox(
    "Escolha uma região",
    df["regiao"].unique()
)

df_filtrado = df[
    df["regiao"]
    == regiao_selecionada
]

fig3 = px.bar(
    df_filtrado,
    x="estado",
    title=(
        f"Estados da Região "
        f"{regiao_selecionada}"
    )
)

st.plotly_chart(
    fig3,
    use_container_width=True
)


# ========================================
# AGENTE
# ========================================

st.divider()

st.subheader(
    "Agente de Análise de Dados"
)

pergunta = st.chat_input(
    "Pergunte algo sobre os dados"
)


if pergunta:

    resposta = agente(
        pergunta,
        df
    )

    st.write(resposta)
