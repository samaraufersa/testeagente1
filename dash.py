# ============================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json

from openai import OpenAI


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Dashboard IBGE com Agente de IA",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CONFIGURAÇÃO DA OPENAI
# ============================================================

# Para testar no Google Colab, você pode usar getpass.
# Para um dashboard publicado, recomenda-se utilizar st.secrets.

import getpass

# Solicita a chave apenas se ela ainda não estiver armazenada
# na sessão do Streamlit.
if "api_key" not in st.session_state:

    api_key = getpass.getpass(
        "Digite sua chave da API da OpenAI: "
    )

    st.session_state.api_key = api_key


# Cria o cliente da OpenAI
client = OpenAI(
    api_key=st.session_state.api_key
)


# ============================================================
# FUNÇÃO 1
# BUSCAR DADOS NA API DO IBGE
# ============================================================

@st.cache_data
def buscar_dados_ibge():

    # Endpoint da API do IBGE
    url = (
        "https://servicodados.ibge.gov.br/"
        "api/v1/localidades/estados"
    )

    # Faz a requisição
    resposta = requests.get(url)

    # Verifica se ocorreu algum erro
    resposta.raise_for_status()

    # Converte o JSON em objeto Python
    dados = resposta.json()

    return dados


# ============================================================
# FUNÇÃO 2
# TRANSFORMAR OS DADOS EM DATAFRAME
# ============================================================

@st.cache_data
def criar_dataframe(dados):

    # Normaliza o JSON
    df = pd.json_normalize(dados)

    # Seleciona as colunas necessárias
    df = df[
        [
            "id",
            "sigla",
            "nome",
            "regiao.nome"
        ]
    ]

    # Renomeia as colunas
    df.columns = [
        "codigo",
        "sigla",
        "estado",
        "regiao"
    ]

    # Ordena os estados
    df = df.sort_values(
        by=["regiao", "estado"]
    )

    # Reinicia os índices
    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# FUNÇÕES QUE O AGENTE PODERÁ UTILIZAR
# ============================================================


# ------------------------------------------------------------
# FUNÇÃO: CONTAR ESTADOS
# ------------------------------------------------------------

def contar_estados(df):

    total = len(df)

    return {
        "quantidade_estados": total
    }


# ------------------------------------------------------------
# FUNÇÃO: REGIÃO COM MAIS ESTADOS
# ------------------------------------------------------------

def regiao_com_mais_estados(df):

    # Conta quantos estados existem em cada região
    contagem = (
        df
        .groupby("regiao")
        .size()
        .reset_index(
            name="quantidade_estados"
        )
    )

    # Identifica a maior quantidade
    maior = contagem.loc[
        contagem[
            "quantidade_estados"
        ].idxmax()
    ]

    return {
        "regiao": maior["regiao"],
        "quantidade_estados":
            int(maior["quantidade_estados"])
    }


# ------------------------------------------------------------
# FUNÇÃO: REGIÃO COM MENOS ESTADOS
# ------------------------------------------------------------

def regiao_com_menos_estados(df):

    # Conta os estados de cada região
    contagem = (
        df
        .groupby("regiao")
        .size()
        .reset_index(
            name="quantidade_estados"
        )
    )

    # Identifica a menor quantidade
    menor = contagem.loc[
        contagem[
            "quantidade_estados"
        ].idxmin()
    ]

    return {
        "regiao": menor["regiao"],
        "quantidade_estados":
            int(menor["quantidade_estados"])
    }


# ------------------------------------------------------------
# FUNÇÃO: LISTAR ESTADOS DE UMA REGIÃO
# ------------------------------------------------------------

def estados_por_regiao(df, regiao):

    # Filtra a região
    resultado = df[
        df["regiao"]
        .str.lower()
        == regiao.lower()
    ]

    # Obtém os estados
    estados = (
        resultado["estado"]
        .sort_values()
        .tolist()
    )

    return {
        "regiao": regiao,
        "estados": estados,
        "quantidade": len(estados)
    }


# ============================================================
# FERRAMENTAS DISPONÍVEIS PARA A LLM
# ============================================================

tools = [

    # --------------------------------------------------------
    # FERRAMENTA 1
    # --------------------------------------------------------

    {
        "type": "function",

        "name": "contar_estados",

        "description": (
            "Use esta função para descobrir "
            "a quantidade total de estados "
            "presentes no DataFrame."
        ),

        "parameters": {
            "type": "object",

            "properties": {}
        }
    },


    # --------------------------------------------------------
    # FERRAMENTA 2
    # --------------------------------------------------------

    {
        "type": "function",

        "name": "regiao_com_mais_estados",

        "description": (
            "Use esta função quando o usuário "
            "perguntar qual região brasileira "
            "possui a maior quantidade de estados."
        ),

        "parameters": {
            "type": "object",

            "properties": {}
        }
    },


    # --------------------------------------------------------
    # FERRAMENTA 3
    # --------------------------------------------------------

    {
        "type": "function",

        "name": "regiao_com_menos_estados",

        "description": (
            "Use esta função quando o usuário "
            "perguntar qual região brasileira "
            "possui a menor quantidade de estados."
        ),

        "parameters": {
            "type": "object",

            "properties": {}
        }
    },


    # --------------------------------------------------------
    # FERRAMENTA 4
    # --------------------------------------------------------

    {
        "type": "function",

        "name": "estados_por_regiao",

        "description": (
            "Use esta função para listar os "
            "estados que pertencem a uma "
            "determinada região brasileira."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "regiao": {

                    "type": "string",

                    "description": (
                        "Nome da região brasileira. "
                        "Exemplos: Norte, Nordeste, "
                        "Centro-Oeste, Sudeste ou Sul."
                    )
                }
            },

            "required": [
                "regiao"
            ]
        }
    }
]


# ============================================================
# FUNÇÃO RESPONSÁVEL POR EXECUTAR AS FERRAMENTAS
# ============================================================

def executar_ferramenta(
    nome_funcao,
    argumentos,
    df
):

    # --------------------------------------------------------
    # CONTAR ESTADOS
    # --------------------------------------------------------

    if nome_funcao == "contar_estados":

        resultado = contar_estados(
            df
        )


    # --------------------------------------------------------
    # REGIÃO COM MAIS ESTADOS
    # --------------------------------------------------------

    elif nome_funcao == (
        "regiao_com_mais_estados"
    ):

        resultado = (
            regiao_com_mais_estados(
                df
            )
        )


    # --------------------------------------------------------
    # REGIÃO COM MENOS ESTADOS
    # --------------------------------------------------------

    elif nome_funcao == (
        "regiao_com_menos_estados"
    ):

        resultado = (
            regiao_com_menos_estados(
                df
            )
        )


    # --------------------------------------------------------
    # ESTADOS POR REGIÃO
    # --------------------------------------------------------

    elif nome_funcao == (
        "estados_por_regiao"
    ):

        resultado = (
            estados_por_regiao(

                df,

                argumentos["regiao"]

            )
        )


    # --------------------------------------------------------
    # CASO A FUNÇÃO NÃO EXISTA
    # --------------------------------------------------------

    else:

        resultado = {
            "erro":
                "Ferramenta não encontrada."
        }


    return resultado


# ============================================================
# AGENTE COM LLM
# ============================================================

def agente_llm(pergunta, df):


    # --------------------------------------------------------
    # ETAPA 1
    # A LLM RECEBE A PERGUNTA E DECIDE QUAL FERRAMENTA USAR
    # --------------------------------------------------------

    resposta = client.responses.create(

        # Escolha um modelo disponível na sua conta.
        model="gpt-5",

        instructions="""
        Você é um agente especializado em
        analisar dados sobre os estados brasileiros.

        Você deve responder às perguntas do usuário
        utilizando as ferramentas disponíveis.

        Sempre que precisar de uma informação que esteja
        no DataFrame, utilize uma ferramenta.

        Não invente dados.

        Responda sempre em português.
        """,

        input=pergunta,

        tools=tools,

        tool_choice="auto"
    )


    # --------------------------------------------------------
    # PROCURA UMA CHAMADA DE FUNÇÃO
    # --------------------------------------------------------

    for item in resposta.output:


        # Verifica se a LLM solicitou
        # a execução de uma função
        if item.type == "function_call":


            # Nome da função escolhida pela LLM
            nome_funcao = item.name


            # Argumentos enviados pela LLM
            argumentos = json.loads(
                item.arguments
            )


            # ------------------------------------------------
            # EXECUTA A FUNÇÃO NO DATAFRAME
            # ------------------------------------------------

            resultado = executar_ferramenta(

                nome_funcao,

                argumentos,

                df

            )


            # ------------------------------------------------
            # ETAPA 2
            # ENVIA O RESULTADO PARA A LLM
            # ------------------------------------------------

            resposta_final = (
                client.responses.create(

                    model="gpt-5",

                    instructions="""
                    Você é um assistente especializado
                    em análise de dados.

                    Responda em português.

                    Utilize exclusivamente os resultados
                    fornecidos pela análise.

                    Não invente informações.

                    Responda de maneira clara e objetiva.
                    """,

                    input=f"""
                    Pergunta do usuário:

                    {pergunta}

                    Resultado da análise:

                    {json.dumps(
                        resultado,
                        ensure_ascii=False
                    )}

                    Responda à pergunta do usuário.
                    """
                )
            )


            # Retorna a resposta final
            return resposta_final.output_text


    # --------------------------------------------------------
    # CASO A LLM NÃO TENHA UTILIZADO UMA FERRAMENTA
    # --------------------------------------------------------

    return resposta.output_text


# ============================================================
# INÍCIO DO DASHBOARD
# ============================================================

st.title(
    "🇧🇷 Dashboard de Estados Brasileiros"
)

st.write(
    "Dashboard criado com dados da API do IBGE, "
    "Pandas, Plotly, Streamlit e OpenAI."
)


# ============================================================
# BUSCAR OS DADOS
# ============================================================

with st.spinner(
    "Buscando dados da API do IBGE..."
):

    dados = buscar_dados_ibge()


# ============================================================
# CRIAR O DATAFRAME
# ============================================================

df = criar_dataframe(
    dados
)


# ============================================================
# MOSTRAR O DATAFRAME
# ============================================================

st.subheader(
    "Dados dos Estados Brasileiros"
)

st.dataframe(
    df,
    use_container_width=True
)


# ============================================================
# PREPARAR DADOS PARA OS GRÁFICOS
# ============================================================

df_regioes = (

    df

    .groupby(
        "regiao"
    )

    .size()

    .reset_index(
        name="quantidade_estados"
    )
)


# ============================================================
# GRÁFICO 1
# QUANTIDADE DE ESTADOS POR REGIÃO
# ============================================================

st.subheader(
    "Gráfico 1 — Quantidade de Estados por Região"
)


fig1 = px.bar(

    df_regioes,

    x="regiao",

    y="quantidade_estados",

    title=(
        "Quantidade de Estados "
        "por Região"
    ),

    labels={
        "regiao": "Região",

        "quantidade_estados":
            "Quantidade de Estados"
    }
)


st.plotly_chart(

    fig1,

    use_container_width=True
)


# ============================================================
# GRÁFICO 2
# DISTRIBUIÇÃO DOS ESTADOS
# ============================================================

st.subheader(
    "Gráfico 2 — Distribuição dos Estados por Região"
)


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


# ============================================================
# GRÁFICO 3
# ESTADOS DA REGIÃO SELECIONADA
# ============================================================

st.subheader(
    "Gráfico 3 — Estados por Região"
)


# Obtém as regiões
regioes = sorted(
    df["regiao"].unique()
)


# Permite ao usuário escolher
regiao_selecionada = st.selectbox(

    "Escolha uma região:",

    regioes
)


# Filtra os dados
df_filtrado = df[

    df["regiao"]
    == regiao_selecionada

]


# Cria o gráfico
fig3 = px.bar(

    df_filtrado,

    x="estado",

    title=(
        f"Estados da Região "
        f"{regiao_selecionada}"
    ),

    labels={
        "estado": "Estado"
    }
)


# Mostra o gráfico
st.plotly_chart(

    fig3,

    use_container_width=True
)


# ============================================================
# AGENTE DE IA
# ============================================================

st.divider()


st.header(
    "🤖 Agente de Análise de Dados"
)


st.write(
    "Faça perguntas sobre os dados "
    "apresentados no dashboard."
)


# Campo de pergunta
pergunta = st.chat_input(

    "Exemplo: Qual região possui mais estados?"
)


# ============================================================
# EXECUTAR O AGENTE
# ============================================================

if pergunta:


    # Mostra a pergunta
    with st.chat_message(
        "user"
    ):

        st.write(
            pergunta
        )


    # Executa o agente
    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "O agente está analisando os dados..."
        ):

            try:

                resposta_agente = (
                    agente_llm(
                        pergunta,
                        df
                    )
                )


                st.write(
                    resposta_agente
                )


            except Exception as erro:

                st.error(
                    f"Ocorreu um erro: {erro}"
                )
