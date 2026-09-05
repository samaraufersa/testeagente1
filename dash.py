
import json
import re
import requests
import pandas as pd
import plotly.express as px
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

print("CUDA disponível:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

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

print("DataFrame carregado com sucesso!")
print("Total de unidades federativas:", len(df))

display(df)


def contar_estados():
    """
    Conta os estados brasileiros.
    O Distrito Federal não é considerado um estado.
    """
    
    dados = df[df["sigla"] != "DF"]

    return {
        "quantidade": len(dados)
    }


def regiao_com_mais_estados():
    """
    Descobre qual região possui mais estados.
    """

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
    """
    Descobre qual região possui menos estados.
    """

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


def estados_por_regiao(regiao):
    """
    Lista os estados de uma determinada região.
    """

    dados = df[
        (df["regiao"].str.lower() == regiao.lower()) &
        (df["sigla"] != "DF")
    ]

    return {
        "regiao": regiao,
        "quantidade": len(dados),
        "estados": dados["estado"].tolist()
    }


def estado_por_sigla(sigla):
    """
    Retorna informações de um estado usando sua sigla.
    """

    resultado = df[
        df["sigla"].str.upper() == sigla.upper()
    ]

    if resultado.empty:
        return {
            "erro": f"Nenhum estado encontrado para a sigla {sigla}."
        }

    linha = resultado.iloc[0]

    return {
        "estado": linha["estado"],
        "sigla": linha["sigla"],
        "regiao": linha["regiao"]
    }


def listar_regioes():
    """
    Lista as regiões brasileiras.
    """

    regioes = sorted(
        df["regiao"].unique().tolist()
    )

    return {
        "regioes": regioes
    }


print("Tools criadas com sucesso!")


FUNCOES = {
    "contar_estados": contar_estados,
    "regiao_com_mais_estados": regiao_com_mais_estados,
    "regiao_com_menos_estados": regiao_com_menos_estados,
    "estados_por_regiao": estados_por_regiao,
    "estado_por_sigla": estado_por_sigla,
    "listar_regioes": listar_regioes
}

TOOLS = {
    "contar_estados": {
        "descricao": "Conta quantos estados existem no Brasil. Não considera o Distrito Federal.",
        "argumentos": {}
    },

    "regiao_com_mais_estados": {
        "descricao": "Descobre qual região brasileira possui mais estados.",
        "argumentos": {}
    },

    "regiao_com_menos_estados": {
        "descricao": "Descobre qual região brasileira possui menos estados.",
        "argumentos": {}
    },

    "estados_por_regiao": {
        "descricao": "Lista os estados pertencentes a uma região brasileira.",
        "argumentos": {
            "regiao": "Nome da região. Exemplo: Nordeste"
        }
    },

    "estado_por_sigla": {
        "descricao": "Retorna o estado e sua região a partir de uma sigla.",
        "argumentos": {
            "sigla": "Sigla do estado. Exemplo: PB"
        }
    },

    "listar_regioes": {
        "descricao": "Lista as regiões brasileiras.",
        "argumentos": {}
    }
}

print("Catálogo de Tools criado!")


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("Carregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Carregando modelo...")

if torch.cuda.is_available():

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

print("Modelo carregado com sucesso!")


def perguntar_llm(mensagens, max_new_tokens=300):

    texto = tokenizer.apply_chat_template(
        mensagens,
        tokenize=False,
        add_generation_prompt=True
    )

    entradas = tokenizer(
        texto,
        return_tensors="pt"
    )

    if torch.cuda.is_available():

        entradas = {
            chave: valor.to(model.device)
            for chave, valor in entradas.items()
        }

    with torch.no_grad():

        saida = model.generate(
            **entradas,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    novos_tokens = saida[
        0,
        entradas["input_ids"].shape[1]:
    ]

    resposta = tokenizer.decode(
        novos_tokens,
        skip_special_tokens=True
    )

    return resposta.strip()


mensagens = [
    {
        "role": "system",
        "content": "Você é um assistente que responde em português do Brasil."
    },
    {
        "role": "user",
        "content": "Explique de forma simples o que é uma LLM."
    }
]

resposta = perguntar_llm(mensagens)

print(resposta)


PROMPT_AGENTE = """
Você é um agente de análise de dados do IBGE.

Você possui ferramentas Python que consultam um DataFrame
com informações sobre os estados brasileiros.

Quando a pergunta depender dos dados do DataFrame,
você deve usar uma ferramenta.

Não invente números.

Você deve responder EXATAMENTE em um dos formatos abaixo.

Para usar uma ferramenta:

{
    "acao": "usar_tool",
    "tool": "NOME_DA_TOOL",
    "argumentos": {}
}

Para responder diretamente:

{
    "acao": "responder",
    "resposta": "Sua resposta"
}

Ferramentas disponíveis:

"""

for nome, info in TOOLS.items():

    PROMPT_AGENTE += f"""

TOOL: {nome}

Descrição:
{info["descricao"]}

Argumentos:
{json.dumps(info["argumentos"], ensure_ascii=False)}

"""

print(PROMPT_AGENTE)

def executar_agente(pergunta):

    mensagens = [

        {
            "role": "system",
            "content": PROMPT_AGENTE
        },

        {
            "role": "user",
            "content": pergunta
        }

    ]

    # Primeira chamada da LLM
    resposta_llm = perguntar_llm(
        mensagens,
        max_new_tokens=300
    )

    print("\n--- DECISÃO DA LLM ---")
    print(resposta_llm)

    # Tentar interpretar a resposta como JSON
    try:

        decisao = json.loads(resposta_llm)

    except:

        trecho = re.search(
            r'\{.*\}',
            resposta_llm,
            re.DOTALL
        )

        if trecho:

            try:

                decisao = json.loads(
                    trecho.group()
                )

            except:

                return resposta_llm

        else:

            return resposta_llm

    # Caso a LLM responda diretamente
    if decisao.get("acao") == "responder":

        return decisao.get(
            "resposta",
            "Não consegui gerar uma resposta."
        )

    # Caso a LLM queira usar uma Tool
    if decisao.get("acao") == "usar_tool":

        nome_tool = decisao.get("tool")

        argumentos = decisao.get(
            "argumentos",
            {}
        )

        # Verifica se a Tool existe
        if nome_tool not in FUNCOES:

            return (
                f"A ferramenta '{nome_tool}' "
                "não existe."
            )

        # Recupera a função Python
        funcao = FUNCOES[nome_tool]

        # Executa a função
        try:

            resultado = funcao(
                **argumentos
            )

        except Exception as erro:

            return f"Erro na ferramenta: {erro}"

        print("\n--- TOOL UTILIZADA ---")
        print(nome_tool)

        print("\n--- RESULTADO DA TOOL ---")
        print(resultado)

        # Envia o resultado novamente para a LLM
        mensagens.append(
            {
                "role": "assistant",
                "content": resposta_llm
            }
        )

        mensagens.append(
            {
                "role": "user",
                "content": f"""
A ferramenta {nome_tool} foi executada.

Resultado:

{json.dumps(
    resultado,
    ensure_ascii=False
)}

Agora responda à pergunta original
utilizando esse resultado.

Não invente informações.

Responda somente em linguagem natural,
sem JSON.
"""
            }
        )

        resposta_final = perguntar_llm(
            mensagens,
            max_new_tokens=300
        )

        return resposta_final

    return resposta_llm

pergunta = "Quantos estados existem no Brasil?"

resposta = executar_agente(
    pergunta
)

print("\n==============================")
print("RESPOSTA FINAL")
print("==============================")
print(resposta)

perguntas = [
    "Quantos estados existem no Brasil?",
    "Qual região possui mais estados?",
    "Qual região possui menos estados?",
    "Quais são os estados do Nordeste?",
    "Quantos estados existem no Nordeste?",
    "Qual estado corresponde à sigla PB?",
    "A Paraíba pertence a qual região?",
    "Quais são as regiões brasileiras?"
]

for pergunta in perguntas:

    print("\n" + "=" * 70)
    print("PERGUNTA:", pergunta)
    print("=" * 70)

    resposta = executar_agente(
        pergunta
    )

    print("\nRESPOSTA:")
    print(resposta)

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

fig1 = px.bar(
    df_regioes,
    x="regiao",
    y="quantidade_estados",
    title="Quantidade de estados por região",
    labels={
        "regiao": "Região",
        "quantidade_estados": "Quantidade de estados"
    }
)

fig1.show()


fig2 = px.pie(
    df_regioes,
    names="regiao",
    values="quantidade_estados",
    title="Distribuição dos estados por região"
)

fig2.show()


regiao_selecionada = "Nordeste"

df_filtrado = df_estados[
    df_estados["regiao"] == regiao_selecionada
]

fig3 = px.bar(
    df_filtrado,
    x="estado",
    title=f"Estados da região {regiao_selecionada}",
    labels={
        "estado": "Estado"
    }
)

fig3.show()
