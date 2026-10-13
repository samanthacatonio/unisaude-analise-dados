from pathlib import Path
import numpy as np
import pandas as pd

# pasta do desafio = uma pasta acima de "codigo"
PASTA = Path(__file__).resolve().parent.parent

#pegar a base de dados - Mensalidade
def carregar_mensalidade():
    return pd.read_excel(PASTA / "BASE DE DADOS - MENSALIDADE - OFICIAL.xlsx")

#renomear a coluna Idade
def renomear_coluna_idade(df_mensalidade):
    return df_mensalidade.rename(columns={"IDADE+2": "IDADE"})

#remover as duplicatas em beneficiário, mantendo o menor valor de mensalidade
def remover_beneficiarios_duplicados(df_mensalidade):
    df_mensalidade = df_mensalidade.sort_values("BRUTO").drop_duplicates(
        subset="BENEFICIARIO",
        keep="first"
    )
    return df_mensalidade.reset_index(drop=True)

#criar coluna de faixa etária, seguindo a tabela com os cortes da ANS.
def criar_faixa_etaria(df_mensalidade):
    df_mensalidade = df_mensalidade.copy()

    condicoes = [
        df_mensalidade["IDADE"].between(0, 18),
        df_mensalidade["IDADE"].between(19, 23),
        df_mensalidade["IDADE"].between(24, 28),
        df_mensalidade["IDADE"].between(29, 33),
        df_mensalidade["IDADE"].between(34, 38),
        df_mensalidade["IDADE"].between(39, 43),
        df_mensalidade["IDADE"].between(44, 48),
        df_mensalidade["IDADE"].between(49, 53),
        df_mensalidade["IDADE"].between(54, 58),
        df_mensalidade["IDADE"] >= 59
    ]

    faixas = [
        "00 a 18", "19 a 23", "24 a 28", "29 a 33", "34 a 38", "39 a 43", "44 a 48", "49 a 53", "54 a 58", "59+"
    ]

    df_mensalidade["FAIXA_ETARIA"] = np.select(
        condicoes,
        faixas,
        default="Não informado"
    )
    return df_mensalidade

#aplica os tratamentos na ordem: a faixa etária depende da coluna IDADE já renomeada
def tratar_mensalidade(df_bruto):
    df_mensalidade = renomear_coluna_idade(df_bruto)
    df_mensalidade = remover_beneficiarios_duplicados(df_mensalidade)
    df_mensalidade = criar_faixa_etaria(df_mensalidade)

    assert df_mensalidade["BENEFICIARIO"].is_unique, "Há beneficiário duplicado na mensalidade"
    return df_mensalidade


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)   # mostra todas as colunas
    pd.set_option("display.width", 200)          # evita quebrar a tabela em blocos

    mensalidade = tratar_mensalidade(carregar_mensalidade())

    linhas, colunas = mensalidade.shape
    print(f"A tabela tem {linhas:,} linhas e {colunas} colunas".replace(",", "."))
    print(mensalidade)