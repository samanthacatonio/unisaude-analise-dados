from pathlib import Path
import pandas as pd

PASTA = Path(__file__).resolve().parent.parent
CAMINHO_CACHE = Path(__file__).resolve().parent / "despesas_bruta.pkl"


#Se o .pkl existe, lê dele; senão lê o Excel e cria o .pkl.
def carregar_despesas():
    if CAMINHO_CACHE.exists():
        return pd.read_pickle(CAMINHO_CACHE)
    df_despesas = pd.read_excel(PASTA / "BASE DE DADOS - DESPESAS - OFICIAL.xlsx")
    df_despesas.to_pickle(CAMINHO_CACHE)
    return df_despesas


#Remove as linhas que têm quantidade zero (todas elas têm o valor pago 99999, que é um valor-padrão do sistema, e não um pagamento real)
def remover_quantidade_zero(df_despesas):
    return df_despesas[df_despesas["QUANTIDADE"] > 0]


#Remove as linhas que estão sem beneficiário e converte a coluna para inteiro.
def remover_beneficiario_vazio(df_despesas):
    vazios = df_despesas[df_despesas["BENEFICIARIO"].isna()]
    print(f"Linhas sem beneficiário removidas: {len(vazios)} "
          f"(R$ {vazios['VALOR_PAGO'].sum():,.2f})")

    df_despesas = df_despesas.dropna(subset=["BENEFICIARIO"]).copy()
    df_despesas["BENEFICIARIO"] = df_despesas["BENEFICIARIO"].astype(int)
    return df_despesas


#Aplica os tratamentos na ordem.
def tratar_despesas(df_bruto):
    df_despesas = remover_quantidade_zero(df_bruto)
    df_despesas = remover_beneficiario_vazio(df_despesas)
    return df_despesas


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)   # mostra todas as colunas
    pd.set_option("display.width", 200)          # evita quebrar a tabela em blocos

    despesa = tratar_despesas(carregar_despesas())

    linhas, colunas = despesa.shape
    print(f"A tabela tem {linhas:,} linhas e {colunas} colunas".replace(",", "."))
    print(despesa)

