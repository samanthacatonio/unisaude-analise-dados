import pandas as pd

#calcula a quantidade de meses que tem na tabela
def contar_meses(df_despesas):
    return df_despesas["COMPETENCIA"].nunique()

#soma todos os valores pagos por beneficiário
def agregar_despesa_por_beneficiario(df_despesas):
    df_agregado = df_despesas.groupby("BENEFICIARIO", as_index=False)["VALOR_PAGO"].sum()
    return df_agregado.rename(columns={"VALOR_PAGO": "DESPESA_TOTAL"})


#Junta a mensalidade com a despesa total de cada beneficiário. Quem está na mensalidade e não teve despesa fica com despesa 0.
def montar_base_analitica(df_mensalidade, df_despesa_por_beneficiario, meses):
    df_base = df_mensalidade.merge(
        df_despesa_por_beneficiario,
        on="BENEFICIARIO",
        how="left",
        validate="one_to_one"
    )

    df_base["DESPESA_TOTAL"] = df_base["DESPESA_TOTAL"].fillna(0)
    df_base["DESPESA_MENSAL"] = df_base["DESPESA_TOTAL"] / meses

    return df_base

#separa as despesas lançadas que não têm um beneficiário cadastrado.
def despesas_sem_beneficiario_cadastrado(df_despesas, df_mensalidade):
    esta_na_mensalidade = df_despesas["BENEFICIARIO"].isin(df_mensalidade["BENEFICIARIO"])
    return df_despesas[esta_na_mensalidade == False]

#soma o valor de cada atendimento: uma conta em uma especialidade
def agregar_atendimentos(df_despesas):
    df_com_especialidade = df_despesas.dropna(subset=["ESPECIALIDADE"])
    return df_com_especialidade.groupby(
        ["CONTA_PRINCIPAL", "ESPECIALIDADE"], as_index=False
    )["VALOR_PAGO"].sum()

#resume a base por plano e faixa etária: quantos beneficiários, mensalidade média e despesa média
def resumir_por_plano_e_faixa(df_base, minimo_beneficiarios):
    df_resumo = df_base.groupby(["ID_PLANO", "FAIXA_ETARIA"], as_index=False).agg(
        BENEFICIARIOS=("BENEFICIARIO", "size"),
        MENSALIDADE_MEDIA=("BRUTO", "mean"),
        DESPESA_MEDIA=("DESPESA_MENSAL", "mean"),
    )
    return df_resumo[df_resumo["BENEFICIARIOS"] >= minimo_beneficiarios]

#uma linha por atendimento (conta x especialidade), com o valor e as características do atendimento e do beneficiário.
#só entram beneficiários cadastrados, porque o modelo precisa de idade, sexo e plano
def montar_base_atendimentos(df_despesas, df_mensalidade):
    df_atendimentos = df_despesas.dropna(subset=["ESPECIALIDADE"]).groupby(
        ["CONTA_PRINCIPAL", "ESPECIALIDADE"], as_index=False
    ).agg(
        VALOR=("VALOR_PAGO", "sum"),
        BENEFICIARIO=("BENEFICIARIO", "first"),
        REGIME_ATENDIMENTO=("REGIME_ATENDIMENTO", "first"),
        NATUREZA_ATENDIMENTO=("NATUREZA_ATENDIMENTO", "first"),
        CLINICA=("CLINICA", "first"),
        TIPO_DE_REDE=("TIPO_DE_REDE", "first"),
    )
    return df_atendimentos.merge(
        df_mensalidade[["BENEFICIARIO", "IDADE", "SEXO", "ID_PLANO"]],
        on="BENEFICIARIO",
        how="inner",
    )

if __name__ == "__main__":
    from tratamento_mensalidade import carregar_mensalidade, tratar_mensalidade
    from tratamento_despesas import carregar_despesas, tratar_despesas

    pd.set_option("display.max_columns", None)   # mostra todas as colunas
    pd.set_option("display.width", 200)          # evita quebrar a tabela em blocos

    mensalidade = tratar_mensalidade(carregar_mensalidade())
    despesas = tratar_despesas(carregar_despesas())

    meses = contar_meses(despesas)
    base = montar_base_analitica(mensalidade, agregar_despesa_por_beneficiario(despesas), meses)

    linhas, colunas = base.shape
    print(f"A tabela tem {linhas:,} linhas e {colunas} colunas".replace(",", "."))
    print(base)