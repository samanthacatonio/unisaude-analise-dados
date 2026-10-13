import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


PASTA_RESULTADOS = Path(__file__).resolve().parent.parent / "resultados"


#calcula os limites de Tukey (IQR) de uma coluna
def calcular_limites_iqr(serie):
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite": q3 + 1.5 * iqr,
        "limite_extremo": q3 + 3 * iqr,
    }

#separa as linhas com valor acima de um limite
def filtrar_acima_do_limite(df, coluna, limite):
    return df[df[coluna] > limite]

#calcula quanto do total fica com os "percentual" maiores (ex.: 0.01 = o 1% que mais gastou)
def calcular_concentracao(df, coluna, percentual):
    ordenado = df.sort_values(coluna, ascending=False)
    quantidade = int(len(df) * percentual)
    return ordenado.head(quantidade)[coluna].sum() / df[coluna].sum()

#limita os valores acima de um percentil ao valor desse percentil (a pessoa continua na conta)
def winsorizar(serie, percentil):
    teto = serie.quantile(percentil)
    return serie.clip(upper=teto)

#tira da conta os valores acima de um percentil
def remover_acima_do_percentil(serie, percentil):
    teto = serie.quantile(percentil)
    return serie[serie <= teto]

#resume uma coluna em média, mediana e desvio-padrão
def resumir(serie):
    return {
        "media": serie.mean(),
        "mediana": serie.median(),
        "desvio_padrao": serie.std(),
    }

#compara a coluna completa, winsorizada e sem os maiores valores
def comparar_impacto(serie, percentil):
    return pd.DataFrame({
        "completa": resumir(serie),
        f"winsorizada (p{percentil:.0%})": resumir(winsorizar(serie, percentil)),
        f"sem acima do p{percentil:.0%}": resumir(remover_acima_do_percentil(serie, percentil)),
    }).T

#marca cada beneficiário que usou o plano como "outlier" ou "demais"
def marcar_outliers(df_quem_usou, coluna, limite):
    df_marcado = df_quem_usou.copy()
    df_marcado["GRUPO"] = np.where(df_marcado[coluna] > limite, "outlier", "demais")
    return df_marcado


#compara o perfil dos outliers com o dos demais
def comparar_perfil(df_marcado):
    df_perfil = df_marcado.copy()
    df_perfil["E_59_MAIS"] = df_perfil["FAIXA_ETARIA"] == "59+"
    df_perfil["E_MULHER"] = df_perfil["SEXO"] == "F"
    df_perfil["E_APARTAMENTO"] = df_perfil["ACOMODACAO"] == "APARTAMENTO"

    return df_perfil.groupby("GRUPO").agg(
        BENEFICIARIOS=("BENEFICIARIO", "size"),
        IDADE_MEDIA=("IDADE", "mean"),
        MENSALIDADE_MEDIA=("BRUTO", "mean"),
        DESPESA_MENSAL_MEDIA=("DESPESA_MENSAL", "mean"),
        PCT_59_MAIS=("E_59_MAIS", "mean"),
        PCT_MULHERES=("E_MULHER", "mean"),
        PCT_APARTAMENTO=("E_APARTAMENTO", "mean"),
    ).T


#mostra, para cada grupo, como a despesa se divide entre as categorias de uma coluna (% do valor)
def comparar_composicao(df_despesas, df_marcado, coluna):
    despesas_com_grupo = df_despesas.merge(
        df_marcado[["BENEFICIARIO", "GRUPO"]], on="BENEFICIARIO", how="inner"
    )
    tabela = despesas_com_grupo.pivot_table(
        index=coluna, columns="GRUPO", values="VALOR_PAGO", aggfunc="sum", fill_value=0
    )
    percentual = tabela / tabela.sum()
    return percentual.sort_values("outlier", ascending=False)


#boxplot da despesa mensal em escala log, com a linha do limite de outlier
def grafico_boxplot(serie, limite, caminho_arquivo):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(serie, whis=1.5)
    ax.set_yscale("log")
    ax.axhline(
        limite, color="red", linestyle="--",
        label=f"Limite de outlier: R$ {limite:,.0f}/mês".replace(",", "."),
    )
    ax.set_title("Despesa mensal por beneficiário (quem usou o plano)")
    ax.set_ylabel("Despesa mensal (R$, escala logarítmica)")
    ax.set_xticks([1], ["Beneficiários"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho_arquivo, dpi=150)
    plt.close(fig)


#curva de concentração: % dos beneficiários (do que mais gastou ao que menos gastou) x % acumulada da despesa
def grafico_concentracao(df, coluna, caminho_arquivo):
    valores = df[coluna].sort_values(ascending=False).to_numpy()
    pct_despesa_acumulada = valores.cumsum() / valores.sum() * 100
    pct_beneficiarios = np.arange(1, len(valores) + 1) / len(valores) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pct_beneficiarios, pct_despesa_acumulada, color="#1F4E8C", label="Despesa real")
    ax.plot([0, 100], [0, 100], color="gray", linestyle=":", label="Se todos gastassem igual")

    for marco in [1, 5, 10]:
        y = pct_despesa_acumulada[int(len(valores) * marco / 100) - 1]
        ax.scatter(marco, y, color="red", zorder=3)
        ax.annotate(f"{marco}% → {y:.0f}% da despesa", (marco, y),
                    textcoords="offset points", xytext=(10, -12))

    ax.set_title("Concentração da despesa entre os beneficiários")
    ax.set_xlabel("% dos beneficiários (do que mais gastou para o que menos gastou)")
    ax.set_ylabel("% acumulada da despesa")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho_arquivo, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from tratamento_mensalidade import carregar_mensalidade, tratar_mensalidade
    from tratamento_despesas import carregar_despesas, tratar_despesas
    from base_analitica import contar_meses, agregar_despesa_por_beneficiario, montar_base_analitica
    from estatistica import teste_t

    mensalidade = tratar_mensalidade(carregar_mensalidade())
    despesas = tratar_despesas(carregar_despesas())
    base = montar_base_analitica(mensalidade, agregar_despesa_por_beneficiario(despesas), contar_meses(despesas))

    #outliers entre quem usou o plano (com os zeros, mais da metade da base ficaria abaixo do Q1)
    quem_usou = base[base["DESPESA_TOTAL"] > 0]
    limites = calcular_limites_iqr(quem_usou["DESPESA_MENSAL"])
    outliers = filtrar_acima_do_limite(quem_usou, "DESPESA_MENSAL", limites["limite"])
    extremos = filtrar_acima_do_limite(quem_usou, "DESPESA_MENSAL", limites["limite_extremo"])

    print("Outliers: despesa mensal por beneficiário (só quem usou)")
    print(f"Q1 = R$ {limites['q1']:.2f} | Q3 = R$ {limites['q3']:.2f} | IQR = R$ {limites['iqr']:.2f}")
    print(f"Limite (Q3 + 1,5 x IQR): R$ {limites['limite']:.2f}/mês")
    print(f"Limite extremo (Q3 + 3 x IQR): R$ {limites['limite_extremo']:.2f}/mês")
    print(f"Outliers: {len(outliers)} beneficiários ({len(outliers) / len(quem_usou):.2%} de quem usou), "
          f"{outliers['DESPESA_TOTAL'].sum() / base['DESPESA_TOTAL'].sum():.2%} da despesa")
    print(f"Extremos: {len(extremos)} beneficiários")

    print("\nConcentração da despesa (todos os cadastrados)")
    for percentual in [0.01, 0.05, 0.10]:
        print(f"Os {percentual:.0%} que mais gastaram ficam com "
              f"{calcular_concentracao(base, 'DESPESA_TOTAL', percentual):.2%} da despesa")
    print("\nImpacto dos outliers na despesa mensal (todos os cadastrados)")
    print(f"Teto do p99: R$ {base['DESPESA_MENSAL'].quantile(0.99):.2f}/mês")
    print(comparar_impacto(base["DESPESA_MENSAL"], 0.99).round(2))

    #o teste t continua significativo sem a influência dos extremos?
    base_winsorizada = base.assign(DESPESA_MENSAL_WINS=winsorizar(base["DESPESA_MENSAL"], 0.99))
    resultado_t = teste_t(base_winsorizada, "ACOMODACAO", "APARTAMENTO", "ENFERMARIA", "DESPESA_MENSAL_WINS")
    print("\nTeste t com a despesa winsorizada (Apartamento x Enfermaria)")
    print(f"Médias: R$ {resultado_t['media_a']:.2f} x R$ {resultado_t['media_b']:.2f} "
          f"(diferença de R$ {resultado_t['diferenca']:.2f})")
    print(f"t = {resultado_t['t']:.2f}, p-valor = {resultado_t['p_valor']:.2e}")

        #Quem são os outliers
    marcado = marcar_outliers(quem_usou, "DESPESA_MENSAL", limites["limite"])

    print("\nPerfil: outliers x demais (só quem usou)")
    print(comparar_perfil(marcado).round(3))

    print("\nEm que a despesa foi gasta (% do valor de cada grupo)")
    for coluna, linhas in [("ITEM_CONTA", 5), ("NATUREZA_ATENDIMENTO", 2),
                           ("ESPECIALIDADE", 6), ("PRESTADOR", 5)]:
        composicao = comparar_composicao(despesas, marcado, coluna)
        print(f"\n{coluna}")
        print((composicao.head(linhas)[["outlier", "demais"]] * 100).round(1))

    print("\nAs 5 linhas de despesa mais caras")
    print(despesas.nlargest(5, "VALOR_PAGO")[
        ["PRESTADOR", "ITEM_CONTA", "DESCRICAO_PROCEDIMENTO", "QUANTIDADE", "VALOR_PAGO"]
    ])

    #Gráficos
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    grafico_boxplot(quem_usou["DESPESA_MENSAL"], limites["limite"],
                    PASTA_RESULTADOS / "outliers_boxplot.png")
    grafico_concentracao(base, "DESPESA_TOTAL",
                         PASTA_RESULTADOS / "outliers_concentracao.png")
    print(f"\nGráficos salvos em: {PASTA_RESULTADOS}")