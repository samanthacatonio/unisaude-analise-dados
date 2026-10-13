import numpy as np
import pandas as pd
from scipy import stats


#separa os valores de uma coluna em dois grupos (ex.: despesa de APARTAMENTO e de ENFERMARIA)
def separar_dois_grupos(df_base, coluna_grupo, grupo_a, grupo_b, coluna_valor):
    valores_a = df_base.loc[df_base[coluna_grupo] == grupo_a, coluna_valor]
    valores_b = df_base.loc[df_base[coluna_grupo] == grupo_b, coluna_valor]
    return valores_a, valores_b


#mede o tamanho da diferença entre as médias, em "desvios-padrão"
def calcular_d_cohen(valores_a, valores_b):
    n_a, n_b = len(valores_a), len(valores_b)
    desvio_combinado = np.sqrt(
        ((n_a - 1) * valores_a.var() + (n_b - 1) * valores_b.var()) / (n_a + n_b - 2)
    )
    return (valores_a.mean() - valores_b.mean()) / desvio_combinado


#compara a média de uma coluna entre dois grupos usando o teste t de Welch
def teste_t(df_base, coluna_grupo, grupo_a, grupo_b, coluna_valor):
    valores_a, valores_b = separar_dois_grupos(df_base, coluna_grupo, grupo_a, grupo_b, coluna_valor)

    resultado_t = stats.ttest_ind(valores_a, valores_b, equal_var=False)
    intervalo = resultado_t.confidence_interval()

    return {
        "n_a": len(valores_a),
        "n_b": len(valores_b),
        "media_a": valores_a.mean(),
        "media_b": valores_b.mean(),
        "diferenca": valores_a.mean() - valores_b.mean(),
        "ic_95_min": intervalo.low,
        "ic_95_max": intervalo.high,
        "t": resultado_t.statistic,
        "p_valor": resultado_t.pvalue,
        "p_levene": stats.levene(valores_a, valores_b).pvalue,
        "d_cohen": calcular_d_cohen(valores_a, valores_b),
        "p_mann_whitney": stats.mannwhitneyu(valores_a, valores_b).pvalue,
    }

#mantém só os grupos que têm pelo menos "minimo" registros
def manter_grupos_com_minimo(df, coluna_grupo, minimo):
    contagem = df[coluna_grupo].value_counts()
    grupos_validos = contagem[contagem >= minimo].index
    return df[df[coluna_grupo].isin(grupos_validos)]


#separa os valores de uma coluna em uma lista, um item por grupo
def separar_varios_grupos(df, coluna_grupo, coluna_valor):
    return [grupo[coluna_valor].values for _, grupo in df.groupby(coluna_grupo)]


#mede quanto da variação total é explicada pela diferença entre os grupos (0 a 1)
def calcular_eta2(grupos):
    todos = np.concatenate(grupos)
    media_geral = todos.mean()
    variacao_entre_grupos = sum(len(g) * (g.mean() - media_geral) ** 2 for g in grupos)
    variacao_total = ((todos - media_geral) ** 2).sum()
    return variacao_entre_grupos / variacao_total


#compara a média de uma coluna entre vários grupos (ANOVA e confirmações)
def anova(df, coluna_grupo, coluna_valor):
    grupos = separar_varios_grupos(df, coluna_grupo, coluna_valor)

    resultado_anova = stats.f_oneway(*grupos)
    return {
        "grupos": len(grupos),
        "n": sum(len(g) for g in grupos),
        "F": resultado_anova.statistic,
        "p_valor": resultado_anova.pvalue,
        "p_alexander_govern": stats.alexandergovern(*grupos).pvalue,
        "p_kruskal": stats.kruskal(*grupos).pvalue,
        "eta2": calcular_eta2(grupos),
    }


#depois da ANOVA: compara os grupos dois a dois e diz quais pares são diferentes
def comparar_pares_tukey(df, coluna_grupo, coluna_valor):
    nomes = sorted(df[coluna_grupo].unique())
    grupos = [df.loc[df[coluna_grupo] == nome, coluna_valor].values for nome in nomes]
    resultado = stats.tukey_hsd(*grupos)

    pares = []
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            pares.append({
                "grupo_a": nomes[i],
                "grupo_b": nomes[j],
                "diferenca": grupos[i].mean() - grupos[j].mean(),
                "p_valor": resultado.pvalue[i, j],
                "diferente": resultado.pvalue[i, j] < 0.05,
            })
    return pd.DataFrame(pares)

#mede se duas colunas andam juntas: Pearson (valores) e Spearman (posições)
def correlacao(df, coluna_x, coluna_y):
    pearson = stats.pearsonr(df[coluna_x], df[coluna_y])
    spearman = stats.spearmanr(df[coluna_x], df[coluna_y])
    return {
        "n": len(df),
        "pearson": pearson.statistic,
        "p_pearson": pearson.pvalue,
        "spearman": spearman.statistic,
        "p_spearman": spearman.pvalue,
    }

if __name__ == "__main__":
    from tratamento_mensalidade import carregar_mensalidade, tratar_mensalidade
    from tratamento_despesas import carregar_despesas, tratar_despesas
    from base_analitica import (
        contar_meses,
        agregar_despesa_por_beneficiario,
        montar_base_analitica,
        agregar_atendimentos,
        resumir_por_plano_e_faixa,
    )

    mensalidade = tratar_mensalidade(carregar_mensalidade())
    despesas = tratar_despesas(carregar_despesas())
    base = montar_base_analitica(mensalidade, agregar_despesa_por_beneficiario(despesas), contar_meses(despesas))

    #Teste t: Apartamento x Enfermaria
    resultado_t = teste_t(base, "ACOMODACAO", "APARTAMENTO", "ENFERMARIA", "DESPESA_MENSAL")
    print("Teste t: despesa mensal de Apartamento x Enfermaria")
    print(f"Apartamento: {resultado_t['n_a']} beneficiários, média R$ {resultado_t['media_a']:.2f}")
    print(f"Enfermaria:  {resultado_t['n_b']} beneficiários, média R$ {resultado_t['media_b']:.2f}")
    print(f"Diferença: R$ {resultado_t['diferenca']:.2f} "
          f"(IC 95%: R$ {resultado_t['ic_95_min']:.2f} a R$ {resultado_t['ic_95_max']:.2f})")
    print(f"t = {resultado_t['t']:.2f}, p-valor = {resultado_t['p_valor']:.2e}")
    print(f"Levene (variâncias iguais?): p = {resultado_t['p_levene']:.2e}")
    print(f"d de Cohen = {resultado_t['d_cohen']:.3f}")
    print(f"Mann-Whitney (confirmação sem supor normalidade): p = {resultado_t['p_mann_whitney']:.2e}")

    #ANOVA: faixa etária e plano (base analítica, um beneficiário por linha)
    for coluna in ["FAIXA_ETARIA", "ID_PLANO"]:
        resultado_anova = anova(base, coluna, "DESPESA_MENSAL")
        print(f"\nANOVA: despesa mensal por {coluna}")
        print(f"{resultado_anova['grupos']} grupos, {resultado_anova['n']} beneficiários")
        print(f"F = {resultado_anova['F']:.2f}, p-valor = {resultado_anova['p_valor']:.2e}")
        print(f"Alexander-Govern: p = {resultado_anova['p_alexander_govern']:.2e} | "
              f"Kruskal-Wallis: p = {resultado_anova['p_kruskal']:.2e}")
        print(f"eta² = {resultado_anova['eta2']:.4f}")

    tukey = comparar_pares_tukey(base, "FAIXA_ETARIA", "DESPESA_MENSAL")
    print(f"\nTukey (faixa etária): {tukey['diferente'].sum()} de {len(tukey)} pares são diferentes")

    #ANOVA: especialidade (um atendimento por linha, só especialidades com 30+ atendimentos)
    atendimentos = manter_grupos_com_minimo(agregar_atendimentos(despesas), "ESPECIALIDADE", 30)
    resultado_anova = anova(atendimentos, "ESPECIALIDADE", "VALOR_PAGO")
    print("\nANOVA: valor do atendimento por especialidade")
    print(f"{resultado_anova['grupos']} especialidades, {resultado_anova['n']} atendimentos")
    print(f"F = {resultado_anova['F']:.2f}, p-valor = {resultado_anova['p_valor']:.2e}")
    print(f"Alexander-Govern: p = {resultado_anova['p_alexander_govern']:.2e} | "
          f"Kruskal-Wallis: p = {resultado_anova['p_kruskal']:.2e}")
    print(f"eta² = {resultado_anova['eta2']:.4f}")

        #Correlação: mensalidade x despesa
    quem_usou = base[base["DESPESA_TOTAL"] > 0]
    resumo_grupos = resumir_por_plano_e_faixa(base, minimo_beneficiarios=20)

    comparacoes = [
        ("Individual (todos os beneficiários)", base, "BRUTO", "DESPESA_MENSAL"),
        ("Individual (só quem usou)", quem_usou, "BRUTO", "DESPESA_MENSAL"),
        ("Por grupo plano x faixa (médias)", resumo_grupos, "MENSALIDADE_MEDIA", "DESPESA_MEDIA"),
    ]
    print("\nCorrelação: mensalidade x despesa mensal")
    for nome, df, coluna_x, coluna_y in comparacoes:
        resultado_cor = correlacao(df, coluna_x, coluna_y)
        print(f"{nome}: n = {resultado_cor['n']}, "
              f"Pearson = {resultado_cor['pearson']:.3f} (p = {resultado_cor['p_pearson']:.2e}), "
              f"Spearman = {resultado_cor['spearman']:.3f} (p = {resultado_cor['p_spearman']:.2e})")