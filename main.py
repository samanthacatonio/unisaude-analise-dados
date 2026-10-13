import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from tratamento_mensalidade import carregar_mensalidade, tratar_mensalidade
from tratamento_despesas import carregar_despesas, tratar_despesas
from base_analitica import (contar_meses, agregar_despesa_por_beneficiario, montar_base_analitica, despesas_sem_beneficiario_cadastrado,
    agregar_atendimentos, resumir_por_plano_e_faixa, montar_base_atendimentos,)
from estatistica import teste_t, anova, comparar_pares_tukey, manter_grupos_com_minimo, correlacao
from outliers import (calcular_limites_iqr, filtrar_acima_do_limite, calcular_concentracao, winsorizar, comparar_impacto,
    marcar_outliers, comparar_perfil, comparar_composicao, grafico_boxplot, grafico_concentracao, PASTA_RESULTADOS,)
from modelo import (preparar_variaveis, dividir_treino_teste, treinar_e_avaliar, avaliar_em_log, coeficientes_com_p_valor,)


#Item 2: teste t, ANOVA (com Tukey) e correlação
def mostrar_testes_estatisticos(base, despesas):
    print("\n========== ITEM 2: ANÁLISES ESTATÍSTICAS ==========")

    #Teste t: Apartamento x Enfermaria
    resultado_t = teste_t(base, "ACOMODACAO", "APARTAMENTO", "ENFERMARIA", "DESPESA_MENSAL")
    print("\nTeste t: despesa mensal de Apartamento x Enfermaria")
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


#Item 3: outliers (detecção, impacto, perfil e gráficos)
def mostrar_outliers(base, despesas):
    print("\n========== ITEM 3: OUTLIERS ==========")

    #Parte 1: detecção pelo IQR, entre quem usou o plano
    quem_usou = base[base["DESPESA_TOTAL"] > 0]
    limites = calcular_limites_iqr(quem_usou["DESPESA_MENSAL"])
    outliers = filtrar_acima_do_limite(quem_usou, "DESPESA_MENSAL", limites["limite"])
    extremos = filtrar_acima_do_limite(quem_usou, "DESPESA_MENSAL", limites["limite_extremo"])

    print("\nOutliers: despesa mensal por beneficiário (só quem usou)")
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

    #Parte 2: impacto dos outliers nas estatísticas e no teste t
    print("\nImpacto dos outliers na despesa mensal (todos os cadastrados)")
    print(f"Teto do p99: R$ {base['DESPESA_MENSAL'].quantile(0.99):.2f}/mês")
    print(comparar_impacto(base["DESPESA_MENSAL"], 0.99).round(2))

    base_winsorizada = base.assign(DESPESA_MENSAL_WINS=winsorizar(base["DESPESA_MENSAL"], 0.99))
    resultado_t = teste_t(base_winsorizada, "ACOMODACAO", "APARTAMENTO", "ENFERMARIA", "DESPESA_MENSAL_WINS")
    print("\nTeste t com a despesa winsorizada (Apartamento x Enfermaria)")
    print(f"Médias: R$ {resultado_t['media_a']:.2f} x R$ {resultado_t['media_b']:.2f} "
          f"(diferença de R$ {resultado_t['diferenca']:.2f})")
    print(f"t = {resultado_t['t']:.2f}, p-valor = {resultado_t['p_valor']:.2e}")

    #Parte 3: quem são os outliers
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

    #Parte 4: gráficos
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    grafico_boxplot(quem_usou["DESPESA_MENSAL"], limites["limite"],
                    PASTA_RESULTADOS / "outliers_boxplot.png")
    grafico_concentracao(base, "DESPESA_TOTAL",
                         PASTA_RESULTADOS / "outliers_concentracao.png")
    print(f"\nGráficos salvos em: {PASTA_RESULTADOS}")


#Item 4: modelos preditivos (parte 1: por beneficiário; parte 2: por atendimento, com especialidade)
def mostrar_modelo(base, despesas, mensalidade):
    print("\n========== ITEM 4: MODELO PREDITIVO ==========")

    #Parte 1: despesa mensal por beneficiário, só com variáveis de cadastro
    #CATEGORIA fica de fora: os Planos 3 e 4 têm só agregados, então plano e categoria seriam redundantes
    x, y = preparar_variaveis(base, ["IDADE", "SEXO", "ID_PLANO"], "DESPESA_MENSAL")
    x_treino, x_teste, y_treino, y_teste = dividir_treino_teste(x, y)
    print(f"\nTreino: {len(x_treino)} beneficiários | Teste: {len(x_teste)} beneficiários")

    modelos = {
        "Linha de base (chuta a média)": DummyRegressor(strategy="mean"),
        "Regressão linear": LinearRegression(),
        "Árvore de decisão": DecisionTreeRegressor(max_depth=4, min_samples_leaf=50, random_state=42),
    }

    print("\nDesempenho no teste (despesa mensal por beneficiário)")
    for nome, modelo in modelos.items():
        metricas = treinar_e_avaliar(modelo, x_treino, x_teste, y_treino, y_teste)
        print(f"{nome}: R² = {metricas['r2']:.4f} | RMSE = R$ {metricas['rmse']:.2f} | MAE = R$ {metricas['mae']:.2f}")

    print("\nRegressão linear (todos os dados): coeficiente, intervalo de confiança e p-valor")
    print(coeficientes_com_p_valor(x, y).round(4))

    print("\nÁrvore de decisão: importância de cada variável (soma 1)")
    print(pd.Series(modelos["Árvore de decisão"].feature_importances_, index=x.columns)
          .round(3).sort_values(ascending=False))

    #Parte 2: valor do atendimento, com a especialidade no lugar do CID
    atendimentos = manter_grupos_com_minimo(
        montar_base_atendimentos(despesas, mensalidade), "ESPECIALIDADE", 30
    )
    atendimentos["LOG_VALOR"] = np.log(atendimentos["VALOR"])
    print(f"\nParte 2: {len(atendimentos)} atendimentos | "
          f"valor mediano R$ {atendimentos['VALOR'].median():.2f} | médio R$ {atendimentos['VALOR'].mean():.2f}")

    cadastro = ["IDADE", "SEXO", "ID_PLANO"]
    conjuntos = {
        "A. Cadastro": cadastro,
        "B. + Especialidade": cadastro + ["ESPECIALIDADE"],
        "C. + Contexto": cadastro + ["ESPECIALIDADE", "REGIME_ATENDIMENTO",
                                     "NATUREZA_ATENDIMENTO", "CLINICA", "TIPO_DE_REDE"],
    }

    for nome_conjunto, colunas in conjuntos.items():
        print(f"\n{nome_conjunto} ({', '.join(colunas)})")
        for nome_modelo in ["Regressão linear", "Árvore de decisão"]:
            if nome_modelo == "Regressão linear":
                modelo_reais, modelo_log = LinearRegression(), LinearRegression()
            else:
                modelo_reais = DecisionTreeRegressor(max_depth=8, min_samples_leaf=100, random_state=42)
                modelo_log = DecisionTreeRegressor(max_depth=8, min_samples_leaf=100, random_state=42)

            x, y = preparar_variaveis(atendimentos, colunas, "VALOR")
            em_reais = treinar_e_avaliar(modelo_reais, *dividir_treino_teste(x, y))

            x, y = preparar_variaveis(atendimentos, colunas, "LOG_VALOR")
            em_log = treinar_e_avaliar(modelo_log, *dividir_treino_teste(x, y), funcao_avaliacao=avaliar_em_log)

            print(f"  {nome_modelo}: R² em reais = {em_reais['r2']:.4f} | "
                  f"R² em log = {em_log['r2']:.4f} | erro típico = x{em_log['erro_tipico_vezes']:.2f}")
            

def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    #1. Tratar as duas bases
    mensalidade = tratar_mensalidade(carregar_mensalidade())
    despesas = tratar_despesas(carregar_despesas())

    #2. Conferir o tamanho de cada uma
    print(f"Mensalidade: {mensalidade.shape[0]:,} linhas e {mensalidade.shape[1]} colunas".replace(",", "."))
    print(f"Despesas: {despesas.shape[0]:,} linhas e {despesas.shape[1]} colunas".replace(",", "."))

    #3. Montar a base analítica (uma linha por beneficiário cadastrado)
    meses = contar_meses(despesas)
    despesa_por_beneficiario = agregar_despesa_por_beneficiario(despesas)
    base = montar_base_analitica(mensalidade, despesa_por_beneficiario, meses)
    sem_cadastro = despesas_sem_beneficiario_cadastrado(despesas, mensalidade)

    linhas, colunas = base.shape
    print(f"Meses no período: {meses}")
    print(f"A base analítica tem {linhas:,} linhas e {colunas} colunas".replace(",", "."))
    print(f"Despesa dos cadastrados: R$ {base['DESPESA_TOTAL'].sum():,.2f}")
    print(f"Beneficiários que tiveram despesa: {(base['DESPESA_TOTAL'] > 0).sum():,}".replace(",", "."))
    print(f"Sem cadastro: {sem_cadastro['BENEFICIARIO'].nunique()} beneficiários, "
          f"R$ {sem_cadastro['VALOR_PAGO'].sum():,.2f}")

    #4. Fechamento: cadastrados + sem cadastro tem que dar o total das despesas tratadas
    total_cadastrados = base["DESPESA_TOTAL"].sum()
    total_sem_cadastro = sem_cadastro["VALOR_PAGO"].sum()
    total_despesas = despesas["VALOR_PAGO"].sum()
    assert round(total_cadastrados + total_sem_cadastro, 2) == round(total_despesas, 2), \
        "A soma de cadastrados + sem cadastro não fecha com o total das despesas"
    print("Fechamento conferido: nenhuma despesa se perdeu na junção")

    #5. Item 2: análises estatísticas
    mostrar_testes_estatisticos(base, despesas)

    #6. Item 3: outliers
    mostrar_outliers(base, despesas)

    #7. Item 4: modelo preditivo
    mostrar_modelo(base, despesas, mensalidade)


if __name__ == "__main__":
    main()