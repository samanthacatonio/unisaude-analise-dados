import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


#separa as variáveis explicativas (X) e a variável a prever (y), transformando texto em colunas 0/1
def preparar_variaveis(df_base, colunas_explicativas, coluna_alvo):
    df_x = df_base[colunas_explicativas].copy()
    if "ID_PLANO" in df_x.columns:
        df_x["ID_PLANO"] = df_x["ID_PLANO"].astype(str)   #plano é categoria, não quantidade
    x = pd.get_dummies(df_x, drop_first=True, dtype=int)
    y = df_base[coluna_alvo]
    return x, y


#divide em 80% para treino e 20% para teste (random_state fixo para o resultado ser sempre o mesmo)
def dividir_treino_teste(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)


#calcula as métricas de desempenho: R², RMSE e MAE
def avaliar(y_real, y_previsto):
    return {
        "r2": r2_score(y_real, y_previsto),
        "rmse": np.sqrt(mean_squared_error(y_real, y_previsto)),
        "mae": mean_absolute_error(y_real, y_previsto),
    }


#treina o modelo com os dados de treino e avalia nos dados de teste
def treinar_e_avaliar(modelo, x_treino, x_teste, y_treino, y_teste, funcao_avaliacao=avaliar):
    modelo.fit(x_treino, y_treino)
    return funcao_avaliacao(y_teste, modelo.predict(x_teste))

#métricas quando o modelo prevê o logaritmo do valor: R² e erro típico em "vezes"
def avaliar_em_log(y_real_log, y_previsto_log):
    rmse_log = np.sqrt(mean_squared_error(y_real_log, y_previsto_log))
    return {
        "r2": r2_score(y_real_log, y_previsto_log),
        "erro_tipico_vezes": np.exp(rmse_log),
    }


#ajusta a regressão linear em todos os dados e mostra coeficiente, intervalo de confiança e p-valor
#(erros-padrão robustos HC3, porque a variabilidade da despesa não é igual entre os grupos)
def coeficientes_com_p_valor(x, y):
    resultado = sm.OLS(y, sm.add_constant(x.astype(float))).fit(cov_type="HC3")
    intervalo = resultado.conf_int()
    return pd.DataFrame({
        "coeficiente": resultado.params,
        "ic_95_min": intervalo[0],
        "ic_95_max": intervalo[1],
        "p_valor": resultado.pvalues,
    })


if __name__ == "__main__":
    from tratamento_mensalidade import carregar_mensalidade, tratar_mensalidade
    from tratamento_despesas import carregar_despesas, tratar_despesas
    from base_analitica import (
        contar_meses,
        agregar_despesa_por_beneficiario,
        montar_base_analitica,
        montar_base_atendimentos,
    )
    from estatistica import manter_grupos_com_minimo

    mensalidade = tratar_mensalidade(carregar_mensalidade())
    despesas = tratar_despesas(carregar_despesas())
    base = montar_base_analitica(mensalidade, agregar_despesa_por_beneficiario(despesas), contar_meses(despesas))

    #Parte 1: despesa mensal por beneficiário, só com variáveis de cadastro
    #CATEGORIA fica de fora: os Planos 3 e 4 têm só agregados, então plano e categoria seriam redundantes
    x, y = preparar_variaveis(base, ["IDADE", "SEXO", "ID_PLANO"], "DESPESA_MENSAL")
    x_treino, x_teste, y_treino, y_teste = dividir_treino_teste(x, y)
    print(f"Treino: {len(x_treino)} beneficiários | Teste: {len(x_teste)} beneficiários")

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
    arvore = modelos["Árvore de decisão"]
    print(pd.Series(arvore.feature_importances_, index=x.columns).round(3).sort_values(ascending=False))

    #Parte 2: valor do atendimento, com a especialidade no lugar do CID
    atendimentos = manter_grupos_com_minimo(
        montar_base_atendimentos(despesas, mensalidade), "ESPECIALIDADE", 30
    )
    atendimentos["LOG_VALOR"] = np.log(atendimentos["VALOR"])
    print(f"\n\nParte 2: {len(atendimentos)} atendimentos | "
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