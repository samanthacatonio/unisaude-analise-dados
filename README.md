# Avaliação Técnica - Analista de Dados (UNISAÚDEMS)

Neste projeto analisei as despesas assistenciais e a arrecadação de mensalidades de uma operadora de plano de saúde (os dados são fictícios).

Este código faz a preparação dos dados e os itens 2, 3 e 4 da avaliação: os testes estatísticos, a análise de outliers e o modelo preditivo. 
A análise exploratória (item 1) e o dashboard (item 6) estão no arquivo do Power BI.

## Como rodar

**1. Onde colocar as planilhas.** As duas bases precisam ficar na pasta acima da pasta `codigo`:

```
Unisaude/
├── BASE DE DADOS - DESPESAS - OFICIAL.xlsx
├── BASE DE DADOS - MENSALIDADE - OFICIAL.xlsx
├── codigo/          (este código)
└── resultados/      (é criada sozinha, com os gráficos)
```

**2. Instalar as bibliotecas** (usei o Python 3.14.7):

```
python -m pip install -r requirements.txt
```

**3. Rodar**, de dentro da pasta `codigo`:

```
python main.py
```

A primeira vez demora alguns minutos, pois a planilha de despesas é grande (quase 956 mil linhas). Nessa primeira leitura, o código salva uma cópia em um arquivo mais leve (`despesas_bruta.pkl`), e a partir da segunda vez ele lê essa cópia e roda em segundos.

Também dá para rodar cada arquivo sozinho (por exemplo, `python estatistica.py`), para ver só aquela parte.

## Como o código está organizado

Separei o código em arquivos, cada um com uma responsabilidade:

| Arquivo | O que faz |
|---|---|
| `tratamento_mensalidade.py` | lê a base de mensalidades, renomeia a idade, remove os beneficiários duplicados e cria a faixa etária |
| `tratamento_despesas.py` | lê a base de despesas e remove as linhas que não são despesas reais |
| `base_analitica.py` | junta as duas bases: uma tabela com uma linha por beneficiário e outra com uma linha por atendimento |
| `estatistica.py` | item 2: teste t, ANOVA (com Tukey) e correlação |
| `outliers.py` | item 3: identifica os outliers, mede o impacto deles, mostra quem são e gera os gráficos |
| `modelo.py` | item 4: regressão linear e árvore de decisão, avaliadas com R², RMSE e MAE |
| `main.py` | roda tudo na ordem e confere se os totais fecham |

Cada função faz uma coisa só: recebe uma tabela, faz o tratamento ou a análise e devolve o resultado, sem alterar a tabela original.

## Tratamentos que fiz nos dados

**Na base de despesas**
- **Removi as linhas com valor 99.999.** São 25.290 linhas, todas com quantidade zero. Se a quantidade é zero, o procedimento não foi feito nem cobrado, então esse 99.999 é um valor-padrão do sistema, e não um pagamento de verdade. Essas linhas somam R$ 2,53 bilhões, ou seja, 95,6% do valor da base. Se elas ficassem, qualquer média ou teste estaria errado.
- **Removi 8 linhas sem beneficiário** (R$ 967,60), pois não tem como ligar essas despesas a ninguém.
- **Mantive as quantidades fracionadas** (como 0,5). Se a coluna fosse convertida para número inteiro, 63 linhas válidas seriam perdidas.
- Depois disso, ficaram 930.615 linhas e R$ 117.252.894,92 em despesas.

**Na base de mensalidades**
- Usei a coluna `IDADE+2` como a idade do beneficiário.
- **Encontrei 4 beneficiários duplicados.** Eles são da mesma família e aparecem em dois planos diferentes, o que me leva a pensar que houve uma troca de plano e o cadastro antigo não foi atualizado. Como nas planilhas não tem como saber qual é o plano atual, mantive a linha de **menor mensalidade**, pois prefiro considerar uma receita menor e depois sobrar dinheiro do que considerar uma receita maior e faltar.
- Criei as faixas etárias seguindo o padrão da ANS (00 a 18, 19 a 23, ..., 54 a 58 e 59+).
- Percebi que os Planos 3 e 4 têm só agregados, e os agregados estão só nos Planos 3 e 4.

**Na junção das duas bases**
- A base principal parte da mensalidade (12.393 beneficiários). Quem não teve despesa no período entra com despesa **zero**, e não fica de fora.
- **486 beneficiários tiveram despesa, mas não estão na base de mensalidade.** Isso soma R$ 8.288.127,85 (7% do total). Não removi essas despesas: separei e mostrei à parte, pois o correto é que todo valor pago esteja ligado a um beneficiário cadastrado, e esse valor precisa ser investigado. Eles só ficam fora das análises por plano e por idade, já que não têm essas informações.
- No código, confiro se a despesa dos cadastrados mais a despesa sem cadastro dá exatamente o total das despesas. Assim tenho certeza de que nada se perdeu na junção.
- O número de meses não está escrito no código: ele é calculado pelas competências que existem na base (aqui são 9). Assim, se a base mudar, o cálculo da despesa mensal se ajusta sozinho.
- A base de mensalidade é a previsão de **um** mês, e a de despesas tem **nove**. Por isso, quando comparo receita e despesa, uso sempre os valores por mês.

## Por que escolhi cada método

### Teste t
- **O que usei:** teste t de Welch, confirmado pelo teste de Mann-Whitney.
- **Por quê:** os grupos têm variabilidades diferentes (confirmado pelo teste de Levene), e a despesa é muito concentrada em poucos valores altos.

### ANOVA
- **O que usei:** ANOVA, confirmada pelos testes de Alexander-Govern e Kruskal-Wallis. Usei o eta² para medir o tamanho do efeito e o teste de Tukey para saber quais grupos são diferentes entre si.
- **Por quê:** pelos mesmos motivos do teste t.

### Especialidade
- **O que usei:** analisei por atendimento (uma conta em uma especialidade), só com as especialidades que têm pelo menos 30 atendimentos.
- **Por quê:** um beneficiário passa por várias especialidades, e grupos muito pequenos dão médias que não são confiáveis.

### Correlação
- **O que usei:** Pearson e Spearman, por beneficiário e por grupo (plano x faixa etária, usando médias).
- **Por quê:** se eu usasse somas por grupo, os grupos grandes teriam receita e despesa grandes só por terem mais gente, e a correlação ia parecer maior do que é.

### Outliers
- **O que usei:** a regra do IQR (acima de Q3 + 1,5 x IQR), entre quem usou o plano. Medi o impacto limitando os valores no percentil 99.
- **Por quê:** é um método que não é puxado pelos próprios valores extremos. Os outliers são gastos reais, então não removi: medi o quanto eles pesam.

### Modelo
- **O que usei:** 80% dos dados para treinar e 20% para testar, comparando com um "modelo" que só chuta a média.
- **Por quê:** para avaliar o modelo em dados que ele nunca viu e saber se ele é melhor que um chute.

### Modelo por beneficiário
- **O que usei:** idade, sexo e plano. Deixei a categoria de fora.
- **Por quê:** como os Planos 3 e 4 são só de agregados, plano e categoria contam a mesma coisa, e o modelo não consegue separar o efeito de cada um.

### Coeficientes do modelo
- **O que usei:** statsmodels, com erros-padrão robustos (HC3).
- **Por quê:** para ter o intervalo de confiança e o p-valor de cada variável, considerando que a variabilidade não é igual entre os grupos.

### Modelo por atendimento
- **O que usei:** a especialidade no lugar do CID, e também previ o valor em escala logarítmica.
- **Por quê:** a base não tem CID, e os valores vão de R$ 10 a R$ 156 mil.

## Limitações

- A base não tem CID, histórico de uso dos beneficiários nem datas de entrada e saída do plano.
- A receita do período foi estimada a partir da mensalidade de um único mês.
- Os resultados mostram associações entre as variáveis, e não relações de causa e efeito.
