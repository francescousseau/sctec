# Pipeline Preditivo para Análise de Risco de Crédito

Projeto avaliativo da trilha **Machine Learning e Visão Computacional — SCTEC/SENAI**.

---

## 1. Problema de negócio

Uma instituição financeira precisa decidir, **antes de liberar o crédito**, se um cliente honrará o empréstimo. O modelo prevê a variável `loan_status`:

- `0` — cliente adimplente (pagou em dia);
- `1` — cliente **inadimplente** (calote).

O desafio não é acertar o máximo de previsões: é **errar do lado certo**. Os dois erros possíveis têm naturezas econômicas opostas:

| Erro | O que aconteceu | Custo para o banco |
|---|---|---|
| **Falso Positivo (FP)** | Bom pagador classificado como risco → **crédito negado** | Margem de juros perdida, pior experiência do cliente, exclusão financeira indevida |
| **Falso Negativo (FN)** | Inadimplente classificado como seguro → **crédito concedido** | Perda direta do principal emprestado, custo de cobrança, provisionamento |

A hipótese inicial — confirmada pela simulação financeira deste projeto — é que o **FN é o erro mais caro**, porque o banco perde o capital, enquanto no FP perde apenas a margem. Mas, como o relatório mostra, **isso não significa que o modelo com melhor recall seja o vencedor** (veja o Resumo Executivo).

---

## 2. Dicionário de dados

### Variáveis originais (12 colunas, 32.581 registros)

| Coluna | Tipo | Descrição |
|---|---|---|
| `person_age` | int | Idade do solicitante, em anos |
| `person_income` | int | Renda anual declarada (US$) |
| `person_home_ownership` | categórica | Situação de moradia: `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `person_emp_length` | float | Tempo de emprego, em anos |
| `loan_intent` | categórica | Finalidade do empréstimo: `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_grade` | **ordinal** | Rating de crédito atribuído ao contrato, de `A` (melhor) a `G` (pior) |
| `loan_amnt` | int | Valor do empréstimo solicitado (US$) |
| `loan_int_rate` | float | Taxa de juros do contrato (% a.a.) |
| `loan_percent_income` | float | Proporção da renda comprometida com o empréstimo (0–1) |
| `cb_person_default_on_file` | binária | Histórico de inadimplência no bureau de crédito: `Y` / `N` |
| `cb_person_cred_hist_length` | int | Tempo de histórico de crédito, em anos |
| **`loan_status`** | **alvo** | `1` = inadimplente · `0` = adimplente |

### ⭐ Variável criada (Feature Engineering)

| Coluna | Fórmula | Descrição e justificativa |
|---|---|---|
| **`comprometimento_renda`** | `(loan_amnt / person_income) * 100` | Percentual da renda anual comprometido com o empréstimo. É a clássica razão dívida/renda, o indicador mais direto de **capacidade de pagamento**. Resultou na **2ª variável mais importante do modelo final** (≈ 30% da importância), atrás apenas do `loan_grade`. |

> **Nota sobre redundância:** `loan_percent_income` correlaciona-se com `comprometimento_renda` a **r = 0,9989** — é a mesma informação em outra escala. Manter as duas duplicaria o peso dessa dimensão no cálculo de distância do KNN, então `loan_percent_income` **é usada na EDA mas removida da matriz de modelagem**.

---

## 3. Resumo executivo (para a diretoria)

### Insights da análise exploratória

1. **A base é desbalanceada: 78,2% adimplentes contra 21,8% inadimplentes.** Isso condena a acurácia como métrica de decisão — aprovar todo mundo já "acerta" 78% e não bloqueia um único calote. A seleção do modelo foi feita por **F1 e recall da classe inadimplente**, e o veredito final foi dado **em reais**.
2. **Existem erros de cadastro, não apenas outliers:** clientes com **144 anos de idade** e **123 anos de tempo de emprego**. Foram tratados como problema de qualidade (removidos/anulados), não como cauda estatística. A limpeza preservou **99,48%** da base.
3. **A renda é fortemente assimétrica à direita** (skewness ≈ 25). Esse é o argumento que justifica a **mediana** como imputador — a média seria arrastada pela cauda dos altos rendimentos.
4. **`loan_grade` e comprometimento de renda são os sinais mais fortes.** A taxa de inadimplência sobe monotonicamente de A para G — motivo pelo qual o grade foi tratado como variável **ordinal**, e não one-hot: codificar A–G como sete colunas independentes destruiria justamente a ordem que carrega o sinal.

### Diagnóstico de overfitting

| | Ponto de ruptura | Evidência |
|---|---|---|
| **Árvore** | `max_depth=None` | F1 de validação **despenca de 0,79 para 0,65**; gap salta para **0,15**. Sem limite, a árvore isola cada cliente numa folha — memoriza em vez de aprender. |
| **KNN** | `k=3` | Gap de **0,08**, encolhendo até 0,03 em k=9. Poucos vizinhos = o modelo copia o ruído local. |

**A lição:** complexidade tem direções opostas nos dois algoritmos. Na árvore, complexidade é profundidade **alta**; no KNN, é k **baixo**. Combater overfitting significa **podar a árvore e alargar a vizinhança**.

### 🏆 Veredito: **Árvore de Decisão com `max_depth = 7`**

Desempenho no conjunto de teste (6.483 clientes nunca vistos pelo modelo):

| Métrica | KNN (k=9) | **Árvore (depth=7)** |
|---|---|---|
| Acurácia | 0,826 | **0,907** |
| Precisão (classe 1) | 0,575 | **0,820** |
| Recall (classe 1) | **0,783** | 0,736 |
| F1 (classe 1) | 0,663 | **0,776** |
| PR-AUC | 0,750 | **0,823** |
| Gap de generalização | 0,031 | **0,001** |
| Falsos Positivos | 821 | **229** |
| Falsos Negativos | **308** | 374 |
| **💰 Custo total do erro** | R$ 3.850.592 | **R$ 2.818.327** |

**O achado contraintuitivo:** o KNN tem **melhor recall** e captura **66 inadimplentes a mais** que a árvore. Se a decisão fosse tomada pelo recall, ele venceria. Mas, para conseguir isso, o KNN **barra 592 bons pagadores a mais** — margem de juros que evapora. Somando calotes e margens perdidas sobre o valor real de cada contrato, o KNN custa **mais de R$ 1 milhão a mais**.

> É a demonstração de que **nenhuma métrica isolada decide um caso de negócio**: o recall diz que o KNN é melhor; a conta bancária diz o contrário.

**Impacto financeiro:**

- Cenário sem modelo (aprovar todos): **R$ 9.930.164** de prejuízo
- Com a Árvore: **R$ 2.818.327** → **71,6% do prejuízo evitado**

**Robustez:** variando a premissa de perda dada a inadimplência (LGD) de 35% a 95%, **a árvore vence em toda a faixa**. O veredito não depende de acertar a premissa.

**Recomendações operacionais:**

1. **Ajustar o threshold de decisão de 0,50 para ≈ 0,77** → custo cai para **R$ 2.711.511** (mais R$ 107 mil economizados sem trocar uma linha do modelo).
2. **Manter a árvore também por governança.** Ela é auditável: cada negativa pode ser explicada ao cliente e ao regulador em linguagem natural ("grade E + 45% da renda comprometida"). O KNN não oferece isso — e em crédito a explicabilidade é exigência regulatória, não luxo.
3. **Usar o modelo como triagem, não como juiz.** Casos de fronteira (probabilidade entre 0,4 e 0,7) devem ir para análise humana.

---

## 4. Premissas do modelo de custo

O custo **não** usa valores fixos arbitrários: errar num empréstimo de R$ 35 mil não custa o mesmo que errar num de R$ 1 mil. O custo é calculado sobre o **valor real de cada contrato** do conjunto de teste:

| Erro | Fórmula | Premissa |
|---|---|---|
| Falso Negativo | `loan_amnt × LGD` | **LGD (Loss Given Default) = 65%** — parte do principal é recuperada via cobrança |
| Falso Positivo | `loan_amnt × taxa_juros × duração` | **Duração média = 2 anos**; representa a margem de juros que o banco deixou de ganhar |

Ambas são **hipóteses declaradas**, por isso o projeto inclui uma **análise de sensibilidade** variando o LGD.

---

## 5. Decisões técnicas — e por quê

| # | Decisão | Justificativa |
|---|---|---|
| 1 | Remover duplicatas **antes** do split | Se uma linha e sua cópia caíssem em lados opostos do split, o modelo veria o gabarito da prova durante o estudo → **vazamento**. |
| 2 | Idade fora de 18–100 → remover | Erro de cadastro, não outlier estatístico. |
| 3 | `emp_length > idade − 14` → `NaN` + imputar | Logicamente impossível. Preserva a linha, descarta só o campo corrompido. |
| 4 | Imputar `loan_int_rate` pela **mediana do `loan_grade`** | A taxa é função quase determinística do rating; a mediana global achataria essa informação. |
| 5 | Imputar numéricas pela **mediana** (não média) | Distribuições fortemente assimétricas (skewness da renda ≈ 25). |
| 6 | **Outliers: manter na árvore, clipping p99 no KNN** | A árvore faz cortes ordinais (indiferente a escala); o KNN mede distância euclidiana e é dominado por valores extremos. Decisão **assimétrica**, validada empiricamente: o clipping melhora o F1 do KNN e é indiferente para a árvore. |
| 7 | `loan_grade` → **OrdinalEncoder** (A<B<...<G) | One-hot destruiria a ordem que carrega o sinal e somaria 7 dimensões — veneno para o KNN. |
| 8 | Remover `loan_percent_income` | r = 0,9989 com a feature criada → peso duplicado no KNN. |
| 9 | Split estratificado 80/20 | Preserva a proporção 78/22 nos dois conjuntos. |
| 10 | **`RandomUnderSampler`** em vez de SMOTE | O SMOTE (a) **quebra o one-hot**, gerando clientes "71% dono e 29% inquilino", e (b) é **sensível à escala**, o que destruiria a prova de invariância da árvore. O RUS não sintetiza, não usa distância — e entregou **melhor recall** na validação cruzada. |
| 11 | **Escalonar apenas o KNN** | Provado no notebook: a árvore com e sem `StandardScaler` produz **predições 100% idênticas**. |
| 12 | Hiperparâmetros por **validação cruzada no treino** | Escolher `k`/`max_depth` olhando o teste é *selection leakage*. O teste é aberto **uma única vez**, na fase final. |
| 13 | Gap medido no **treino real**, não no balanceado | F1 depende da prevalência: comparar treino balanceado (50% positivos) com teste (22%) infla o "gap" com artefato de proporção, não overfitting. |
| 14 | Todo o pré-processamento dentro de um `Pipeline` | Torna o vazamento **estruturalmente impossível**: imputadores, scaler e balanceador são ajustados apenas no fold de treino. |

---

## 6. Estrutura do projeto

```text
02-projeto-avaliativo-risco-credito/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/credit_risk_dataset.csv
│   └── processed/
├── notebooks/
│   └── 01_risco_credito_pipeline.ipynb   # narrativa analítica (importa de src/)
├── src/                                  # fonte única da verdade
│   ├── config.py            # constantes, features e premissas de custo
│   ├── data_quality.py      # carga, auditoria de regras, limpeza
│   ├── preprocessing.py     # transformadores + montagem dos pipelines
│   ├── modeling.py          # grid por CV, gap real, prova de escala
│   └── evaluation.py        # custo de negócio, threshold, sensibilidade
├── reports/
│   ├── figures/          # 11 gráficos gerados automaticamente
│   └── results/          # tabelas em CSV
└── docs/
    ├── roteiro-video.md
    ├── guia-git.md
    └── decisoes-tecnicas.md
```

---

## 7. Como executar

### 1. Criar o ambiente

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Registrar o kernel

```bash
python -m ipykernel install --user \
  --name risco-credito-sctec \
  --display-name "Python - Risco de Crédito SCTEC"
```

### 4. Abrir o notebook

```bash
jupyter lab notebooks/01_risco_credito_pipeline.ipynb
```

Execute todas as células em ordem (`Run All`). As figuras são gravadas em `reports/figures/` e as tabelas em `reports/results/`.

---

## 8. Modelagem

**KNN:** `n_neighbors` = 3, 5, 7, 9
**Árvore de Decisão:** `max_depth` = 3, 5, 7, None

Métricas monitoradas: acurácia, precisão, recall, F1, PR-AUC, matriz de confusão, gap de generalização e **custo financeiro estimado**.

---

## 9. Limitações

- Base pública e possivelmente sintética/anonimizada.
- LGD e duração média são **premissas**, não dados reais do banco.
- **Não há validação temporal.** O risco de crédito muda com o ciclo econômico; um split aleatório superestima a performance futura.
- **Não foi feita auditoria de viés.** Antes de qualquer produção seria obrigatório verificar se o modelo penaliza sistematicamente algum grupo protegido.
- **Não deve ser usado em produção como está.**

## 10. Próximos passos

Regressão logística como baseline interpretável · Random Forest / LightGBM · SHAP para explicabilidade individual · calibração de probabilidades · validação out-of-time · auditoria de fairness · monitoramento de drift.

---

## Autor

**Francesco Cousseau**
SCTEC/SENAI — Machine Learning e Visão Computacional
