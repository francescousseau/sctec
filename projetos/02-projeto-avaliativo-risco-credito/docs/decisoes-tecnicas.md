# Decisões técnicas

> Cada decisão abaixo está **implementada** em `src/` e **executada** no notebook. Este documento é o registro do *porquê*; o código é o registro do *como*. Onde há um número, ele veio da execução real — não de estimativa.

---

## 1. Arquitetura: o pipeline como garantia estrutural

Todo o pré-processamento vive dentro de um `imblearn.pipeline.Pipeline` (`src/preprocessing.build_pipeline`). Imputadores, `StandardScaler`, `Winsorizer` e balanceador são ajustados **apenas no fold de treino**, porque só o `fit` do pipeline os toca.

Isso não é elegância: é a diferença entre um pipeline em que o vazamento é **estruturalmente impossível** e um em que ele é *evitado por disciplina*. Disciplina falha; arquitetura não.

Consequência prática: nenhum transformador do projeto aprende parâmetro fora de um `fit`.

---

## 2. Duplicatas — remover **antes** do split

**165 linhas** duplicadas removidas.

O motivo é vazamento, não estética. Se uma linha e sua cópia idêntica fossem sorteadas para lados opostos do split, o modelo teria **visto o gabarito da prova durante o estudo**, inflando artificialmente a métrica de teste.

---

## 3. Regras de negócio — separar erro de cadastro de outlier estatístico

Esta é a distinção conceitual mais importante da fase de limpeza. Um cliente que ganha muito é um **outlier legítimo** e precisa ser preservado — é justamente o bom pagador que o banco não quer perder. Um cliente de **144 anos** é **dado sujo**.

| Regra | Registros | Ação | Justificativa |
|---|---|---|---|
| Idade fora de 18–100 anos | 5 | remove a **linha** | O cadastro inteiro é suspeito |
| `emp_length > idade − 14` | 2 | anula o **campo** | A linha é aproveitável; só o campo corrompido vira `NaN` e é imputado depois |
| Renda ou empréstimo ≤ 0 | 0 | remove a linha | Inviabiliza a feature de comprometimento |

**Retenção final: 99,48% da base** (32.581 → 32.411).

Note a assimetria entre as duas primeiras regras: idade impossível compromete o cadastro todo; tempo de emprego impossível compromete apenas aquele campo. Descartar a linha inteira no segundo caso seria jogar fora informação boa.

---

## 4. Imputação — mediana, e mediana **por grade**

**Numéricas → mediana, não média.** A renda tem **skewness ≈ 25**: distribuição fortemente assimétrica à direita. A média seria arrastada pela cauda dos altos rendimentos e imputaria valores irrealisticamente altos. A mediana é robusta a isso.

**Categóricas → moda.**

**`loan_int_rate` → mediana dentro de cada `loan_grade`** (`RateByGradeImputer`). A taxa de juros é função quase determinística do rating do contrato — um cliente grade G não recebe a taxa mediana global. Imputar por grade preserva essa estrutura; a mediana global a achataria. Fallback para a mediana global cobre grades ausentes no treino.

---

## 5. Outliers — decisão **assimétrica** entre os modelos

Não há uma decisão correta única. Há uma decisão correta **por algoritmo**:

| | Árvore de Decisão | KNN |
|---|---|---|
| Como usa o valor | corte ordinal (`renda > X?`) | **distância euclidiana** |
| Efeito de um outlier | nenhum — a ordem não muda | **domina a métrica de distância** |
| **Decisão** | **manter** os valores íntegros | **clipping no percentil 99** |

**Por que não remover por IQR:** o IQR aponta ~4.000 linhas. Removê-las jogaria fora milhares de clientes de alta renda perfeitamente legítimos — que são exatamente os bons pagadores mais rentáveis. O clipping (`Winsorizer`) preserva a **linha** e apenas achata o **valor extremo**, e é aplicado só onde causa dano.

**Validação empírica** (F1 por validação cruzada):

| Modelo | Sem clipping | Com clipping p99 |
|---|---|---|
| KNN (k=9) | 0,6582 | **0,6616** ⬆ |
| Árvore (depth=7) | 0,7908 | 0,7914 ≈ |

O clipping melhora o KNN e é indiferente para a árvore — exatamente o que a teoria previa.

---

## 6. Balanceamento — **RandomUnderSampler**, e por que **não** SMOTE

O enunciado permite SMOTE **ou** Random Under Sampling. Testei os dois. O SMOTE tem dois defeitos graves neste pipeline:

### Defeito 1 — o SMOTE quebra o one-hot

O SMOTE interpola linearmente entre vizinhos. Aplicado sobre a matriz codificada, produz **valores fracionários nas colunas one-hot**. Medido na execução real:

> **12.685 de 40.514 linhas do treino balanceado (31,3%) têm one-hot quebrado.**
> Exemplo de linha sintética: `[0.71, 0.00, 0.00, 0.29, ...]` — um cliente que é **71% proprietário e 29% inquilino**.

Isso não existe no mundo real. A árvore então aprende cortes como `home_ownership_MORTGAGE <= 0.35`, que nunca se repetirão no teste.

### Defeito 2 — o SMOTE é sensível à escala

O SMOTE busca vizinhos por **distância euclidiana**. No ramo da árvore, cujos dados não são escalonados, `person_income` (~10⁴) domina totalmente a busca — os dados sintéticos viram interpolações quase só de renda.

Pior: isso **destrói a prova de invariância de escala da árvore**. Com SMOTE, escalonar muda os dados sintéticos e, portanto, muda a árvore — a concordância cai de 100% para ~92%. O balanceador precisa ser **livre de distância** para que a prova do §8 funcione.

### A escolha

O `RandomUnderSampler` não sintetiza nada, não calcula distância, não tem escala. E vence no que importa (validação cruzada, árvore `max_depth=7`):

| Estratégia | F1 (CV) | **Recall (CV)** |
|---|---|---|
| Sem balanceamento | 0,8167 | 0,7011 |
| SMOTE | 0,8003 | 0,6893 |
| `class_weight=balanced` | 0,7949 | **0,7307** |
| **RandomUnderSampler** | 0,7908 | **0,7307** |

Como o custo do negócio está concentrado no falso negativo, **recall é a métrica que paga a conta**. O RUS entrega o melhor recall entre as técnicas de reamostragem, sem fabricar clientes impossíveis.

> **Trade-off assumido:** o RUS descarta dados da classe majoritária, e seu F1 é ligeiramente inferior ao do cenário sem balanceamento. É uma troca consciente: aceito perder um pouco de F1 para ganhar recall, porque é o recall que evita calote.

---

## 7. Encoding — cada tipo com a técnica certa

| Variável | Técnica | Justificativa |
|---|---|---|
| `loan_grade` | **OrdinalEncoder** (A<B<...<G) | A taxa de inadimplência sobe **monotonicamente** de A para G. One-hot destruiria essa ordem — o próprio sinal — e ainda somaria 7 dimensões, veneno para o KNN (maldição da dimensionalidade). |
| `cb_person_default_on_file` | **binária** (N=0, Y=1) | Uma coluna, não duas. |
| `person_home_ownership`, `loan_intent` | **OneHotEncoder** | Genuinamente nominais: não existe "aluguel < hipoteca". |

---

## 8. Escalonamento — apenas no KNN, e **provado**

Não basta afirmar. `modeling.prove_scale_invariance` treina a **mesma árvore** com e sem `StandardScaler` e compara as predições no teste:

> **Concordância: 100,00%** (`assert concordancia == 1.0` no notebook)

**Por quê:** a árvore só pergunta *"renda > X?"*. O `StandardScaler` é uma transformação **monotônica** — não reordena valores. Logo, os mesmos cortes são encontrados, e escalonar é trabalho inútil.

Já o KNN mede **distância euclidiana**: sem escala, `person_income` (~10⁴) esmagaria `person_age` (~10¹), e a vizinhança seria decidida praticamente só pela renda.

---

## 9. Split — estratificado 80/20

```python
train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
```

O `stratify=y` preserva a proporção 78/22 nos dois conjuntos. Sem ele, o sorteio poderia entregar um teste com prevalência diferente, tornando a métrica incomparável.

---

## 10. Seleção de hiperparâmetros — **validação cruzada**, nunca o teste

Escolher `k` ou `max_depth` olhando a performance no teste é **selection leakage**: o holdout deixa de ser honesto e vira um conjunto de validação disfarçado.

Uso `StratifiedKFold` de 5 folds **dentro do treino**. O conjunto de teste é aberto **uma única vez**, na Fase 6. Os melhores parâmetros são extraídos **programaticamente** da tabela de CV (`select_best`), nunca chumbados a mão.

---

## 11. Diagnóstico de overfitting — o gap medido **corretamente**

**A armadilha:** comparar o F1 do treino **balanceado** (50% de positivos) com o F1 do teste (22% de positivos). O F1 depende da prevalência da classe positiva, então boa parte do "gap" resultante é **artefato de proporção**, não overfitting.

**A correção:** `f1_train` é calculado sobre o **treino real**, na distribuição original.

```text
gap = f1_train (distribuição real) − f1_cv
```

A diferença não é acadêmica. Com a métrica errada, a árvore `max_depth=3` aparentava um gap de 0,159 — alarmante. Com a métrica correta, o gap real é **−0,007**: não há overfitting nenhum.

**Diagnóstico final:**

| | Ponto de ruptura | Evidência |
|---|---|---|
| **Árvore** | `max_depth=None` | F1 de validação **despenca de 0,79 para 0,65**; gap salta para **0,15**. Sem limite, a árvore isola cada cliente numa folha — **memoriza** em vez de aprender. |
| **KNN** | `k=3` | Gap de **0,08**, encolhendo até **0,03** em k=9. Poucos vizinhos = o modelo copia o ruído local. |

**A lição:** complexidade tem **direções opostas** nos dois algoritmos. Na árvore é profundidade **alta**; no KNN é k **baixo**. Combater overfitting significa **podar a árvore e alargar a vizinhança**.

Selecionados: **`max_depth=7`** (f1_cv = 0,7908, gap = 0,0012) e **`k=9`** (f1_cv = 0,6616, gap = 0,0305).

---

## 12. Métrica de decisão — reais, não percentuais

**Acurácia é inútil aqui.** Aprovar todo mundo "acerta" 78% e não bloqueia um único calote.

A hierarquia de métricas usada:

1. **Custo financeiro estimado** — o critério final
2. **Recall da classe 1** — quantos calotes o modelo pega
3. **F1 da classe 1** — equilíbrio com a precisão
4. **Gap de generalização** — o modelo aprendeu ou decorou?
5. PR-AUC — mais honesta que ROC-AUC em base desbalanceada

### O modelo de custo (`evaluation.BusinessCost`)

Custos fixos e arbitrários (FP = R$ 500, FN = R$ 5.000) escondem o fato de que **errar num empréstimo de R$ 35 mil não custa o mesmo que errar num de R$ 1 mil**. O custo é modelado sobre o **valor real de cada contrato**:

| Erro | Fórmula | Premissa |
|---|---|---|
| **Falso Negativo** | `loan_amnt × LGD` | **LGD = 65%** — o banco perde o **principal**; parte é recuperada via cobrança |
| **Falso Positivo** | `loan_amnt × taxa × duração` | **Duração = 2 anos** — o banco perde a **receita de juros**, não o capital |

**Nota de terminologia.** A fórmula do FP estima a **receita bruta de juros não realizada**, e não a *margem líquida*. Ela não desconta custo de captação, custo de capital, custo operacional, amortização do principal, perda esperada nem impostos. É uma proxy simplificada — e conservadora, já que superestima o custo de rejeitar um bom pagador.

Ambas são **hipóteses declaradas**. Por isso existe a análise de sensibilidade (§13).

---

## 13. Robustez do veredito — análise de sensibilidade

Se o modelo vencedor mudasse ao variar o LGD, a recomendação seria frágil e dependeria de eu ter acertado um chute.

| LGD | KNN | Árvore | Vencedor |
|---|---|---|---|
| 0,45 | R$ 3.332.552 | R$ 2.134.392 | **Árvore** |
| 0,65 | R$ 3.850.592 | R$ 2.818.327 | **Árvore** |
| 0,85 | R$ 4.368.632 | R$ 3.502.262 | **Árvore** |

**A árvore vence em toda a faixa plausível (35% a 95%).** O veredito é robusto.

---

## 14. O achado contraintuitivo — por que nenhuma métrica isolada decide

| | KNN (k=9) | **Árvore (depth=7)** |
|---|---|---|
| Recall (classe 1) | **0,783** ⬅ melhor | 0,736 |
| Falsos Negativos | **308** ⬅ menos calotes! | 374 |
| Falsos Positivos | 821 | **229** |
| **Custo total** | R$ 3.850.592 | **R$ 2.818.327** ⬅ vence |

O **KNN captura 66 inadimplentes a mais**. Se a decisão fosse tomada pelo recall, ele venceria.

Mas, para conseguir isso, **barra 592 bons pagadores a mais** — cada um deles é margem de juros que evapora, além de custo reputacional e possível exclusão financeira injusta. Somando tudo, o KNN custa **mais de R$ 1 milhão a mais**.

> **O recall diz que o KNN é melhor. A conta bancária diz o contrário.**

---

## 15. Otimização de threshold — aprendida no treino, não no teste

O corte de 0,50 é ótimo **apenas quando FP e FN custam a mesma coisa** — o que quase nunca é verdade em crédito.

### A armadilha que este projeto quase caiu

Varrer os thresholds contra `y_test` e escolher o que minimiza o custo **é selection leakage**. É exatamente o erro do §10, cometido num lugar diferente: o holdout deixa de medir e passa a decidir. O número resultante é otimista, porque o corte foi ajustado ao ruído daquela amostra específica.

### A correção

O threshold é aprendido com **probabilidades out-of-fold do treino** (`modeling.out_of_fold_proba`): cada observação recebe a probabilidade prevista pelo fold que **não a viu**. Minimizamos o custo sobre elas, fixamos o corte, e só então o aplicamos ao teste — **uma vez, sem reajuste**.

| Threshold | Origem | Custo no teste |
|---|---|---|
| 0,50 | padrão | R$ 2.815.557 |
| **0,69** | **out-of-fold no treino (honesto)** | **R$ 2.726.060** |
| 0,77 | otimizado no teste (vazado) | R$ 2.711.511 |

**Ganho honesto: R$ 89.497**, sem trocar uma linha do modelo.

### Quanto o vazamento teria custado em credibilidade

O threshold vazado prometeria R$ 2.711.511 — **R$ 14.549 a menos (0,53%)**. O otimismo é pequeno e a conclusão sobrevive intacta. Mas a magnitude do erro não é o ponto: **o método é.** Um pipeline que prega "o teste é aberto uma vez" e depois ajusta o corte de decisão contra ele se contradiz. O teste mede; não escolhe.

---

## 16. Responsible AI — o que faltaria antes de produção

Nada disto foi feito neste projeto, e **é honesto dizer**:

- **Validação temporal (out-of-time).** O risco de crédito muda com o ciclo econômico. Um split aleatório **superestima** a performance futura. Esta é a limitação mais séria do trabalho.
- **Auditoria de viés.** O modelo penaliza sistematicamente algum grupo protegido? Não sei — e não saber é motivo suficiente para não colocar em produção.
- **Explicabilidade individual (SHAP).** A árvore já é auditável em nível global; falta a explicação caso a caso.
- **Calibração de probabilidades.** A otimização de threshold pressupõe probabilidades calibradas — árvores raramente as fornecem.
- **Monitoramento de drift.** O perfil dos solicitantes muda; o modelo, não.
- **Governança e conformidade regulatória.** Em crédito, a explicabilidade não é luxo: é exigência legal.


---

## 17. Disponibilidade temporal das variáveis — o risco mais sério do projeto

Uma ressalva que, se confirmada, invalidaria boa parte dos resultados.

**`loan_grade` é o rating de risco que a própria instituição atribui ao contrato.** `loan_int_rate` deriva dele. Se forem gerados *durante* a análise de crédito, o modelo não prevê inadimplência de forma independente — ele **reaproveita a avaliação de risco que o banco já fez**.

Isso não é hipotético: `loan_grade` é a **variável #1 do modelo** (≈33% da importância), e a taxa de juros é função dela. Boa parte do desempenho observado pode refletir a qualidade do processo de rating existente, não a capacidade preditiva do modelo.

### O que foi assumido

Que ambas estão disponíveis **antes** da decisão final de concessão. É a leitura mais natural do dataset público — mas é **hipótese, não fato verificado**.

### O que uma implantação real exigiria

1. **Validar o fluxo operacional:** em que momento exato cada variável é produzida?
2. **Se forem posteriores à aprovação:** removê-las e retreinar. O modelo atual seria inutilizável em produção.
3. **Treinar dois cenários:**
   - **completo** — com `loan_grade` e `loan_int_rate` (o que este projeto entrega);
   - **pré-aprovação** — sem elas, medindo quanto o modelo agrega *antes* de o bureau opinar.

O cenário de pré-aprovação está nos próximos passos. Declará-lo como limitação é mais honesto do que executá-lo às pressas.
