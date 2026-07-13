# Roteiro do vídeo — 7 minutos

> A rubrica exige que os **5 tópicos do item 5.3** sejam respondidos. Cada bloco abaixo está marcado com o tópico que ele responde. Rosto visível + tela compartilhada.

---

## 0:00 – 0:45 — Contexto e objetivo `[TÓPICO 1]`

> "Escolhi a **Base A — Risco de Crédito**. O objetivo não é acertar o máximo de previsões: é **errar do lado certo**. Um banco que aprova todo mundo já acerta 78% dos casos nessa base — e quebra, porque não bloqueia um único calote. Então a pergunta que eu respondo aqui não é 'qual modelo tem mais acurácia', é **'qual modelo perde menos dinheiro'**."

Mostrar: gráfico do desbalanceamento (78,2% × 21,8%).

---

## 0:45 – 2:00 — EDA: o que mudou minha visão `[TÓPICO 2]`

Três insights, um por vez:

1. **"Encontrei clientes de 144 anos e com 123 anos de tempo de emprego."**
   > "Isso não é outlier estatístico, é erro de cadastro. A distinção importa porque muda o tratamento: outlier eu penso duas vezes antes de remover; erro de digitação eu corrijo."

2. **"A renda tem skewness de 25."** (mostrar histograma)
   > "É esse número que justifica a mediana, e não a média, em toda imputação. A média seria arrastada pela cauda dos milionários."

3. **"A taxa de inadimplência sobe de forma monotônica de A até G."** (mostrar barplot por `loan_grade`)
   > "Isso me fez tratar `loan_grade` como **variável ordinal**, não one-hot. One-hot transformaria A, B, C... em sete colunas independentes e **destruiria exatamente a ordem que carrega o sinal de risco**."

---

## 2:00 – 3:10 — Nulos, outliers e o impacto DIFERENTE em cada modelo `[TÓPICO 3]`

**Nulos:**
> "Mediana, pelo argumento da assimetria. Mas fui além: a taxa de juros é quase uma função determinística do `loan_grade`, então imputo pela **mediana dentro de cada grade** — muito mais informativo que uma mediana global."

**Outliers — o ponto central do bloco:** mostrar a tabela.

| | Árvore | KNN |
|---|---|---|
| Como usa o valor | corte ordinal (`renda > X?`) | **distância euclidiana** |
| Efeito do outlier | nenhum | **domina o cálculo** |
| Decisão | **manter** | **clipping p99** |

> "Tomei uma decisão **assimétrica**, e não uma remoção cega por IQR. Remover as 4 mil linhas do IQR jogaria fora clientes de alta renda perfeitamente legítimos — que são justamente os bons pagadores que o banco não quer perder. Então **mantenho na árvore e faço clipping só no KNN**, porque só lá o outlier faz estrago. E validei isso: o clipping melhora o F1 do KNN e é indiferente para a árvore."

**Escalonamento (mostrar a célula da prova):**
> "Não basta afirmar que a árvore dispensa escala — eu **provo**. Treino a mesma árvore com e sem StandardScaler: **as predições são 100% idênticas**. Porque a árvore só pergunta 'renda maior que X?', e o StandardScaler é uma transformação monotônica: ele não reordena nada."

---

## 3:10 – 4:30 — Overfitting: onde cada modelo quebra `[TÓPICO 4]`

Mostrar as **curvas de complexidade** lado a lado.

> "Antes de mostrar os números, duas correções metodológicas que fazem toda a diferença:
> Primeira: **eu escolho o hiperparâmetro por validação cruzada dentro do treino**, nunca olhando o teste. Escolher `k` olhando a performance de teste é vazamento — o holdout deixa de ser honesto.
> Segunda: **meço o F1 de treino na distribuição real**, não na balanceada. Comparar treino com 50% de positivos contra teste com 22% infla o 'gap' com artefato de prevalência — isso não é overfitting, é aritmética."

**Árvore:**
> "Em profundidade 3, 5 e 7 o gap é **praticamente zero**. O ponto de ruptura é brutal em `max_depth=None`: o F1 de validação **despenca de 0,79 para 0,65** e o gap salta para **0,15**. Sem limite, a árvore cresce até isolar cada cliente em sua própria folha. Ela **decora** o treino."

**KNN:**
> "O padrão é o **inverso**: o gap é maior em k=3 e **encolhe** conforme k cresce. Com poucos vizinhos, o modelo copia o ruído local."

> "**A lição:** complexidade tem direções opostas nos dois algoritmos. Na árvore é profundidade alta; no KNN é k baixo. Combater overfitting aqui significa **podar a árvore e alargar a vizinhança**. Escolhi `max_depth=7` e `k=9`."

---

## 4:30 – 6:10 — Matriz de confusão e o achado contraintuitivo `[TÓPICO 5]`

Mostrar as duas matrizes lado a lado.

> "Falso Positivo é negar crédito a quem pagaria — o banco perde a **margem**. Falso Negativo é emprestar a quem não paga — o banco perde o **capital**. O FN é o erro caro."

**Aqui vem a virada — mostrar a tabela de custo:**

> "E é aqui que o projeto fica interessante. O **KNN tem melhor recall** — 0,78 contra 0,74 — e captura **66 inadimplentes a mais** que a árvore. Se eu decidisse pelo recall, o KNN venceria.
>
> Mas olhem o outro lado da matriz: para conseguir isso, o KNN **barra 821 bons pagadores, contra 229 da árvore**. São 592 clientes rentáveis rejeitados a mais.
>
> Modelei o custo em cima do **valor real de cada contrato** — não usei R$ 500 e R$ 5.000 fixos, porque errar num empréstimo de R$ 35 mil não custa o mesmo que errar num de R$ 1 mil. Falso Negativo custa `valor × 65%` de perda; Falso Positivo custa a margem de juros perdida.
>
> Resultado: **o KNN custa mais de R$ 1 milhão a mais que a árvore.**"

| | KNN | **Árvore** |
|---|---|---|
| FN (calotes) | 308 | 374 |
| FP (bons barrados) | 821 | **229** |
| **Custo** | R$ 3,85 mi | **R$ 2,82 mi** |

> "**Nenhuma métrica isolada decide um caso de negócio.** O recall diz que o KNN é melhor. A conta bancária diz o contrário."

---

## 6:10 – 7:00 — Veredito

> "Coloco em produção a **Árvore de Decisão com `max_depth=7`**.
>
> **Tecnicamente:** vence em F1, precisão, PR-AUC, e tem gap de generalização de **0,001** — não há overfitting.
>
> **Financeiramente:** sem modelo, o prejuízo é de R$ 9,9 milhões. Com a árvore, R$ 2,8 milhões. Ela **evita 71,6% do prejuízo** — e o veredito é **robusto**: variando a premissa de perda de 35% a 95%, a árvore vence em toda a faixa.
>
> **Duas recomendações:** ajustar o threshold de 0,50 para 0,77 economiza mais R$ 107 mil sem trocar uma linha do modelo. E manter a árvore também **por governança** — ela explica cada negativa em linguagem natural ('grade E, 45% da renda comprometida'), e em crédito a explicabilidade é exigência regulatória, não luxo.
>
> **Limitação honesta:** não há validação temporal. O risco de crédito muda com o ciclo econômico, e um split aleatório superestima a performance futura. Antes de produção, isso e uma auditoria de viés seriam obrigatórios."

---

## Checklist antes de gravar

- [ ] Rosto visível + tela compartilhada
- [ ] Notebook **já executado** (nada de esperar célula rodar no vídeo)
- [ ] Os 5 tópicos respondidos explicitamente
- [ ] Menos de 7 minutos
- [ ] Vídeo em modo de visualização **público / leitor**
- [ ] Links (GitHub + vídeo) postados no AVA — **prazo: 20/07/2026, 22h**
