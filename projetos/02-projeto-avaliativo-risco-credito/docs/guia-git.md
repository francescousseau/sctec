# Guia de Git — branches e commits

> A rubrica avalia **versionamento** (critério 2): *"código estruturado via branches com commits incrementais frequentes e mensagens claras. **Não trabalhou direto na main**"*.
>
> ⚠️ **A data de entrega é o timestamp do último commit no GitHub.** Prazo: **20/07/2026, 22h**. Commits após isso invalidam a entrega.

---

## Fluxo padrão

```bash
# sempre partir da main atualizada
git switch main
git pull origin main

# criar a branch da etapa
git switch -c fase/eda
```

Trabalhar, e então:

```bash
git status
git add caminho/do/arquivo
git commit -m "feat: adiciona auditoria de regras de negócio"
git push -u origin fase/eda
```

Abrir o Pull Request no GitHub, fazer o merge, e limpar:

```bash
git switch main
git pull origin main
git branch -d fase/eda
```

---

## Convenção de commits semânticos

```text
<tipo>: <descrição objetiva, imperativa, minúscula>
```

| Tipo | Quando usar |
|---|---|
| `feat` | nova funcionalidade ou análise |
| `fix` | correção de bug ou erro metodológico |
| `docs` | documentação (README, docs/) |
| `refactor` | reorganização sem mudar comportamento |
| `chore` | estrutura, ambiente, dependências |
| `data` | adição ou atualização de base |
| `test` | validações e provas |

---

## Roteiro de branches e commits

Cada branch corresponde a uma fase do enunciado. Os commits abaixo refletem o **pipeline que efetivamente foi construído** — não um plano genérico.

### `chore/estrutura-projeto`
```text
chore: cria estrutura de pastas do projeto
chore: adiciona .gitignore e requirements.txt
chore: configura ambiente e kernel Jupyter
```

### `data/base-credito`
```text
data: adiciona base original de risco de crédito
docs: documenta o dicionário de dados no README
```

### `fase/eda`
```text
feat: adiciona relatório de qualidade dos dados
feat: adiciona auditoria de regras de negócio
feat: plota distribuição da variável alvo
feat: plota distribuição de renda e evidencia assimetria
feat: adiciona mapa de correlação de Pearson
feat: adiciona boxplots para diagnóstico de outliers
feat: analisa inadimplência por loan_grade e moradia
docs: escreve o parágrafo analítico da EDA
```

### `fase/data-prep`
```text
feat: remove duplicatas antes do split para evitar vazamento
feat: trata idades impossiveis como erro de cadastro
feat: converte tempo de emprego inconsistente em ausente
docs: justifica mediana sobre media pela assimetria da renda
feat: implementa imputacao de juros pela mediana do loan_grade
docs: fundamenta decisao assimetrica de outliers (arvore x knn)
```

### `fase/feature-engineering`
```text
feat: cria a feature comprometimento_renda
fix: protege a divisao contra renda zero
feat: mede redundancia com loan_percent_income
feat: remove loan_percent_income da matriz de modelagem
```

### `fase/preprocessamento`
```text
refactor: extrai transformadores customizados para src/preprocessing
feat: adiciona RateByGradeImputer
feat: adiciona Winsorizer para clipping p99 no knn
feat: codifica loan_grade como ordinal em vez de one-hot
feat: encapsula o pipeline no imblearn para impedir vazamento
feat: adiciona split estratificado 80/20
```

### `fase/balanceamento`
```text
feat: compara SMOTE, RandomUnderSampler e class_weight
test: demonstra que o SMOTE quebra as colunas one-hot
fix: substitui SMOTE por RandomUnderSampler
docs: justifica a escolha do balanceador pelo recall
```

### `fase/escalonamento`
```text
feat: aplica StandardScaler exclusivamente no ramo do knn
test: prova invariancia de escala da arvore de decisao
```

### `fase/modelagem`
```text
feat: adiciona grid de k para o knn
feat: adiciona grid de max_depth para a arvore
fix: mede o gap no treino real em vez do balanceado
feat: seleciona hiperparametros por validacao cruzada
feat: plota as curvas de complexidade
docs: diagnostica o ponto de ruptura de cada modelo
```

### `fase/avaliacao-negocio`
```text
feat: gera classification report e matrizes de confusao
feat: adiciona PR-AUC como metrica de base desbalanceada
feat: modela o custo sobre o valor real de cada emprestimo
feat: adiciona curva de custo por threshold
feat: adiciona analise de sensibilidade ao LGD
feat: extrai importancia das variaveis da arvore
docs: escreve o veredito executivo
```

### `docs/entrega-final`
```text
docs: finaliza README com resumo executivo
docs: atualiza decisoes tecnicas
docs: adiciona roteiro do video
```

### `release/entrega-final`
```text
chore: consolida a entrega final
```

---

## Boas práticas

**Commits pequenos e frequentes.** A rubrica penaliza explicitamente o "commit massivo único". Um commit deve contar **uma decisão**, não um dia de trabalho.

**Mensagens no imperativo, descrevendo o efeito.**

- ❌ `feat: mudanças no notebook`
- ❌ `update`
- ✅ `fix: mede o gap no treino real em vez do balanceado`

**Nunca commitar direto na `main`.** Toda alteração passa por branch + PR — inclusive as suas, mesmo trabalhando sozinho. É isso que a rubrica está avaliando.

**Não versionar dados pesados nem o ambiente.** Já coberto pelo `.gitignore`:

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
data/processed/
```

---

## Checklist antes de entregar

- [ ] Repositório **público** (ou privado com o avaliador convidado)
- [ ] Histórico mostra **branches**, não só commits na `main`
- [ ] Notebook commitado **com as saídas executadas** (o avaliador precisa ver os gráficos sem rodar nada)
- [ ] `README.md` com dicionário de dados e resumo executivo
- [ ] Link do vídeo em modo **público / leitor**
- [ ] Os **dois links** (GitHub + vídeo) postados no AVA
- [ ] **Último commit antes de 20/07/2026, 22h**
