# Mini-Projeto — Machine Learning e Visão Computacional

> **Trilha:** Machine Learning e Visão Computacional — SCTEC/SENAI

> **Módulo:** 1 — Semana 07

> **Entrega:** Mini-Projeto Avaliativo

> **Autor:** Francesco Cousseau

---

## Descrição do Projeto

Este projeto foi desenvolvido como parte da trilha **Machine Learning e Visão Computacional — SCTEC/SENAI**.

O desafio simula um cenário real de Engenharia e Análise de Dados em uma empresa de e-commerce, utilizando dados da Olist. A equipe de Engenharia de Dados identificou inconsistências nos arquivos `olist_products_dataset.csv` e `olist_orders_dataset.csv`, que estavam prejudicando relatórios automatizados e poderiam comprometer futuros modelos de Machine Learning.

A solução proposta é um pipeline de sanitização de dados construído inteiramente com bibliotecas nativas do Python, sem o uso de Pandas, demonstrando domínio de lógica estruturada, manipulação de arquivos CSV, funções, condicionais, laços, expressões regulares e tratamento temporal.

---

## Objetivo

Criar um script em Python capaz de:

* ler arquivos CSV de forma nativa com `csv.DictReader`;
* tratar valores ausentes em categorias de produtos;
* tratar valores ausentes ou inválidos em dimensões físicas;
* padronizar nomes de categorias com `.lower()`, `.strip()` e Expressões Regulares;
* validar regras de negócio sobre pedidos cancelados e datas de entrega ausentes;
* converter datas do formato original para o padrão brasileiro;
* gerar um relatório estatístico manual ao final do processamento.

---

## Contexto do Problema

Empresas de e-commerce lidam diariamente com grandes volumes de dados transacionais. Antes que esses dados sejam utilizados em dashboards, relatórios automatizados ou modelos de Machine Learning, é necessário garantir que estejam limpos, consistentes e padronizados.

No cenário proposto, os arquivos da Olist apresentam inconsistências como:

* categorias de produtos ausentes;
* strings fora de padrão;
* dimensões físicas vazias;
* datas de entrega ausentes;
* necessidade de validação de pedidos cancelados;
* datas em formato inadequado para relatórios brasileiros.

Esse tipo de tratamento é uma etapa essencial em pipelines de dados e Machine Learning, pois dados inconsistentes podem comprometer análises, métricas e modelos preditivos.

---

## Estrutura do Projeto

```text
01-mini-projeto-ml-visao-computacional/
│
├── data/
│   ├── olist_products_dataset.csv
│   └── olist_orders_dataset.csv
│
├── output/
│   ├── olist_products_dataset_tratado.csv
│   └── olist_orders_dataset_tratado.csv
│
├── main.py
├── funcoes.py
├── .gitignore
└── README.md
```

---

## Guia de Execução

### Pré-requisitos

* Python 3.8 ou superior instalado
* Git instalado
* Nenhuma dependência externa

Este projeto utiliza apenas bibliotecas nativas da linguagem Python.

---

### 1. Clone o repositório

```bash
git clone https://github.com/francescousseau/sctec.git
```

Entre na pasta do mini-projeto:

```bash
cd sctec/projetos/01-mini-projeto-ml-visao-computacional
```

---

### 2. Verifique os arquivos de dados

Os arquivos CSV devem estar dentro da pasta `data/`:

```text
data/
├── olist_products_dataset.csv
└── olist_orders_dataset.csv
```

Caso os arquivos não estejam presentes, obtenha-os no repositório oficial do desafio:

```text
https://github.com/fiesc-junior-prado/mine_projeto_bloco_1
```

---

### 3. Execute o pipeline

```bash
python main.py
```

---

### 4. Verifique os resultados

Após a execução, os arquivos tratados serão gerados na pasta `output/`:

```text
output/
├── olist_products_dataset_tratado.csv
└── olist_orders_dataset_tratado.csv
```

O relatório estatístico final será exibido diretamente no terminal.

---

## Funcionalidades Implementadas

### 1. Leitura de Arquivos CSV

Os arquivos são lidos com `csv.DictReader`, permitindo acessar cada linha como um dicionário e manipular os dados pelo nome das colunas.

---

### 2. Tratamento de Categorias Ausentes

Quando a coluna `product_category_name` está vazia ou nula, o valor é preenchido com a string padrão:

```text
Sem Categoria
```

Essa regra evita categorias vazias e melhora a consistência da base para relatórios e análises futuras.

---

### 3. Padronização de Strings com Regex

Todos os nomes de categorias passam pelo seguinte fluxo de normalização:

1. remoção de espaços excedentes com `.strip()`;
2. conversão para letras minúsculas com `.lower()`;
3. remoção de caracteres especiais indevidos com `re.sub()`;
4. remoção de espaços múltiplos internos com `re.sub(r"\s+", " ", ...)`.

---

### 4. Tratamento de Dimensões Físicas

Os campos abaixo são verificados:

* `product_weight_g`
* `product_length_cm`
* `product_height_cm`
* `product_width_cm`

Valores ausentes ou inválidos são substituídos por `0`.

Essa escolha técnica preserva o registro no dataset e permite identificar posteriormente quais produtos possuíam cadastro incompleto. Em um projeto real de Machine Learning, outras estratégias poderiam ser avaliadas, como imputação por média, mediana por categoria ou exclusão controlada conforme o impacto analítico.

---

### 5. Validação de Regra de Negócio — Pedidos Cancelados

O pipeline verifica a seguinte hipótese de negócio:

> Todas as datas de entrega ausentes (`order_delivered_customer_date`) pertencem a pedidos com status `canceled`?

A validação é feita com estruturas condicionais `if`, `elif` e `else`, separando os registros conforme o status do pedido e a presença ou ausência da data de entrega.

O resultado é exibido no relatório final.

---

### 6. Conversão de Datas

A coluna `order_approved_at` é convertida do formato original:

```text
2017-05-16 15:05:35
```

Para o padrão brasileiro simplificado:

```text
16/05/2017
```

A conversão é feita com o módulo nativo `datetime`.

O script adiciona uma nova coluna ao arquivo tratado:

```text
order_approved_at_br
```

---

### 7. Relatório Estatístico Manual

Ao final do processamento, o script exibe no terminal um sumário com:

* total de produtos processados;
* total de categorias corrigidas;
* total de dimensões físicas corrigidas;
* total de pedidos processados;
* total de pedidos cancelados;
* total de entregas com data ausente;
* total de entregas ausentes em pedidos cancelados;
* total de entregas ausentes em pedidos não cancelados;
* total de datas de aprovação convertidas;
* validação final da hipótese de negócio.

---

## Tecnologias e Bibliotecas Utilizadas

| Biblioteca | Uso                                                               |
| ---------- | ----------------------------------------------------------------- |
| `csv`      | Leitura e escrita de arquivos CSV com `DictReader` e `DictWriter` |
| `re`       | Limpeza de strings com Expressões Regulares                       |
| `datetime` | Conversão e formatação de datas                                   |
| `pathlib`  | Manipulação de caminhos de arquivos de forma multiplataforma      |

> O uso da biblioteca `pandas` não foi utilizado nesta entrega, conforme as regras do desafio.

---

## Exemplo de Saída Esperada

```text
======================================================================
RELATÓRIO FINAL DO PIPELINE DE SANITIZAÇÃO
======================================================================

PRODUTOS
----------------------------------------------------------------------
Total de produtos processados: 32951
Categorias corrigidas: ...
Dimensões físicas corrigidas: ...

PEDIDOS
----------------------------------------------------------------------
Total de pedidos processados: 99441
Total de pedidos cancelados: ...
Entregas com data ausente: ...
Entregas ausentes com status canceled: ...
Entregas ausentes com outros status: ...
Datas de aprovação convertidas: ...

VALIDAÇÃO DA HIPÓTESE DE NEGÓCIO
----------------------------------------------------------------------
Hipótese não confirmada: existem pedidos com data de entrega ausente que não estão com status canceled.
======================================================================
```

---

## Reflexão Teórica — Machine Learning e Qualidade de Dados

A limpeza e sanitização de dados é uma etapa fundamental em projetos de Machine Learning. Quando um dataset contém valores ausentes, categorias inconsistentes, strings mal formatadas ou datas inválidas, o modelo pode aprender padrões artificiais presentes nos erros da base, e não nos fenômenos reais do negócio.

Esse problema está relacionado ao princípio **Garbage In, Garbage Out**: se dados ruins entram no pipeline, resultados ruins serão produzidos. Em modelos de Machine Learning, dados mal tratados podem gerar overfitting, underfitting, viés estatístico e baixa capacidade de generalização.

Além disso, dados categóricos mal padronizados podem distorcer a interpretação dos algoritmos. Por exemplo, tratar `"Sem Categoria"`, `"sem categoria"`, `"sem_categoria"` e valores vazios como categorias diferentes pode criar ruído e prejudicar análises futuras.

O pipeline desenvolvido neste projeto busca reduzir esses riscos ao padronizar categorias, tratar valores ausentes, validar regras de negócio e converter datas para um formato consistente. Com isso, os dados ficam mais confiáveis para etapas posteriores de análise exploratória, criação de dashboards, modelagem preditiva e tomada de decisão automatizada.

---

## Autor

**Francesco Cousseau**
AI & Data Science Consultant
Faithful Data Solutions

Projeto desenvolvido como entrega avaliativa do **Módulo 1 — Mini-Projeto**, na trilha **Machine Learning e Visão Computacional — SCTEC/SENAI**.
