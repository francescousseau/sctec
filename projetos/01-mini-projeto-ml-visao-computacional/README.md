# Mini-Projeto — Machine Learning e Visão Computacional

> **Trilha:** Machine Learning e Visão Computacional — SCTEC/SENAI  
> **Módulo:** 1 — Semana 07  
> **Entrega:** Mini-Projeto Avaliativo

---

## Descrição do Projeto

Este projeto foi desenvolvido como parte da trilha **Machine Learning e Visão Computacional — SCTEC/SENAI**.

O desafio simula um cenário real de Engenharia e Análise de Dados em uma empresa de e-commerce, utilizando dados da **Olist**. A equipe de Engenharia de Dados identificou inconsistências nos arquivos `olist_products_dataset.csv` e `olist_orders_dataset.csv`, que estavam prejudicando relatórios automatizados e poderiam comprometer futuros modelos de Machine Learning.

A solução proposta é um **pipeline de sanitização de dados** construído inteiramente com bibliotecas nativas do Python, sem o uso de Pandas — garantindo o domínio da lógica estruturada de programação desde a camada mais fundamental do tratamento de dados.

---

## Objetivo

Criar um script em Python capaz de:

- Ler arquivos CSV de forma nativa com `csv.DictReader`;
- Tratar valores ausentes em categorias e dimensões físicas de produtos;
- Padronizar nomes de categorias com `.lower()`, `.strip()` e Expressões Regulares;
- Validar regras de negócio sobre pedidos cancelados e datas de entrega ausentes;
- Converter datas do formato ISO para o padrão brasileiro (`dd/mm/aaaa`);
- Gerar um relatório estatístico manual ao final do processamento.

---

## Estrutura do Projeto

```
mini_projeto/
│
├── data/                               # Arquivos CSV de entrada (não versionados)
│   ├── olist_products_dataset.csv
│   └── olist_orders_dataset.csv
│
├── output/                             # Arquivos CSV tratados (gerados automaticamente)
│   ├── olist_products_dataset_tratado.csv
│   └── olist_orders_dataset_tratado.csv
│
├── main.py                             # Ponto de entrada do pipeline
├── funcoes.py                          # Funções auxiliares de tratamento
└── README.md                           # Documentação do projeto
```

---

## Guia de Execução

### Pré-requisitos

- Python 3.8 ou superior instalado
- Nenhuma dependência externa — apenas bibliotecas nativas da linguagem

### Passo a Passo

**1. Clone o repositório**

```bash
git clone https://github.com/<seu-usuario>/<seu-repositorio>.git
cd <seu-repositorio>
```

**2. Adicione os arquivos de dados**

Crie a pasta `data/` na raiz do projeto e coloque os dois arquivos CSV dentro dela:

```bash
mkdir data
# Mova ou copie os arquivos:
# olist_products_dataset.csv → data/
# olist_orders_dataset.csv   → data/
```

Os arquivos CSV podem ser obtidos no repositório oficial do desafio:  
🔗 https://github.com/fiesc-junior-prado/mine_projeto_bloco_1

**3. Execute o pipeline**

```bash
python main.py
```

**4. Verifique os resultados**

Os arquivos tratados serão gerados automaticamente na pasta `output/`:

```
output/olist_products_dataset_tratado.csv
output/olist_orders_dataset_tratado.csv
```

O relatório estatístico final será exibido diretamente no terminal.

---

## Funcionalidades Implementadas

### 1. Leitura de Arquivos CSV

Os arquivos são lidos com `csv.DictReader`, permitindo acessar os dados por nome de coluna de forma estruturada e sem dependências externas.

### 2. Tratamento de Categorias Ausentes

Quando a coluna `product_category_name` está vazia ou nula, o valor é preenchido com a string padrão:

```
Sem Categoria
```

### 3. Padronização de Strings com Regex

Todos os nomes de categorias passam pelo seguinte fluxo de normalização:

1. Remoção de espaços excedentes com `.strip()`
2. Conversão para letras minúsculas com `.lower()`
3. Remoção de caracteres especiais indevidos com `re.sub()`
4. Remoção de espaços múltiplos internos com `re.sub(r"\s+", " ", ...)`

### 4. Tratamento de Dimensões Físicas

Os campos `product_weight_g`, `product_length_cm`, `product_height_cm` e `product_width_cm` são verificados. Valores ausentes ou inválidos são substituídos por `0`.

**Justificativa técnica:** a substituição por `0` foi escolhida para preservar o registro no dataset e permitir identificação posterior de produtos com cadastro incompleto. Em um projeto real de ML, estratégias como imputação pela média ou mediana por categoria seriam avaliadas conforme o impacto nos modelos.

### 5. Validação de Regra de Negócio — Pedidos Cancelados

O pipeline verifica a hipótese da Olist:

> *"Todas as datas de entrega ausentes (`order_delivered_customer_date` vazia) pertencem a pedidos com status `canceled`?"*

A validação é feita com estruturas condicionais `if/else` e o resultado é exibido no relatório final.

### 6. Conversão de Datas

A coluna `order_approved_at` é convertida do formato ISO `"2017-05-16 15:05:35"` para o padrão brasileiro `"16/05/2017"` usando o módulo `datetime`. Uma nova coluna `order_approved_at_br` é adicionada ao arquivo de saída.

### 7. Relatório Estatístico Manual

Ao final do processamento, o script exibe no terminal um sumário com:

- Total de produtos processados
- Total de categorias corrigidas
- Total de dimensões físicas corrigidas
- Total de pedidos processados
- Total de pedidos cancelados
- Total de entregas com data ausente
- Validação da hipótese de negócio

---

## Tecnologias e Bibliotecas Utilizadas

| Biblioteca | Uso |
|---|---|
| `csv` | Leitura e escrita de arquivos CSV com `DictReader` e `DictWriter` |
| `re` | Expressões Regulares para limpeza de strings |
| `datetime` | Parsing e formatação de datas |
| `pathlib` | Manipulação de caminhos de arquivos de forma multiplataforma |

> ⚠️ **O uso da biblioteca `pandas` não foi utilizado nesta entrega**, conforme as regras do desafio.

---

## Reflexão Teórica — Machine Learning e Qualidade de Dados

A limpeza e sanitização de dados é a primeira linha de defesa contra dois dos problemas mais críticos em Machine Learning: o **Overfitting** e o **viés de modelo**. Quando um dataset contém valores ausentes preenchidos de forma arbitrária, categorias inconsistentes ou strings mal formatadas, o algoritmo de treinamento aprende padrões espúrios presentes nos erros — e não nos fenômenos reais do negócio. O resultado é um modelo que performa bem nos dados de treino, mas falha sistematicamente em produção: o clássico cenário de Overfitting por ruído induzido, ilustrado pelo princípio **Garbage In, Garbage Out**.

Além disso, dados categóricos mal padronizados — como tratar `"Sem Categoria"`, `"sem_categoria"` e `""` como três classes distintas — introduzem **viés sistemático** nas previsões, distorcendo métricas de avaliação e gerando modelos injustos ou imprecisos. O pipeline desenvolvido neste projeto garante que os dados injetados em futuros modelos de ML da Olist representem fielmente a realidade operacional: categorias uniformes, dimensões numéricas consistentes e datas devidamente formatadas. Essa fundação sólida é o que permite que algoritmos de classificação, regressão ou segmentação aprendam os padrões corretos — e não os artefatos da sujeira nos dados.

---

## Autor

Desenvolvido como entrega avaliativa do **Módulo 1 — Mini-Projeto**  
Trilha: Machine Learning e Visão Computacional — SCTEC/SENAI