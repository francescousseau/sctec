# SCTEC — Projetos de Machine Learning, Visão Computacional e IA

Este repositório reúne projetos práticos desenvolvidos durante minha jornada na trilha **SCTEC/SENAI — Machine Learning e Visão Computacional**.

O objetivo é organizar, documentar e versionar projetos relacionados a **Ciência de Dados, Machine Learning, Visão Computacional, IA Generativa, ChatBots, Agentes de IA e Engenharia de IA**, construindo um portfólio técnico progressivo e profissional.

## Sobre a Trilha

A formação SCTEC/SENAI em **Machine Learning e Visão Computacional** aborda fundamentos e aplicações práticas de Inteligência Artificial, incluindo:

- Fundamentos de Machine Learning
- Tipos de aprendizado: supervisionado, não supervisionado e por reforço
- Ciclo de vida de projetos de ML
- Ingestão, limpeza e pré-processamento de dados
- Análise exploratória de dados
- Padronização, normalização e codificação de variáveis
- Modelos avançados de Machine Learning
- Random Forest, Gradient Boosting e Support Vector Machines
- Validação cruzada e ajuste de hiperparâmetros
- Visão Computacional com Python
- Redes neurais convolucionais
- YOLO e detecção de objetos
- Criação de APIs para integração de modelos
- Deploy de soluções em plataformas como Render e Hugging Face
- Introdução à IA Generativa
- ChatBots e Agentes de IA

## Objetivo do Repositório

Este repositório funciona como um **monorepo educacional e profissional**, concentrando projetos práticos em uma única estrutura organizada.

A proposta é demonstrar evolução técnica em:

- Programação em Python
- Manipulação de arquivos e dados
- Limpeza e tratamento de dados
- Regras de negócio aplicadas
- Machine Learning
- Visão Computacional
- IA Generativa
- Documentação técnica
- Versionamento com Git e GitHub
- Boas práticas de organização de projetos

## Projetos

| Nº | Projeto                                                                                               | Status    | Principais Tópicos                                                                                        |
| -: | ----------------------------------------------------------------------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| 01 | [Mini-Projeto — ML e Visão Computacional](./projetos/01-mini-projeto-ml-visao-computacional)          | Concluído | Python, CSV, Regex, Datetime, ETL e limpeza de dados                                                      |
| 02 | [Pipeline Preditivo para Análise de Risco de Crédito](./projetos/02-projeto-avaliativo-risco-credito) | Concluído | KNN, Árvore de Decisão, prevenção de data leakage, diagnóstico de overfitting e análise de custo de erros |
| 03 | Projeto 03                                                                                            | Planejado | A definir                                                                                                 |



### 🔍 Destaque — Projeto 02: Risco de Crédito

Pipeline completo de classificação binária comparando **KNN** e **Árvore de Decisão**,
com foco em três eixos: prevenção de vazamento de dados, diagnóstico analítico de
overfitting e **tradução dos erros do modelo em impacto financeiro**.

O achado central é contraintuitivo: o KNN tem **melhor recall** e captura mais
inadimplentes — mas rejeita 592 bons pagadores a mais e, no somatório,
custa **R$ 1 milhão a mais** que a árvore.

Nenhuma métrica isolada decide um caso de negócio.

## Estrutura do Repositório

```text
sctec/
├── README.md
├── LICENSE
├── .gitignore
├── projetos/
│   ├── 01-mini-projeto-ml-visao-computacional/
│   └── 02-projeto-avaliativo-risco-credito/
│       ├── README.md          # dicionário de dados + resumo executivo
│       ├── notebooks/         # pipeline completo, executado
│       ├── src/               # módulos: preprocessing, modeling, evaluation
│       ├── data/
│       ├── reports/           # figuras e tabelas de resultado
│       └── docs/              # decisões técnicas, guia de git
└── requirements.txt
