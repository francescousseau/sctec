import csv
import re
from datetime import datetime


def limpar_categoria(categoria):
    """
    Limpa e padroniza o nome da categoria do produto.

    Regras:
    - Se estiver vazia, retorna 'sem categoria';
    - Remove espaços excedentes;
    - Converte para letras minúsculas;
    - Remove caracteres especiais indevidos com regex.
    """
    if categoria is None or categoria.strip() == "":
        return "Sem Categoria", True  # ← capitalizado conforme enunciado

    categoria_limpa = categoria.strip().lower()
    categoria_limpa = re.sub(r"[^a-zA-Z0-9_à-ÿ\s-]", "", categoria_limpa)
    categoria_limpa = re.sub(r"\s+", " ", categoria_limpa)

    return categoria_limpa, False 

def converter_numero(valor):
    """
    Converte valores numéricos vindos do CSV.

    Caso o valor esteja vazio, inválido ou ausente, retorna None.
    """
    if valor is None or valor.strip() == "":
        return None

    try:
        return float(valor)
    except ValueError:
        return None


def tratar_dimensao(valor, valor_padrao=0):
    """
    Trata dimensões físicas ausentes.

    Escolha técnica:
    Para este mini-projeto, valores ausentes em dimensões físicas são
    substituídos por 0. Essa escolha preserva o registro para análise,
    evita a perda de linhas e permite identificar posteriormente quais
    produtos estavam com cadastro incompleto.

    Em um projeto real de Machine Learning, poderíamos testar outras
    estratégias, como média, mediana por categoria ou exclusão controlada.
    """
    numero = converter_numero(valor)

    if numero is None:
        return valor_padrao, True

    return numero, False


def converter_data_brasileira(data_original):
    """
    Converte uma data no formato:
    '2017-05-16 15:05:35'

    Para:
    '16/05/2017'

    Caso esteja vazia ou inválida, retorna string vazia.
    """
    if data_original is None or data_original.strip() == "":
        return ""

    try:
        data_convertida = datetime.strptime(data_original.strip(), "%Y-%m-%d %H:%M:%S")
        return data_convertida.strftime("%d/%m/%Y")
    except ValueError:
        return ""


def processar_produtos(caminho_entrada, caminho_saida):
    """
    Lê o arquivo de produtos, trata categorias e dimensões físicas,
    e salva um novo CSV tratado.
    """
    total_linhas = 0
    categorias_corrigidas = 0
    dimensoes_corrigidas = 0

    campos_dimensoes = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    with open(caminho_entrada, mode="r", encoding="utf-8") as arquivo_entrada:
        leitor = csv.DictReader(arquivo_entrada)
        nomes_colunas = leitor.fieldnames

        with open(caminho_saida, mode="w", encoding="utf-8", newline="") as arquivo_saida:
            escritor = csv.DictWriter(arquivo_saida, fieldnames=nomes_colunas)
            escritor.writeheader()

            for linha in leitor:
                total_linhas += 1

                categoria_limpa, foi_corrigida = limpar_categoria(
                    linha.get("product_category_name")
                )
                linha["product_category_name"] = categoria_limpa

                if foi_corrigida:
                    categorias_corrigidas += 1

                for campo in campos_dimensoes:
                    valor_tratado, dimensao_corrigida = tratar_dimensao(linha.get(campo))
                    linha[campo] = valor_tratado

                    if dimensao_corrigida:
                        dimensoes_corrigidas += 1

                escritor.writerow(linha)

    return {
        "total_produtos_processados": total_linhas,
        "categorias_corrigidas": categorias_corrigidas,
        "dimensoes_corrigidas": dimensoes_corrigidas,
    }


def processar_pedidos(caminho_entrada, caminho_saida):
    """
    Lê o arquivo de pedidos, valida datas de entrega ausentes,
    identifica pedidos cancelados e converte a data de aprovação.
    """
    total_linhas = 0
    total_cancelados = 0
    entregas_ausentes = 0
    entregas_ausentes_canceladas = 0
    entregas_ausentes_nao_canceladas = 0
    datas_aprovacao_convertidas = 0

    with open(caminho_entrada, mode="r", encoding="utf-8") as arquivo_entrada:
        leitor = csv.DictReader(arquivo_entrada)

        nomes_colunas = list(leitor.fieldnames)

        nova_coluna = "order_approved_at_br"
        if nova_coluna not in nomes_colunas:
            nomes_colunas.append(nova_coluna)

        with open(caminho_saida, mode="w", encoding="utf-8", newline="") as arquivo_saida:
            escritor = csv.DictWriter(arquivo_saida, fieldnames=nomes_colunas)
            escritor.writeheader()

            for linha in leitor:
                total_linhas += 1

                status = linha.get("order_status", "").strip().lower()
                data_entrega = linha.get("order_delivered_customer_date", "").strip()

                if status == "canceled":
                    total_cancelados += 1

                if data_entrega == "":
                    entregas_ausentes += 1

                    if status == "canceled":
                        entregas_ausentes_canceladas += 1
                    else:
                        entregas_ausentes_nao_canceladas += 1

                data_convertida = converter_data_brasileira(linha.get("order_approved_at"))
                linha[nova_coluna] = data_convertida

                if data_convertida != "":
                    datas_aprovacao_convertidas += 1

                escritor.writerow(linha)

    hipotese_confirmada = entregas_ausentes_nao_canceladas == 0

    return {
        "total_pedidos_processados": total_linhas,
        "total_cancelados": total_cancelados,
        "entregas_ausentes": entregas_ausentes,
        "entregas_ausentes_canceladas": entregas_ausentes_canceladas,
        "entregas_ausentes_nao_canceladas": entregas_ausentes_nao_canceladas,
        "datas_aprovacao_convertidas": datas_aprovacao_convertidas,
        "hipotese_confirmada": hipotese_confirmada,
    }


def exibir_relatorio(relatorio_produtos, relatorio_pedidos):
    """
    Exibe no terminal um relatório estatístico manual do processamento.
    """
    print("=" * 70)
    print("RELATÓRIO FINAL DO PIPELINE DE SANITIZAÇÃO")
    print("=" * 70)

    print("\nPRODUTOS")
    print("-" * 70)
    print(f"Total de produtos processados: {relatorio_produtos['total_produtos_processados']}")
    print(f"Categorias corrigidas: {relatorio_produtos['categorias_corrigidas']}")
    print(f"Dimensões físicas corrigidas: {relatorio_produtos['dimensoes_corrigidas']}")

    print("\nPEDIDOS")
    print("-" * 70)
    print(f"Total de pedidos processados: {relatorio_pedidos['total_pedidos_processados']}")
    print(f"Total de pedidos cancelados: {relatorio_pedidos['total_cancelados']}")
    print(f"Entregas com data ausente: {relatorio_pedidos['entregas_ausentes']}")
    print(
        "Entregas ausentes com status canceled: "
        f"{relatorio_pedidos['entregas_ausentes_canceladas']}"
    )
    print(
        "Entregas ausentes com outros status: "
        f"{relatorio_pedidos['entregas_ausentes_nao_canceladas']}"
    )
    print(
        "Datas de aprovação convertidas: "
        f"{relatorio_pedidos['datas_aprovacao_convertidas']}"
    )

    print("\nVALIDAÇÃO DA HIPÓTESE DE NEGÓCIO")
    print("-" * 70)

    if relatorio_pedidos["hipotese_confirmada"]:
        print(
            "Hipótese confirmada: todas as datas de entrega ausentes "
            "pertencem a pedidos cancelados."
        )
    else:
        print(
            "Hipótese não confirmada: existem pedidos com data de entrega "
            "ausente que não estão com status canceled."
        )

    print("=" * 70)