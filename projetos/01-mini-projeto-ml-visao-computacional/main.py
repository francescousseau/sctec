from pathlib import Path

from funcoes import processar_produtos, processar_pedidos, exibir_relatorio


def main():
    """
    Função principal do pipeline.

    Responsável por definir os caminhos dos arquivos,
    executar o processamento e exibir o relatório final.
    """
    pasta_base = Path(__file__).parent
    pasta_data = pasta_base / "data"
    pasta_output = pasta_base / "output"

    pasta_output.mkdir(exist_ok=True)

    caminho_produtos_entrada = pasta_data / "olist_products_dataset.csv"
    caminho_pedidos_entrada = pasta_data / "olist_orders_dataset.csv"

    caminho_produtos_saida = pasta_output / "olist_products_dataset_tratado.csv"
    caminho_pedidos_saida = pasta_output / "olist_orders_dataset_tratado.csv"

    if not caminho_produtos_entrada.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_produtos_entrada}\n"
            "Coloque o arquivo olist_products_dataset.csv dentro da pasta data/."
        )

    if not caminho_pedidos_entrada.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_pedidos_entrada}\n"
            "Coloque o arquivo olist_orders_dataset.csv dentro da pasta data/."
        )

    relatorio_produtos = processar_produtos(
        caminho_entrada=caminho_produtos_entrada,
        caminho_saida=caminho_produtos_saida,
    )

    relatorio_pedidos = processar_pedidos(
        caminho_entrada=caminho_pedidos_entrada,
        caminho_saida=caminho_pedidos_saida,
    )

    exibir_relatorio(relatorio_produtos, relatorio_pedidos)


if __name__ == "__main__":
    main()