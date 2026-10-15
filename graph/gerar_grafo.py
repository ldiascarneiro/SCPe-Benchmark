from pathlib import Path

from graph.laudo_graph import laudo_graph


def gerar_grafo_do_agente_em_png(output_name: str = "grafo_chain.png") -> Path:
    """
    Gera o PNG da arquitetura real do projeto.
    Fluxo esperado (CSV-only, passagem única): planner_node -> writer_node -> evoluator_node -> save.
    """
    output_path = Path(__file__).resolve().parent / output_name
    laudo_graph.get_graph().draw_png(str(output_path))
    print("Fluxo (CSV-only, passagem única): planner_node -> writer_node -> evoluator_node -> save")
    print(f"✅ Grafo salvo em {output_path}")
    return output_path

if __name__ == "__main__":
    gerar_grafo_do_agente_em_png()
