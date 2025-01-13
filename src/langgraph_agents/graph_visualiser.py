from langgraph.graph import StateGraph
from graphviz import Digraph


def visualise_graph(graph: StateGraph, output_path: str = "graph"):
    """
    Visualise the StateGraph using Graphviz.
    Saves the visualisation to the specified path.
    """
    graphviz_repr = graph.get_graph()
    dot = Digraph(format='png')  # Create a Graphviz Digraph object

    # Populate the Graphviz object with nodes and edges
    for node_id, label in graphviz_repr['nodes'].items():
        dot.node(node_id, label=label)

    for edge in graphviz_repr['edges']:
        dot.edge(edge['source'], edge['target'])

    # Save the graph as an image
    dot.render(output_path, cleanup=True)
    print(f"Graph visualisation saved as {output_path}.png")
