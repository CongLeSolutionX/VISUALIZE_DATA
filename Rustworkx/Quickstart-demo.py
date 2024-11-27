import rustworkx as rx
from typing import Tuple, List, Optional, Union

def create_and_populate_graph() -> rx.PyGraph:
    """
    Create and populate a graph with nodes and edges.
    
    Returns:
        rx.PyGraph: The populated graph.
    """
    graph = rx.PyGraph()

    # Adding nodes with descriptive variable names
    NODE_A: int = graph.add_node("A")
    NODE_B: int = graph.add_node("B")
    NODE_C: int = graph.add_node("C")

    # Tuple type hint for clarity
    EDGE_LIST: List[Tuple[int, int, float]] = [(NODE_A, NODE_B, 1.5), (NODE_A, NODE_C, 5.0), (NODE_B, NODE_C, 2.5)]
    # Adding edges with descriptive variable names
    graph.add_edges_from(EDGE_LIST)
    
    return graph

def examine_graph(graph: rx.PyGraph) -> None:
    """
    Examine and print the elements of a graph.
    
    Args:
        graph (rx.PyGraph): The graph to examine.
    """
    NODE_INDICES: List[int] = list(graph.node_indices())
    EDGE_INDICES: List[int] = list(graph.edge_indices())
    print(f"Node Indices: {NODE_INDICES}")
    print(f"Edge Indices: {EDGE_INDICES}")

    if NODE_INDICES:
        FIRST_NODE_INDEX: int = NODE_INDICES[0]
        print(f"Data for first node index ({FIRST_NODE_INDEX}): {graph[FIRST_NODE_INDEX]}")

    if EDGE_INDICES:
        FIRST_EDGE_INDEX: int = EDGE_INDICES[0]
        FIRST_EDGE_ENDPOINTS: Tuple[int, int] = graph.get_edge_endpoints_by_index(FIRST_EDGE_INDEX)
        FIRST_EDGE_DATA: float = graph.get_edge_data_by_index(FIRST_EDGE_INDEX)

        print(f"Endpoints of first edge ({FIRST_EDGE_INDEX}): {FIRST_EDGE_ENDPOINTS}")
        print(f"Data for first edge ({FIRST_EDGE_INDEX}): {FIRST_EDGE_DATA}")

def find_and_display_shortest_path(graph: rx.PyGraph, start_node: int, end_node: int) -> None:
    """
    Find and display the shortest path between two nodes in a graph.
    
    Args:
        graph (rx.PyGraph): The graph to search in.
        start_node (int): The index of the starting node.
        end_node (int): The index of the ending node.
    """

    PATH_RESULT: rx.PathMapping = rx.dijkstra_shortest_paths(graph, start_node, end_node, float)
    print(f"The shortest path from node A to node C is: {PATH_RESULT}")
    return

def display_node_and_edge_data(graph: rx.PyGraph) -> None:
    """
    Display node and edge data payloads.
    
    Args:
        graph (rx.PyGraph): The graph to display data from.
    """
    print("Node data payloads")
    for node_index, node_data in zip(graph.node_indices(), graph.nodes()):
        print(f"  Node {node_index}: {node_data}")
    
    print("Edge data payloads")
    for edge_index, edge_data in zip(graph.edge_indices(), graph.edges()):
        endpoints = graph.get_edge_endpoints_by_index(edge_index)
        print(f"  Edge {edge_index} (from Node {endpoints[0]} to Node {endpoints[1]}): {edge_data}")

def demonstrate_node_removal_and_reuse() -> None:
    """Demonstrates node removal and index reuse in a graph."""
    GRAPH_FOR_REMOVAL_DEMO: rx.PyGraph = rx.PyGraph()
    GRAPH_FOR_REMOVAL_DEMO.add_nodes_from(list(range(5)))
    GRAPH_FOR_REMOVAL_DEMO.add_nodes_from(list(range(2)))
    
    print("Initial Nodes for Removal Demo:", list(GRAPH_FOR_REMOVAL_DEMO.node_indices()))

    GRAPH_FOR_REMOVAL_DEMO.remove_node(0)
    print("Nodes after removing node 0:", list(GRAPH_FOR_REMOVAL_DEMO.node_indices()))

    GRAPH_FOR_REMOVAL_DEMO.remove_node(3)
    print("Nodes after removing node 3:", list(GRAPH_FOR_REMOVAL_DEMO.node_indices()))

    GRAPH_FOR_REMOVAL_DEMO.add_node(100)  # This will reuse the lowest available index (0 or 3) when calling .add_node() directly
    print("Nodes after adding a new node (reusing index):", list(GRAPH_FOR_REMOVAL_DEMO.node_indices()))

    # Demonstrating reuse when adding multiple nodes with add_nodes_from
    GRAPH_FOR_REMOVAL_DEMO.add_nodes_from(["new_node_1", "new_node_2"])
    print("Nodes after adding multiple new nodes:", list(GRAPH_FOR_REMOVAL_DEMO.node_indices()))
    print("Node data after reuse:", list(GRAPH_FOR_REMOVAL_DEMO.nodes()))


if __name__ == "__main__":
    graph: rx.PyGraph = create_and_populate_graph()
    examine_graph(graph)

    node_a: int = 0  # Assuming 'A' is at index 0
    node_c: int = 2  # Assuming 'C' is at index 2
    find_and_display_shortest_path(graph, node_a, node_c)

    display_node_and_edge_data(graph)

    demonstrate_node_removal_and_reuse()