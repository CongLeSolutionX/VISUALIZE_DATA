#import rustworkx as rx

## Usecase: Store the index as an attribute on the object you add to the graph:
# class GraphNode:

#     def __init__(self, value):
#         self.value = value
#         self.index = None

# graph = rx.PyGraph()
# index = graph.add_node(GraphNode("A"))
# graph[index].index = index


##Usecase: Update the index references all at once after creation:
# class GraphNode:
#     def __init__(self, value):
#         self.index = None
#         self.value = value

#     def __str__(self):
#         return f"GraphNode: {self.value} @ index: {self.index}"

# class GraphEdge:
#     def __init__(self, value):
#         self.index = None
#         self.value = value

#     def __str__(self):
#         return f"EdgeNode: {self.value} @ index: {self.index}"

# graph = rx.PyGraph()
# graph.add_nodes_from([GraphNode(i) for i in range(5)])
# graph.add_edges_from([(i, i + 1, GraphEdge(i)) for i in range(4)])
# # Populate index attribute in GraphNode objects
# for index in graph.node_indices():
#     graph[index].index = index
# # Populate index attribute in GraphEdge objects
# for index, data in graph.edge_index_map().items():
#     data[2].index = index
# print("Nodes:")
# for node in graph.nodes():
#     print(node)
# print("Edges:")
# for edge in graph.edges():
#     print(edge)

"""
This module demonstrates the creation and manipulation of graphs using the rustworkx library.
It defines Node and Edge classes to represent graph elements and provides functionality to create
a graph with a specified number of nodes and edges, assigning unique indices to each element.
"""

import rustworkx as rx
from typing import Any, Tuple, List

class Node:
    """Represents a node in the graph.

    Attributes:
        index: The index of the node in the graph.
        value: The value associated with the node.
    """
    def __init__(self, value: Any):
        self.index: int | None = None
        self.value: Any = value

    def __str__(self) -> str:
        return f"::NODE::{self.index}: {self.value}"

class Edge:
    """Represents an edge in the graph.

    Attributes:
        index: The index of the edge in the graph.
        value: The value associated with the edge.
    """
    def __init__(self, value: Any):
        self.index: int | None = None
        self.value: Any = value

    def __str__(self) -> str:
        return f"::EDGE::{self.index}: {self.value}"

def create_graph(num_nodes: int, edge_list: List[Tuple[int, int, Any]]) -> rx.PyDiGraph:
    """Creates a graph with the specified number of nodes and edges.

    Args:
        num_nodes: The number of nodes in the graph.
        edge_list: A list of tuples, where each tuple represents an edge
                  and contains the source node index, the target node index, and the edge value.

    Returns:
        A PyDiGraph object representing the created graph.
    """
    graph = rx.PyDiGraph()
    node_indices = graph.add_node([Node(i) for i in range(num_nodes)])
    edge_indices = graph.add_edge([(node_indices[src], node_indices[tgt], Edge(f"Edge from {src} to {tgt}")) for src, tgt, value in edge_list])

    for i, node_index in enumerate(graph.node_indices()):
        graph[node_index].index = i

    for i, edge_index in enumerate(graph.edge_indices()):
        graph[edge_index].index = i

    return graph

def main():
    """Main function to create and print graph elements."""
    num_nodes = 5
    edge_list = [
        (0, 1, "Edge 1"),
        (1, 2, "Edge 2"),
        (2, 3, "Edge 3"),
        (3, 0, "Edge 4"),
    ]

    graph = create_graph(num_nodes, edge_list)

    print("--- Nodes ---")
    for node_index in graph.node_indices():
        print(graph[node_index])

    # print("\n--- Edges ---")
    # for edge_index in graph.edge_indices():
    #     print(graph[edge_index])

if __name__ == "__main__":
    main()