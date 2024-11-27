import rustworkx as rx
from typing import Tuple

graph = rx.PyGraph()

# Adding nodes with descriptive variable names
node_a = graph.add_node("A")
node_b = graph.add_node("B")
node_c = graph.add_node("C")

# Tuple type hint for clarity
edge_list: list[Tuple[int, int, float]] = [(node_a, node_b, 1.5), (node_a, node_c, 5.0), (node_b, node_c, 2.5)]
# Adding edges with more descriptive variable names
graph.add_edges_from(edge_list)

# Examining elements of a graph
node_indices = list(graph.node_indices())
edge_indices = list(graph.edge_indices())
print(f"Node Indices: {node_indices}")
print(f"Edge Indices: {edge_indices}")

print(f"Data for first node index ({node_indices[0]}): {graph[node_indices[0]]}")

# Accessing edge data using consistent descriptive variable names
first_edge_index: int = edge_indices[0]

# Getting edge endpoints and data by their index is more readable than the bracket access in this case.
first_edge_endpoints: Tuple[int, int] = graph.get_edge_endpoints_by_index(first_edge_index)
first_edge_data: float = graph.get_edge_data_by_index(first_edge_index)

print(f"Endpoints of first edge ({first_edge_index}): {first_edge_endpoints}")
print(f"Data for first edge ({first_edge_index}): {first_edge_data}")

# # Get the shortest path & distance using Dijkstra's algorithm
#rx.dijkstra_shortest_paths(graph, node_a, node_c, weight_fn=float)
path = rx.dijkstra_shortest_paths(graph, node_a, node_c, weight_fn=float)
print(path)
print(f"The shortest path from A to C is: {path}")