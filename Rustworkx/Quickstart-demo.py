import rustworkx as rx

graph = rx.PyGraph()

# Each time add node is called, it returns a new node index
a = graph.add_node("A")
b = graph.add_node("B")
c = graph.add_node("C")

# add_edges_from takes tuples of node indices and weights,
# and returns edge indices
graph.add_edges_from([(a, b, 1.5), (a, c, 5.0), (b, c, 2.5)])


# Examining elements of a graph
node_indices = graph.node_indices()
edge_indices = graph.edge_indices()
print(node_indices)
print(edge_indices)

print("The first index data on the graph is:")
first_index_data = graph[node_indices[0]]
print(first_index_data)

first_index_data = graph.get_edge_data_by_index(edge_indices[0])
first_index_edgepoints = graph.get_edge_endpoints_by_index(edge_indices[0])
print(first_index_edgepoints)
print(first_index_data)

# Returns the path A -> B -> C
result = rx.dijkstra_shortest_paths(graph, a, c, weight_fn=float)
print("The shortest path between A and C is:")
print(result)