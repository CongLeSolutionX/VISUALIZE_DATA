import rustworkx as rx

graph = rx.PyGraph()

# Each time add node is called, it returns a new node index
a = graph.add_node("A")
b = graph.add_node("B")
c = graph.add_node("C")

# add_edges_from takes tuples of node indices and weights,
# and returns edge indices
graph.add_edges_from([(a, b, 1.5), (a, c, 5.0), (b, c, 2.5)])

# Returns the path A -> B -> C
result = rx.dijkstra_shortest_paths(graph, a, c, weight_fn=float)
print("The shortest path between A and C is:")
print(result)