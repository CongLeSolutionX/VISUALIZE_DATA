## Draw a PyGraph or PyDiGraph object using graphviz
## Source: https://www.rustworkx.org/apiref/rustworkx.visualization.graphviz_draw.html#rustworkx.visualization.graphviz_draw

import rustworkx as rx
from rustworkx.visualization import graphviz_draw

def node_attr(node):
  if node == 0:
    return {'color': 'yellow', 'fillcolor': 'yellow', 'style': 'filled'}
  if node % 2:
    return {'color': 'blue', 'fillcolor': 'blue', 'style': 'filled'}
  else:
    return {'color': 'red', 'fillcolor': 'red', 'style': 'filled'}

graph = rx.generators.directed_star_graph(weights=list(range(32)))
graphviz_draw(graph, node_attr_fn=node_attr, method='sfdp')