## Draw a graph with Matplotlib.
## Source: https://www.rustworkx.org/apiref/rustworkx.visualization.mpl_draw.html#rustworkx.visualization.mpl_draw

import matplotlib.pyplot as plt

import rustworkx as rx
from rustworkx.visualization import mpl_draw

G = rx.generators.directed_path_graph(25)
mpl_draw(G)
plt.draw()

plt.show()