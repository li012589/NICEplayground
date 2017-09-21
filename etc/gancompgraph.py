import pygraphviz as pgv

g=pgv.AGraph(directed=True, rankdir='TD', compound=True)

g.add_node("z")
g.add_node("realData")
g.add_node("batchData")

g.add_node("v")
