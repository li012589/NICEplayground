import pygraphviz as pgv

g=pgv.AGraph(directed=True, rankdir='TD', compound=True)

g.add_node("self.z")
g.add_node("self.realData")
g.add_node("self.batchData")

g.add_node("v")
g.add_node("NiceNetworkOperator,ifMH = False")
g.add_edge("z","NiceNetworkOperator,ifMH = False")
g.add_edge("v","NiceNetworkOperator,ifMH = False")
g.add_node("self.v_")
g.add_node("self.z_")
g.add_edge("NiceNetworkOperator,ifMH = False","self.v_")
g.add_edge("NiceNetworkOperator,ifMH = False","self.z_")

g.add_node("v_")
g.add_node("NiceNetworkOperator0,ifMH = True")
g.add_node("z1")
g.add_node("v1")
g.add_edge("self.z","NiceNetworkOperator0,ifMH = True")
g.add_edge("v_","NiceNetworkOperator0,ifMH = True")
g.add_edge("NiceNetworkOperator0,ifMH = True","z1")
g.add_edge("NiceNetworkOperator0,ifMH = True","v1")

g.add_node("stop_gradient")
g.add_edge("z1","stop_gradient")
g.add_node("z1_")
g.add_edge("stop_gradient","z1_")

g.add_node("v1_")
g.add_node("NiceNetworkOperator1,ifMH = True")
g.add_edge("self.realData","NiceNetworkOperator1,ifMH = True",label="abc")
g.add_edge("v1_","NiceNetworkOperator1,ifMH = True")
g.add_edge("NiceNetworkOperator1,ifMH = True","z2")
g.add_edge("NiceNetworkOperator1,ifMH = True","v2")

g.add_node("v2_")
g.add_node("NiceNetworkOperator2,ifMH = True")
g.add_edge("v2_","NiceNetworkOperator2,ifMH = True")
g.add_edge("z1_","NiceNetworkOperator2,ifMH = True")
g.add_edge("NiceNetworkOperator2,ifMH = True","z3")
g.add_edge("NiceNetworkOperator2,ifMH = True","v3")

g.add_node("tf.concat")

g.layout('dot')
g.draw('file.png')