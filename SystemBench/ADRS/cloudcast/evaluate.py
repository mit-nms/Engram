from logging import BASIC_FORMAT
from utils import *
from simulator import *
from networkx.algorithms.tree.branchings import Edmonds
from broadcast import BroadCastTopology
from pathlib import Path
import graphviz as gv
import networkx as nx
import subprocess
import argparse
import json
import sys
import os
from baselines import *

def networkx_to_graphviz(g, src, dsts, label="partitions"):
    """Convert `networkx` graph `g` to `graphviz.Digraph`.

    @type g: `networkx.Graph` or `networkx.DiGraph`
    @rtype: `graphviz.Digraph`
    """
    if g.is_directed():
        h = gv.Digraph()
    else:
        h = gv.Graph()
    for u, d in g.nodes(data=True):
        # u = u.split(",")[0]
        if u.split(",")[0] == src:
            h.node(str(u.replace(":", " ")), fillcolor="red", style="filled")
        elif u.split(",")[0] in dsts:
            h.node(str(u.replace(":", " ")), fillcolor="green", style="filled")
        h.node(str(u.replace(":", " ")))
    for u, v, d in g.edges(data=True):
        # print('edge', u, v, d)
        h.edge(str(u.replace(":", " ")), str(v.replace(":", " ")), label=str(d[label]))
    return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonfile", help="input json file")
    parser.add_argument("-a", "--algo", type=str, nargs="?", const="")
    parser.add_argument("-n", "--num-vms", type=int, nargs="?", const="")
    args = vars(parser.parse_args())
    print("Args:", args)

    print(f"\n==============> Baseline generation")
    with open(args["jsonfile"], "r") as f:
        config_name = args["jsonfile"].split("/")[1].split(".")[0]
        config = json.loads(f.read())

    # generate default graph with node and edge info
    # G = make_nx_graph(throughput_path="profiles/aws_throughput_11_8.csv")
    G = make_nx_graph(num_vms=int(args["num_vms"]))

    # src, dst
    source_node = config["source_node"]
    terminal_nodes = config["dest_nodes"]

    print(f"source_v = '{source_node}'")
    print(f"dest_v = {terminal_nodes}")
    # baseline path generations
    if args["algo"] is None:
        algorithms = [
            "Ndirect",
            "MDST",
            # "HST",
        ]
    else:
        algorithms = [args["algo"]]
    print(f"Algorithms: {algorithms}\n")

    directory = f"paths/{config_name}"
    if not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    num_partitions = config["num_partitions"]
    for algo in algorithms:
        outf = f"{directory}/{algo}.json"
        print(f"Generate {algo} paths into {outf}")
        if algo == "Ndirect":
            bc_t = N_direct(source_node, terminal_nodes, G, num_partitions)
        elif algo == "MDST":
            bc_t, mdgraph = MDST(source_node, terminal_nodes, G, num_partitions)
        elif algo == "MULTI-MDST":
            bc_t = MULTI_MDST(source_node, terminal_nodes, G, num_partitions)
        elif algo == "HST":
            bc_t = Min_Steiner_Tree(source_node, terminal_nodes, G, num_partitions)
        elif algo == "Ndijkstra":
            bc_t = N_dijkstra(source_node, terminal_nodes, G, num_partitions)
        else:
            raise NotImplementedError(algo)

        bc_t.set_num_partitions(config["num_partitions"])  # simple baseline, don't care about partitions, simply set it

        with open(outf, "w") as outfile:
            outfile.write(
                json.dumps(
                    {
                        "algo": algo,
                        "source_node": bc_t.src,
                        "terminal_nodes": bc_t.dsts,
                        "num_partitions": bc_t.num_partitions,
                        "generated_path": bc_t.paths,
                    }
                )
            )

    # put the evaluate logic here
    input_dir = "paths"  # input paths
    output_dir = "evals"  # eval results
    with open(sys.argv[1], "r") as f:
        config_name = sys.argv[1].split("/")[1].split(".")[0]
        config = json.loads(f.read())

    input_dir += f"/{config_name}"
    output_dir += f"/{config_name}"
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    simulator = BCSimulator(int(args["num_vms"]), output_dir)
    for algo in algorithms:
        path = f"{input_dir}/{algo}.json"
        simulator.evaluate_path(path, config)  # path of algorithm output, basic config to evaluate

    # nx.draw(mdgraph, with_labels=True)
    # plt.show()
    # h = networkx_to_graphviz(mdgraph, source_node, terminal_nodes)
    # h.render(filename="Ndirect")
