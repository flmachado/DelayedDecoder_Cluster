import numpy as np
import math
from itertools import product
import time
from ErasureDecoder import *
from ParsingDataCode.ParseLargeGraphs import *
from GraphDatabase import GraphInformation



if __name__ == '__main__':

    from CodeFunctions.graphs import *
    from itertools import permutations

    import sys
    import os

    GraphDescription = sys.argv[1]
    
    # Destination of permutation files that Gefen uploaded, for instance: r"C:\Users\bdt697\Downloads\final_best_permutations_13_1_5_d.csv"    
    
    save_name = GraphDescription + "_StabilizerInformation.npy"
    Graph = GraphInformation[GraphDescription]
    graph_edges = Graph["graph_edges"]
    last_node = Graph["last_node"]
    distance = Graph["distance"]

    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
    
    filename = os.path.join( dirname, Graph["filename"])
    
    print("Loading file : ", filename)

    no_anti_com_flag = False
    n_qbts = last_node
    graph_nodes = list(range(n_qbts + 1))
    in_qubit = 0
    graph_edges = interchange_nodes(last_node, graph_edges)
    gstate = graph_from_nodes_and_edges(graph_nodes,
                                        graph_edges)

    erasure_decoder = LT_Erasure_decoder_All_Strats(n_qbts, distance, gstate, in_qbt=in_qubit)
    
    np.save(save_name, erasure_decoder)

    print("Saved stabilizer information to file : ", save_name)