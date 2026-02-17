"""
Compute strategies for a chunk of loss patterns and save to a .npy file.

Usage:
    python ComputeStabilizerChunk.py <GraphDescription> <IdxMin> <IdxMax>

This processes loss patterns in the range [IdxMin, IdxMax) and saves:
  - The strategies found for those patterns
  - Metadata needed for merging (n_qbts, distance, in_qbt, loss pattern indices)

Output file: <GraphDescription>_chunk_<IdxMin>_<IdxMax>.npy
"""

import numpy as np
import sys
import os
import copy
import time
from itertools import combinations

from ErasureDecoder import (
    LT_Erasure_decoder_All_Strats,
    stabilizer_generators,
    interchange_nodes,
    GaussElim_X,
    GaussElim_Z,
    get_full_stabilizer_group,
    single_qubit_commute,
)
from GraphDatabase import GraphInformation
from CodeFunctions.graphs import graph_from_nodes_and_edges


def compute_chunk(n_qbts, distance, gstate, in_qbt, idx_min, idx_max):
    """
    Set up the decoder infrastructure (H matrices, loss patterns) and
    process only loss patterns in range [idx_min, idx_max).

    Returns a dict with:
      - "strats": list of [pauli_X, pauli_Z] strategies
      - "strategies_ordered": ordered strategy list
      - "idx_min": start index
      - "idx_max": end index (exclusive)
      - "n_loss_patterns_total": total number of loss patterns
      - "n_qbts": number of qubits
      - "distance": code distance
      - "in_qbt": input qubit index
    """
    # Build a partial decoder: initialize everything except find_strategies
    # We reuse the class but override the heavy computation.
    decoder = LT_Erasure_decoder_All_Strats.__new__(LT_Erasure_decoder_All_Strats)
    decoder.gstate = gstate
    decoder.n_qbts = n_qbts
    decoder.distance = distance
    decoder.in_qbt = in_qbt
    decoder.n_rows = n_qbts + 1
    decoder.printing = False
    decoder.cnt = 0
    decoder.clashed_strats = []
    decoder.log_X_idx = -2
    decoder.log_Z_idx = -1
    decoder.numb_useful_loss_patts = 0

    # Compute H matrices (fast)
    decoder.H_X, decoder.H_Z = decoder.H_matrices()

    # Generate all loss patterns (fast)
    decoder.loss_patts = decoder.get_loss_patterns()
    n_total = len(decoder.loss_patts)

    # Clamp range
    idx_max = min(idx_max, n_total)
    idx_min = max(idx_min, 0)

    print("Total loss patterns: %d" % n_total)
    print("Processing range: [%d, %d)" % (idx_min, idx_max))

    # Process only the chunk
    strategies = []
    t1 = time.time()
    for idx_lp in range(idx_min, idx_max):
        loss_patt = decoder.loss_patts[idx_lp]
        if (idx_lp - idx_min) % 100 == 0:
            t2 = time.time()
            print("  Loss pattern %d / %d (elapsed: %.1fs)" % (
                idx_lp - idx_min, idx_max - idx_min, t2 - t1))

        flag, strats = decoder.run_specific_loss_pattern(loss_patt)
        if flag:
            for strat in strats:
                strategies.append(strat)
        else:
            print("  Failed at loss pattern %d: %s" % (idx_lp, loss_patt))

    t2 = time.time()
    print("Chunk complete. Found %d strategies in %.1fs" % (len(strategies), t2 - t1))

    # Order strategies (same logic as LT_Erasure_decoder_All_Strats.order_strats)
    decoder.strats = strategies
    strategies_ordered = decoder.order_strats()

    return {
        "strats": strategies,
        "strategies_ordered": strategies_ordered,
        "idx_min": idx_min,
        "idx_max": idx_max,
        "n_loss_patterns_total": n_total,
        "n_qbts": n_qbts,
        "distance": distance,
        "in_qbt": in_qbt,
    }


if __name__ == "__main__":
    GraphDescription = sys.argv[1]
    IdxMin = int(sys.argv[2])
    IdxMax = int(sys.argv[3])

    Graph = GraphInformation[GraphDescription]
    graph_edges = Graph["graph_edges"]
    last_node = Graph["last_node"]
    distance = Graph["distance"]

    n_qbts = last_node
    graph_nodes = list(range(n_qbts + 1))
    in_qubit = 0
    graph_edges = interchange_nodes(last_node, graph_edges)
    gstate = graph_from_nodes_and_edges(graph_nodes, graph_edges)

    dirname = os.getcwd()
    print("Will save to %s"%dirname)
    print("Graph: %s, n_qbts=%d, distance=%d" % (GraphDescription, n_qbts, distance))

    chunk_result = compute_chunk(n_qbts, distance, gstate, in_qubit, IdxMin, IdxMax)


    save_name = "%s_chunk_%d_%d.npy" % (GraphDescription, IdxMin, IdxMax)
    save_path = os.path.join(dirname, save_name)

    np.save(save_path, chunk_result)
    print("Saved chunk to: %s" % save_path)
