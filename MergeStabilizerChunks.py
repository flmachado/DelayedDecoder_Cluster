"""
Merge chunk .npy files produced by ComputeStabilizerChunk.py into a single
LT_Erasure_decoder_All_Strats-compatible object.

Usage:
    python MergeStabilizerChunks.py <GraphDescription> <ChunkDir>

  - GraphDescription: key in GraphDatabase (e.g. "23_1_7")
  - ChunkDir: directory containing the chunk .npy files
              (defaults to the script directory if omitted)

The script globs for files matching <GraphDescription>_chunk_*.npy,
verifies that all loss patterns are covered, merges the strategies,
and saves the result as <GraphDescription>_StabilizerInformation.npy.

This output is compatible with the existing loading code:
    erasure_decoder = np.load(file, allow_pickle=True).item()
    input_strats = erasure_decoder.strategies_ordered
"""

import numpy as np
import sys
import os
import glob
import re

from ErasureDecoder import (
    LT_Erasure_decoder_All_Strats,
    stabilizer_generators,
    interchange_nodes,
    single_qubit_commute,
)
from GraphDatabase import GraphInformation
from CodeFunctions.graphs import graph_from_nodes_and_edges


def merge_chunks(graph_description, chunk_dir):
    """
    Load all chunk files for a given graph, merge strategies,
    and reconstruct a full decoder object.
    """
    pattern = os.path.join(chunk_dir, "%s_chunk_*.npy" % graph_description)
    chunk_files = sorted(glob.glob(pattern))

    if not chunk_files:
        print("No chunk files found matching: %s" % pattern)
        sys.exit(1)

    print("Found %d chunk files" % len(chunk_files))

    # Deduplicate strategies (same [pauli_X, pauli_Z] pair)
    seen = set()
    unique_strats = []
    for strat in all_strats:
        if key not in seen:
            seen.add(key)
            unique_strats.append(strat)
    print("Unique strategies after dedup: %d" % len(unique_strats))

    # Load all chunks and sort by idx_min
    chunks = []
    for f in chunk_files:
        chunk = np.load(f, allow_pickle=True).item()

        for strat in chunk["strats"]:
            key = (strat[0], strat[1])
            if key not in seen:
                seen.add(key)
                unique_strats.append(strat)
        
        del chunk["strats"]
        del chunk["strategies_ordered"]
        chunks.append(chunk)

        print("  Loaded %s: loss patterns [%d, %d), %d strategies" % (
            os.path.basename(f), chunk["idx_min"], chunk["idx_max"],
            len(chunk["strats"])))
        print("Number of unique strats: ", len(unique_strats))

    chunks.sort(key=lambda c: c["idx_min"])

    # Verify full coverage
    n_total = chunks[0]["n_loss_patterns_total"]
    n_qbts = chunks[0]["n_qbts"]
    distance = chunks[0]["distance"]
    in_qbt = chunks[0]["in_qbt"]

    covered = set()
    for chunk in chunks:
        assert chunk["n_loss_patterns_total"] == n_total, \
            "Inconsistent total loss patterns across chunks"
        assert chunk["n_qbts"] == n_qbts, \
            "Inconsistent n_qbts across chunks"
        assert chunk["distance"] == distance, \
            "Inconsistent distance across chunks"
        for i in range(chunk["idx_min"], chunk["idx_max"]):
            covered.add(i)

    missing = set(range(n_total)) - covered
    if missing:
        print("WARNING: %d loss patterns not covered by any chunk!" % len(missing))
        print("  Missing indices (first 20): %s" % sorted(missing)[:20])
    else:
        print("All %d loss patterns are covered." % n_total)

    
    # Build the full decoder object, matching LT_Erasure_decoder_All_Strats
    Graph = GraphInformation[graph_description]
    graph_edges = Graph["graph_edges"]
    last_node = Graph["last_node"]

    graph_nodes = list(range(n_qbts + 1))
    edges = interchange_nodes(last_node, graph_edges)
    gstate = graph_from_nodes_and_edges(graph_nodes, edges)

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

    decoder.H_X, decoder.H_Z = decoder.H_matrices()
    decoder.loss_patts = decoder.get_loss_patterns()
    decoder.flag_check = False
    decoder.strats = unique_strats
    decoder.strategies_ordered = decoder.order_strats()

    print("Strategies ordered: %d" % len(decoder.strategies_ordered))
    return decoder


if __name__ == "__main__":
    GraphDescription = sys.argv[1]
    ChunkDir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(sys.argv[0]))

    print("Merging chunks for: %s" % GraphDescription)
    print("Chunk directory: %s" % ChunkDir)

    decoder = merge_chunks(GraphDescription, ChunkDir)

    save_name = "%s_StabilizerInformation.npy" % GraphDescription
    save_path = os.path.join(ChunkDir, save_name)
    np.save(save_path, decoder)
    print("Saved merged decoder to: %s" % save_path)
