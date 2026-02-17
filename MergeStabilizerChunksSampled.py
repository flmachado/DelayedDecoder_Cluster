"""
Merge chunk .npy files with a sampling strategy that limits the total number
of strategies to at most the number of loss patterns.

Usage:
    python MergeStabilizerChunksSampled.py <GraphDescription> <ChunkDir> [--top X] [--seed S]

  - GraphDescription: key in GraphDatabase (e.g. "23_1_7")
  - ChunkDir: directory containing the chunk .npy files
  - --top X: for each loss pattern, sample from the X lowest-weight
             strategies (default: 10)
  - --seed S: random seed for reproducibility (default: 42)

Algorithm (memory-efficient, one chunk at a time):
  1. Load one chunk file at a time.
  2. Regenerate the loss patterns for that chunk's [idx_min, idx_max) range.
  3. For each loss pattern in that range, find which of the chunk's strategies
     have identity support covering the lost qubits, sort by measurement
     weight, and randomly pick one from the top X lightest.
  4. Discard the chunk data, keeping only the selected strategies.
  5. After all chunks: deduplicate, build decoder object.

This avoids loading all ~200GB of strategy data into memory at once.

Output: <GraphDescription>_StabilizerInformation_Sampled.npy
Compatible with:
    erasure_decoder = np.load(file, allow_pickle=True).item()
    input_strats = erasure_decoder.strategies_ordered
"""

import numpy as np
import sys
import os
import glob
import random
import time
import argparse
from itertools import combinations

from ErasureDecoder import (
    LT_Erasure_decoder_All_Strats,
    interchange_nodes,
    single_qubit_commute,
)
from GraphDatabase import GraphInformation
from CodeFunctions.graphs import graph_from_nodes_and_edges


def compute_meas_weight(stab1, stab2, n_qbts):
    """Compute measurement weight: number of non-identity positions (excluding input qubit)."""
    measurement = [stab1[qbt] if stab1[qbt] != 'I' else stab2[qbt]
                   for qbt in range(n_qbts + 1)]
    return n_qbts - measurement.count('I')


def identity_support(stab1, stab2, n_qbts):
    """
    Return the set of qubit indices (0-based, code qubits only) where both
    pauli_X and pauli_Z are identity. These are the qubits that can be lost.
    """
    support = set()
    for qbt_idx in range(1, n_qbts + 1):
        if stab1[qbt_idx] == "I" and stab2[qbt_idx] == "I":
            support.add(qbt_idx - 1)
    return support


def merge_chunks_sampled(graph_description, chunk_dir, top_x, seed):
    rng = random.Random(seed)

    pattern = os.path.join(chunk_dir, "%s_chunk_*.npy" % graph_description)
    chunk_files = sorted(glob.glob(pattern))

    if not chunk_files:
        print("No chunk files found matching: %s" % pattern)
        sys.exit(1)

    print("Found %d chunk files" % len(chunk_files))

    Graph = GraphInformation[graph_description]
    graph_edges = Graph["graph_edges"]
    last_node = Graph["last_node"]
    distance = Graph["distance"]
    n_qbts = last_node
    in_qbt = 0

    # Generate all loss pattern index combos once (lightweight, just tuples of ints)
    all_loss_idx_combos = list(combinations(range(n_qbts), distance))
    n_total = len(all_loss_idx_combos)
    print("Total loss patterns: %d" % n_total)

    # Selected strategies: keyed by (pauli_X, pauli_Z) to deduplicate
    selected_strats = {}  # (stab1, stab2) -> [stab1, stab2]

    covered_lp = set()
    n_no_valid = 0
    t_start = time.time()

    for kf, f in enumerate(chunk_files):
        t_chunk = time.time()
        try:
            chunk = np.load(f, allow_pickle=True).item()
        except Exception:
            print("  Cannot load file: %s" % f)
            continue

        idx_min = chunk["idx_min"]
        idx_max = chunk["idx_max"]
        chunk_strats = chunk["strats"]
        n_chunk_strats = len(chunk_strats)

        # Precompute identity support and meas_weight for each strategy in this chunk
        chunk_id_supports = []
        chunk_weights = []
        for strat in chunk_strats:
            chunk_id_supports.append(identity_support(strat[0], strat[1], n_qbts))
            chunk_weights.append(compute_meas_weight(strat[0], strat[1], n_qbts))

        # Process each loss pattern in this chunk's range
        n_selected_this_chunk = 0
        for lp_idx in range(idx_min, idx_max):
            lost_set = set(all_loss_idx_combos[lp_idx])
            covered_lp.add(lp_idx)

            # Find strategies valid for this loss pattern
            valid = []
            for s_idx in range(n_chunk_strats):
                if lost_set <= chunk_id_supports[s_idx]:
                    valid.append((chunk_weights[s_idx], s_idx))

            if not valid:
                n_no_valid += 1
                continue

            # Sort by weight, pick randomly from top X
            valid.sort(key=lambda x: x[0])
            candidates = valid[:top_x]
            _, chosen_idx = rng.choice(candidates)
            chosen = chunk_strats[chosen_idx]
            key = (chosen[0], chosen[1])
            if key not in selected_strats:
                selected_strats[key] = chosen
                n_selected_this_chunk += 1

        dt = time.time() - t_chunk
        print("  %d / %d -- %s: lp [%d, %d), %d strats in chunk, "
              "%d new selected (%.1fs)" % (
                  kf + 1, len(chunk_files), os.path.basename(f),
                  idx_min, idx_max, n_chunk_strats,
                  n_selected_this_chunk, dt))

        # Free chunk data
        del chunk, chunk_strats, chunk_id_supports, chunk_weights

    dt_total = time.time() - t_start
    print("\nProcessing complete in %.1fs" % dt_total)

    missing = set(range(n_total)) - covered_lp
    if missing:
        print("WARNING: %d loss patterns not covered by any chunk!" % len(missing))
        print("  Missing indices (first 20): %s" % sorted(missing)[:20])
    else:
        print("All %d loss patterns are covered." % n_total)

    print("Loss patterns with no valid strategy: %d" % n_no_valid)
    print("Unique strategies selected: %d" % len(selected_strats))

    # Build the decoder object
    final_strats = list(selected_strats.values())

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
    decoder.strats = final_strats
    decoder.strategies_ordered = decoder.order_strats()

    print("Final strategies_ordered: %d" % len(decoder.strategies_ordered))
    return decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge stabilizer chunks with per-loss-pattern sampling")
    parser.add_argument("GraphDescription", help="Key in GraphDatabase")
    parser.add_argument("ChunkDir", nargs="?", default=None,
                        help="Directory with chunk .npy files (default: script dir)")
    parser.add_argument("--top", type=int, default=10,
                        help="Sample from the top X lightest strategies per loss pattern")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    chunk_dir = args.ChunkDir or os.path.dirname(os.path.abspath(sys.argv[0]))

    print("Merging chunks (sampled) for: %s" % args.GraphDescription)
    print("Chunk directory: %s" % chunk_dir)
    print("Top X: %d, Seed: %d" % (args.top, args.seed))

    decoder = merge_chunks_sampled(args.GraphDescription, chunk_dir, args.top, args.seed)

    save_name = "%s_StabilizerInformation_Sampled.npy" % args.GraphDescription
    save_path = os.path.join(chunk_dir, save_name)
    np.save(save_path, decoder)
    print("Saved sampled decoder to: %s" % save_path)
