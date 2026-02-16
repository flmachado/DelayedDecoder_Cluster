#!/usr/bin/python3
"""
Submit SLURM array jobs for parallelized stabilizer computation.

Usage:
    python submit_stabilizer.py <GraphDescription> <ChunkSize> [--dry-run]

Arguments:
    GraphDescription : Key in GraphDatabase (e.g. "23_1_7")
    ChunkSize        : Number of loss patterns per job
    --dry-run        : Print the batch script without submitting

This computes C(n_qbts, distance) total loss patterns, splits them into
chunks of ChunkSize, and submits a SLURM array job where each task
processes one chunk via ComputeStabilizerChunk.py.

After all jobs finish, run:
    python MergeStabilizerChunks.py <GraphDescription> <OutputDir>
to combine the results.
"""

import sys
import os
import math
from subprocess import check_output
from GraphDatabase import GraphInformation


def binom(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python submit_stabilizer.py <GraphDescription> <ChunkSize> [--dry-run]")
        sys.exit(1)

    graph_desc = sys.argv[1]
    chunk_size = int(sys.argv[2])
    dry_run = "--dry-run" in sys.argv

    if graph_desc not in GraphInformation:
        print("Unknown graph: %s" % graph_desc)
        print("Available: %s" % list(GraphInformation.keys()))
        sys.exit(1)

    Graph = GraphInformation[graph_desc]
    n_qbts = Graph["last_node"]
    distance = Graph["distance"]
    n_total = binom(n_qbts, distance)
    n_jobs = math.ceil(n_total / chunk_size)

    print("Graph: %s" % graph_desc)
    print("n_qbts=%d, distance=%d" % (n_qbts, distance))
    print("Total loss patterns: %d" % n_total)
    print("Chunk size: %d" % chunk_size)
    print("Number of jobs: %d" % n_jobs)

    # Directory setup
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    output_dir = os.path.join(script_dir, "stabilizer_chunks_%s" % graph_desc)

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        print("Output directory: %s" % output_dir)

    # Build SLURM batch script
    batch_script = """#!/bin/bash
#SBATCH -J stab_{graph_desc}
#SBATCH -p sapphire
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 3-0:00:00
#SBATCH --mem-per-cpu 4G
#SBATCH --account yao_lab
#SBATCH --requeue
#SBATCH -o {output_dir}/%a_stabilizer.out
#SBATCH -e {output_dir}/%a_stabilizer.err

echo "Job Id: $SLURM_JOB_ID"
echo "Array Task Id: $SLURM_ARRAY_TASK_ID"

IDX_MIN=$((SLURM_ARRAY_TASK_ID * {chunk_size}))
IDX_MAX=$(( (SLURM_ARRAY_TASK_ID + 1) * {chunk_size}))

# Cap IDX_MAX to total number of loss patterns
if [ $IDX_MAX -gt {n_total} ]; then
    IDX_MAX={n_total}
fi

echo "Processing loss patterns [$IDX_MIN, $IDX_MAX)"

cd {script_dir}
source /n/home03/fmachado/QuantumNetworks/DelayedDecoder_Cluster/venv/bin/activate
python {script_dir}/ComputeStabilizerChunk.py {graph_desc} $IDX_MIN $IDX_MAX
""".format(
        graph_desc=graph_desc,
        output_dir=output_dir,
        chunk_size=chunk_size,
        n_total=n_total,
        script_dir=script_dir,
    )

    if dry_run:
        print("\n--- Batch Script ---")
        print(batch_script)
        print("Would submit: sbatch --array=0-%d" % (n_jobs - 1))
    else:
        batch_file = os.path.join(output_dir, "submit_stabilizer.batch")
        with open(batch_file, "w") as f:
            f.write(batch_script)
        print("Batch script written to: %s" % batch_file)

        result = check_output(
            ["sbatch", "--array=0-%d" % (n_jobs - 1)],
            input=batch_script,
            universal_newlines=True,
        )
        print(result.strip(), "(%d jobs)" % n_jobs)

        print("\nAfter all jobs complete, merge with:")
        print("  python MergeStabilizerChunks.py %s %s" % (graph_desc, script_dir))
