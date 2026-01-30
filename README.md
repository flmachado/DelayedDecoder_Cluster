# Delayed Measurement Decoder for Graph-Based Quantum Codes

A hybrid delayed measurement decoder for quantum error correction in graph-based quantum codes. This project implements algorithms to optimize quantum information recovery when qubits are lost during transmission or storage, finding measurement strategies that minimize the number of additional "matter qubits" needed for error correction.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Description](#algorithm-description)
- [Project Structure](#project-structure)
- [Data Formats](#data-formats)

## Overview

This decoder evaluates different measurement strategies to determine the optimal order of measurement operations for quantum information recovery. The key objectives are:

- Find decoding strategies that minimize matter qubits (additional quantum resources needed for error correction)
- Test various measurement orderings to find optimal patterns
- Support quantum teleportation and error correction through stabilizer codes
- Handle qubit loss/erasure scenarios with known graph structures

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setting Up the Virtual Environment

1. Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd DelayedDecoder_Cluster
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

4. Install the required dependencies:

```bash
pip install numpy networkx pandas matplotlib
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation and linear algebra |
| `networkx` | Graph representation and manipulation |
| `pandas` | CSV file reading and data manipulation |
| `matplotlib` | Visualization (optional, for plotting results) |

## Usage

### Command Line Interface

The main entry point is `RunningDecoderLaptop.py`:

```bash
python RunningDecoderLaptop.py <GraphDescription> <IdxMin> <IdxMax>
```

**Arguments:**
- `GraphDescription`: Identifier for the graph configuration (see available graphs below)
- `IdxMin`: Starting index for measurement patterns to test
- `IdxMax`: Ending index for measurement patterns to test

**Available Graph Configurations:**
- `"13_1_5_a"`, `"13_1_5_c"`, `"13_1_5_d"` - 13-qubit codes (distance 4)
- `"16_1_6_b"` - 16-qubit code (distance 5)
- `"24_1_7"` - 24-qubit code (distance 6)
- `"25_1_8"` - 25-qubit code (distance 7)

**Examples:**

```bash
# Run indices 0-100 for 13-qubit graph variant 'a'
python RunningDecoderLaptop.py "13_1_5_a" 0 100

# Run first 50 patterns for 16-qubit graph
python RunningDecoderLaptop.py "16_1_6_b" 0 50

# Run indices 100-200 for 24-qubit graph
python RunningDecoderLaptop.py "24_1_8" 100 200
```

### Python API Usage

#### Basic Example: Running the Decoder

```python
from CodeFunctions.graphs import graph_from_nodes_and_edges
from ErasureDecoder import LT_Erasure_decoder
from HybridDelayedMeasDecoderFixedMeasPatt import LT_FullHybridDecoderNew
import copy

# Define a simple graph (linear graph with 5 qubits)
n_qbts = 5
graph_nodes = list(range(n_qbts + 1))  # Nodes 0-5
graph_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
gstate = graph_from_nodes_and_edges(graph_nodes, graph_edges)

# Find decoding strategies
distance = 2
erasure_decoder = LT_Erasure_decoder(n_qbts, distance, gstate, in_qbt=0)
input_strats = erasure_decoder.strategies_ordered

# Run hybrid decoder with a specific measurement order
measurement_order = [1, 2, 3, 4, 5]
adaptive_decoder = LT_FullHybridDecoderNew(
    copy.deepcopy(gstate),
    copy.deepcopy(input_strats),
    measurement_order=measurement_order,
    no_anti_com_flag=False,
    printing=True  # Enable debug output
)

# Get results
print(f"Matter qubits needed: {adaptive_decoder.max_number_of_m_dec}")
print(f"Minimal loss patterns: {adaptive_decoder.all_min_loss_patterns}")
```

#### Loading and Testing Pre-computed Measurement Patterns

```python
from ParsingDataCode.ParseLargeGraphs import get_full_m_patt_list

# Load measurement patterns from CSV
patterns = get_full_m_patt_list("final_best_permutations_13_1_5_a.csv")

# Inspect the first few patterns
for i, pattern in enumerate(patterns[:5]):
    print(f"Pattern {i}: {pattern}")
```

#### Parallel Processing Example

```python
from multiprocessing import Pool
from CodeFunctions.graphs import graph_from_nodes_and_edges
from ErasureDecoder import LT_Erasure_decoder
from HybridDelayedMeasDecoderFixedMeasPatt import LT_FullHybridDecoderNew
import copy

# Setup graph and strategies (done once)
# ... (graph setup code) ...

def run_single_pattern(args):
    """Worker function to evaluate a single measurement pattern."""
    pattern, idx, gstate, strats = args
    decoder = LT_FullHybridDecoderNew(
        copy.deepcopy(gstate),
        copy.deepcopy(strats),
        measurement_order=pattern
    )
    return {str(idx): decoder.max_number_of_m_dec}

# Process patterns in parallel
if __name__ == "__main__":
    patterns = [...]  # Your measurement patterns
    pool = Pool(processes=16)
    args_list = [(patterns[i], i, gstate, input_strats) for i in range(100)]
    results = pool.map(run_single_pattern, args_list)
    pool.close()
    pool.join()

    print(f"Results: {results}")
```

## Algorithm Description

### Overview

The decoder implements a **hybrid delayed measurement strategy** for quantum error correction. It takes a graph state (representing the entanglement structure) and a measurement order as input, then adaptively selects measurements to recover quantum information while minimizing the number of matter qubits required.

### Key Concepts

- **Graph State**: A quantum state defined by an entangling graph structure and local measurements. Each node represents a qubit, and edges represent entanglement.

- **Stabilizer**: An operator that returns +1 eigenvalue on valid quantum states, used for error detection and correction.

- **Loss Pattern**: A subset of qubits that may be lost during transmission (up to the code distance).

- **Measurement Order**: The sequence in which qubits are measured (the key input parameter to optimize).

- **Matter Qubits**: Additional quantum resources needed to store measurement outcomes when measurements must be delayed.

- **Delayed Measurement**: A measurement whose outcome depends on the results of other measurements, requiring quantum memory.

### Algorithm Components

#### 1. Erasure Decoder (`ErasureDecoder.py`)

The erasure decoder finds all valid decoding strategies for a given graph state:

1. **Build Parity Check Matrices**: Convert graph stabilizers to matrices H_X and H_Z over GF(2) (binary field)

2. **Generate Loss Patterns**: Enumerate all combinations of qubit losses up to the code distance d

3. **Find Valid Strategies**: For each loss pattern:
   - Perform Gaussian elimination over GF(2)
   - Extract logical X and Z operators that satisfy commutation relations
   - Build list of valid recovery operations

4. **Order Strategies**: Rank strategies by measurement weight, preferring those with Z operators

#### 2. Hybrid Decoder (`HybridDelayedMeasDecoderFixedMeasPatt.py`)

The hybrid decoder performs adaptive measurement selection:

1. **Calculate Qubit Support**: For each output qubit, determine which strategies can recover it

2. **Tree Search Decoding**: For each output qubit:
   - Initialize measurement tree with the initial state
   - Select qubits to measure according to the measurement order
   - Filter strategies based on anticommutation constraints
   - Branch for each possible measurement outcome
   - Track measurement decisions along tree paths

3. **Extract Results**:
   - Count maximum number of delayed measurements required
   - This count equals the number of matter qubits needed

### Mathematical Framework

The algorithm operates in GF(2) (binary field) using the Pauli operator formalism:

- **Pauli operators**: I (identity), X, Z, Y = iXZ
- **Commutation relations**:
  - Same operators commute: XX = ZZ = I
  - X and Z anticommute: XZ = -ZX
- **Gaussian elimination** over GF(2) finds the kernel of parity check matrices

## Project Structure

```
DelayedDecoder_Cluster/
├── Core Decoder Files
│   ├── ErasureDecoder.py                 # Erasure-based decoding strategies
│   ├── HybridDelayedMeasDecoderFixedMeasPatt.py  # Main hybrid decoder
│   └── RunningDecoderLaptop.py           # Entry point for execution
│
├── CodeFunctions/                        # Utility library
│   ├── GraphStateClass.py                # Graph state representation
│   ├── graphs.py                         # Graph generation utilities
│   ├── StabilizerStateClass.py           # Stabilizer state operations
│   ├── StabStateToGraphState.py          # State conversions
│   ├── lc_equivalence.py                 # Local complementation
│   └── linear_algebra_inZ2.py            # GF(2) linear algebra
│
├── ParsingDataCode/                      # Data processing scripts
│   ├── ParseLargeGraphs.py               # Parse measurement patterns
│   └── RunLargeGraphs.py                 # Execute on large graphs
│
├── Data Files (CSV)                      # Pre-computed measurement patterns
│   ├── final_best_permutations_13_1_5_a.csv
│   ├── final_best_permutations_13_1_5_c.csv
│   ├── final_best_permutations_13_1_5_d.csv
│   ├── final_best_permutations_16_1_6_b.csv
│   ├── final_best_permutations_23_1_7.csv
│   └── final_best_permutations_24_1_8.csv
│
└── Additional Scripts
    ├── MultiproccessSearchLaptop.py      # Parallel search wrapper
    ├── OptimizationGefenFiles.py         # Parse optimization data
    └── ParseDecodingData.py              # Parse decoder output
```

## Data Formats

### Input: Measurement Pattern CSV

```csv
PermutationIndex,Permutation,MatterQubits
75072,"[7,6,1,8,3,9,2,12,10,11,4,5,13]",4
205553,"[2,7,6,8,4,9,5,12,11,10,3,1,13]",4
```

- `PermutationIndex`: Unique identifier for the pattern
- `Permutation`: Measurement order as a list [q1, q2, ..., qn]
- `MatterQubits`: Number of matter qubits required (the metric to minimize)

### Output: JSON Results

```json
{
  "matt": [4, 4, 5, 3],
  "loss": [5, 5, 6, 4]
}
```

- `matt`: Array of matter qubit counts for each tested pattern
- `loss`: Array of corresponding loss tolerance values
