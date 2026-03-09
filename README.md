# JMR Genetic Game of Neural Network Life

A bio-inspired neural network simulator where neurons are living cells with **9 genes** and **5 proteins** that grow, connect, learn, and die on a 28×28 grid. Networks self-assemble through genetic rules, learn via backpropagation with weights stored *inside* each cell's dendrites, and are sculpted by environmental pruning — combining Conway's Game of Life mechanics with gradient descent.

**100/100 on MNIST. 98/100 on Fashion-MNIST.**

---

## Why This Is Different

Traditional neural networks are engineered: you choose the layer sizes, connectivity, and hyperparameters, then train a fixed weight matrix. **This simulator inverts that.** Cells are autonomous agents that:

1. **Grow their own connections** — each cell's dendrite size is encoded in its genes, not set by an architect
2. **Are born and die** — new cells emerge when conditions match their birth gene; cells die from overcrowding, isolation, or environmental pruning
3. **Carry heritable genes** — when two cells breed, offspring inherit genes via crossover and mutation
4. **Store weights locally** — weights live in each cell's dendrite array, not in a separate layer matrix
5. **Are sculpted by the environment** — cells that don't contribute (low charge change or low gradient) can be pruned, mimicking how the brain eliminates weak synapses during development

The result is a network that **self-organizes its own topology** through genetic evolution, then learns through backpropagation, then gets pruned by environmental pressure — a cycle of growth, learning, and selection that mirrors biological neural development.

---

## The Cell: Genes and Proteins

Every cell in the simulator has two types of information, inspired by molecular biology:

- **Genes (9 values)** — inherited, mostly stable parameters that define the cell's identity and structure. These are the cell's *genotype*.
- **Proteins (5 values)** — dynamic, mutable state that changes every training step. These are the cell's *phenotype* — the expressed behavior that results from genes interacting with the environment.

### The 9 Genes

Genes 0–2 control **survival and reproduction** (Game of Life rules). Genes 3–8 control **network behavior** (learning and connectivity).

| Gene | Name | Controls | Values | Biological Analogy |
|------|------|----------|--------|--------------------|
| **0** | Overcrowding Tolerance (OT) | Max alive neighbors before cell dies | 2–15 | Apoptosis from contact inhibition |
| **1** | Isolation Tolerance (IT) | Min alive neighbors before cell dies | 2–15 (≤ gene 0) | Death from lack of trophic factors |
| **2** | Birth Threshold (BT) | Exact neighbor count needed to reproduce | 2–15 | Morphogen concentration for mitosis |
| **3** | Mutation Rate (MR) | Probability of gene mutation | 0–100 (used as MR/100000) | DNA repair fidelity |
| **4** | Dendrite Size (WG) | Number of synaptic weights | 9, 25, 49, or 81 (perfect squares) | Dendritic arbor complexity |
| **5** | Bias Range (BR) | Initial bias magnitude | 0.001–0.01 | Resting membrane potential range |
| **6** | Fan-In (AW) | Weight initialization scaling | Count of connected upstream cells | Synaptic normalization factor |
| **7** | Charge Delta (CD) | Threshold for "significant" activity | 0.000001–0.01 | Activity-dependent survival signal |
| **8** | Weight Decay (WD) | L2 regularization strength | 1e-6 to 1e-4 | Synaptic protein turnover rate |

**Gene 4 (Dendrite Size)** deserves special attention. It determines `reach = (√genes[4] − 1) / 2`:

| Gene 4 Value | Weight Matrix | Reach | Receptive Field |
|-------------|---------------|-------|-----------------|
| 9 | 3×3 | 1 | 1 cell in each direction |
| 25 | 5×5 | 2 | 2 cells in each direction |
| 49 | 7×7 | 3 | 3 cells in each direction |
| 81 | 9×9 | 4 | 4 cells in each direction |

In autonomous mode, different cells can have different dendrite sizes — some develop wide receptive fields while others stay local, analogous to how biological neurons vary enormously in their dendritic complexity.

### The 5 Proteins

Proteins are the dynamic state that changes every forward/backward pass. They are the "expressed behavior" of the cell.

| Protein | What It Is | How It Changes | Biological Analogy |
|---------|-----------|----------------|-------------------|
| **Charge** | The cell's activation signal | Forward pass: weighted sum of upstream charges. Clipped to [−10, 10] | Membrane potential / firing rate |
| **Error** | Backpropagation error signal | Backward pass: accumulated from downstream cells' errors × weights | Retrograde signaling molecules |
| **Bias** | Offset added to activation | Updated by gradient descent: `bias -= lr × error` | Resting potential / intrinsic excitability |
| **Weights** | Synaptic connection strengths (1D array) | Updated by gradient descent: `w -= lr × gradient + decay × w` | Synaptic receptor density |
| **Gradient** | Most recent learning signal | `error × upstream_charge`, clipped | Calcium/CaMKII activity level |

### How Genes and Proteins Interact

The key insight is that **genes set the structural constraints, proteins do the work:**

- Gene 4 determines *how many* weights a cell has → Proteins (weights) fill that array and are trained
- Gene 7 determines the *threshold* for significant activity → Protein (charge) is measured against it to decide survival
- Gene 8 determines *how fast* weights decay → Protein (weights) are shrunk by that factor each update
- Gene 6 determines *how* weights are initialized → Protein (weights) start at values scaled by this gene
- Genes 0–2 determine *who lives and dies* → The population of cells (and their proteins) is shaped by these rules

This creates a two-timescale system:
- **Fast timescale:** Proteins change every training step (gradient descent)
- **Slow timescale:** Genes change across generations (evolution and mutation)

---

## The Life Cycle of a Network

### Phase 1: Growth (Andromida Mode)
Starting from a sparse grid, cells reproduce according to their birth genes. A cell is born at an empty location if its parent-derived gene 2 matches the local neighbor count. Offspring inherit genes from two parents via crossover, with mutation controlled by gene 3.

### Phase 2: Learning (Training Mode)
Input layer cells are loaded with MNIST pixel data. Charge propagates forward through dendritic weights. Error propagates backward through reversed weight indices. Each cell updates its own weights and bias — there is no global weight matrix.

### Phase 3: Pruning (Environmental Selection)
Cells that don't contribute to the network can be removed:
- **Activity-based pruning (P key):** Cells whose charge doesn't change significantly across training samples (measured by gene 7) are killed
- **Gradient-based pruning (O key):** Cells with near-zero average gradient magnitude (not learning) are killed
- **Overcrowding/isolation (Andromida mode):** Cells die if their neighbor count exceeds gene 0 or falls below gene 1

This pruning is analogous to **synaptic pruning** in brain development — the brain overproduces neurons and connections, then eliminates the weak ones based on activity.

### Phase 4: Regrowth
After pruning, evolution can restart. New cells fill gaps, potentially with mutated genes that produce different dendrite sizes, decay rates, or survival thresholds. The cycle repeats: grow → learn → prune → regrow.

---

## Cell Autonomy: What's Per-Cell vs Global

A central design question is whether network parameters should be **cell-autonomous** (each cell has its own value, encoded in its genes) or **global** (all cells share the same value from config). The `U` key toggles `autonomous_network_genes`:

### Currently Cell-Autonomous (genes 0–2, always)
- Survival thresholds (OT, IT) — each cell has its own death rules
- Birth threshold (BT) — each cell type requires different neighbor counts

### Cell-Autonomous When `autonomous_network_genes = True` (genes 3–8)
- Dendrite size / reach (gene 4) — cells can have different receptive fields
- Mutation rate (gene 3) — some cells mutate faster than others
- Weight decay (gene 8) — different L2 regularization per cell
- Bias range, fan-in, charge delta (genes 5–7)

### Always Global (no gene exists)
These parameters are shared by ALL cells and represent opportunities for future cell-autonomous evolution:

| Parameter | Current Source | Impact | Why It Should Be Per-Cell |
|-----------|---------------|--------|--------------------------|
| **Learning Rate** | `config.learning_rate` | HIGH | Different layers/positions should learn at different speeds. Biology: synaptic plasticity varies enormously across brain regions |
| **Gradient Clip Range** | `config.gradient_clip_range` | MEDIUM | Deep-layer cells might need different clipping than shallow ones |
| **Gradient Threshold** | `config.gradient_threshold` | MEDIUM-HIGH | Determines which cells survive pruning — should be self-regulated |
| **Activation Function** | Hardcoded leaky ReLU | MEDIUM | Biology has many neuron types (excitatory, inhibitory, neuromodulatory) with different response curves |
| **Charge Clipping** | Hardcoded [−10, 10] | LOW | Some cells might need wider dynamic range |

Making learning rate a per-cell gene (gene 9) would be the single most impactful change — it would allow cells to evolve their own plasticity rates, with deep-layer cells potentially learning slower and output cells faster, all determined by natural selection rather than manual tuning.

---

## Installation

```bash
git clone https://github.com/jmrothberg/3D-neuron-game-of-life-simulator.git
cd 3D-neuron-game-of-life-simulator
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, pygame, numpy, Pillow. Optional: PyOpenGL (for 3D view), tensorflow/matplotlib (for data prep scripts in `old_code/`).

## Running

```bash
python3 -m neurosim.main
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| **Space** | Toggle Game of Life evolution (cells born/die by neighbor count) |
| **A** | Toggle Andromida mode (genetic birth/death using cell genes) |
| **T** | Toggle training mode (forward pass + optional backprop) |
| **B** | Toggle backpropagation (error signal + weight updates) |
| **F / R** | Forward / Reverse charge flow direction |
| **P** | Toggle activity-based pruning (kills cells with low charge change) |
| **O** | Toggle gradient-based pruning (kills cells with low gradient) |
| **U** | Toggle autonomous cell genes (per-cell vs global parameters) |
| **M** | Load MNIST or Fashion-MNIST training data |
| **G** | Switch genes/proteins display mode |
| **V** | Cycle statistics views |
| **E** | Edit all parameters interactively |
| **I** | Change learning rate |
| **X** | Reset network genes/proteins |
| **N** | Nuke all hidden layer cells |
| **S / L** | Save / Load network state |
| **3** | Toggle 3D OpenGL view |
| **4** | Toggle 3D backprop visualization |
| **Q** | Dump per-layer telemetry report |
| **W** | Reset gradient tracking |
| **D** | Toggle display updating (faster training when off) |
| **H** | Cycle help screens |
| **Mouse** | Left-click: place/remove cells, Right-click: inspect cell |

## Typical Workflow

1. Launch the simulator: `python3 -m neurosim.main`
2. Load a saved network (`L`) or draw cells manually
3. Load training data (`M`) — choose MNIST digits or Fashion-MNIST
4. Set forward direction (`F`), enable backprop (`B`), start training (`T`)
5. Optionally enable evolution (`Space` + `A`) between training cycles for growth/pruning
6. Toggle pruning (`P` for activity, `O` for gradient) to remove dead cells
7. Disable display (`D`) for faster training
8. Watch accuracy climb — save good networks (`S`)

## How Forward/Backward Pass Works

**Forward pass:** Each cell computes its charge by summing `(upstream_cell.charge × weight)` for all cells within dendrite reach in the layer above. Weight index: `(dx + reach) × matrix_width + (dy + reach)`.

**Backward pass:** Error propagates through the same dendritic connections in reverse. The reversed weight index `len(weights) − 1 − weight_index` maps to the (−dx, −dy) connection. This is mathematically equivalent to standard backprop through a transposed weight matrix, but computed locally by each cell.

**Weight update:** Each cell updates its own weights: `w -= lr × (error × upstream_charge) + weight_decay × w`. No external optimizer — each cell runs its own gradient descent.

## Module Structure

```
neurosim/
  __init__.py           Package init
  config.py             SimConfig dataclass + grid constants
  state.py              SimState dataclass + neighbor cache
  cell.py               Cell class: 9 genes, 5 proteins, forward/backward/die
  training.py           Forward/backward propagation loops
  evolution.py          Andromida mode: breeding, mutation, death
  io_manager.py         Save/load networks, MNIST data loading
  visualization.py      2D cell rendering + statistics overlay
  visualization_3d.py   3D OpenGL rendering (cached vertex arrays, HUD)
  ui.py                 Input dialogs, side panel
  telemetry.py          Per-layer validation and NaN detection
  main.py               Event loop + entry point
  smoke_test.py         Regression tests
```

## Data Preparation

MNIST and Fashion-MNIST data must be preprocessed into per-image pickle files. Scripts are in `old_code/`:

- `JMR_fashion_mnist_to_cell_Oct_3_from_webdata.py` — Fashion-MNIST
- `JMR_pick_mnist_to_cell_Oct_23.py` — MNIST digits
- `importMNEST_Save_local.py` — Raw MNIST download

These create directories of `.pkl` files (one per image) used by the `M` key loader.

## Results

- **MNIST digits:** 100/100 correct on 6-layer network with 25 weights/cell
- **Fashion-MNIST:** 98/100 on similar architecture
- Networks survive save/reload and can continue training or evolution

## Author

Jonathan Marc Rothberg

## License

MIT
