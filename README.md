# JMR Genetic Game of Neural Network Life

A bio-inspired neural network simulator where neurons are living cells with **12 genes** and **5 proteins** that grow, connect, learn, and die on a 28×28 grid. Networks self-assemble through genetic rules, learn via backpropagation with weights stored *inside* each cell's dendrites, and are sculpted by environmental pruning — combining Conway's Game of Life mechanics with gradient descent.

**100/100 on MNIST. 98/100 on Fashion-MNIST.**

---

## Why This Is Different

Traditional neural networks are engineered: you choose the layer sizes, connectivity, and hyperparameters, then train a fixed weight matrix. **This simulator inverts that.** Cells are autonomous agents that:

1. **Grow their own connections** — each cell's dendrite size is encoded in its genes, not set by an architect
2. **Are born and die** — new cells emerge when conditions match their birth gene; cells die from overcrowding, isolation, or environmental pruning
3. **Carry heritable genes** — when two cells breed, offspring inherit genes via crossover and mutation
4. **Store weights locally** — weights live in each cell's dendrite array, not in a separate layer matrix
5. **Set their own learning rate** — each cell can evolve its own plasticity speed, gradient sensitivity, and activation curve
6. **Are sculpted by the environment** — cells that don't contribute (low charge change or low gradient) are pruned, mimicking synaptic pruning during brain development

The result is a network that **self-organizes its own topology** through genetic evolution, then learns through backpropagation, then gets pruned by environmental pressure — a cycle of growth, learning, and selection that mirrors biological neural development.

---

## The Cell: Genes and Proteins

Every cell has two types of information, inspired by molecular biology:

- **Genes (12 values)** — inherited, mostly stable parameters that define the cell's identity and structure. These are the cell's *genotype*.
- **Proteins (5 values)** — dynamic, mutable state that changes every training step. These are the cell's *phenotype* — the expressed behavior resulting from genes interacting with the environment.

### The 12 Genes

Genes 0–2 control **survival and reproduction** (Game of Life rules). Genes 3–11 control **network behavior** (learning and connectivity).

| Gene | Name | Controls | Values | Biological Analogy |
|------|------|----------|--------|--------------------|
| **0** | Overcrowding Tolerance (OT) | Max alive neighbors before cell dies | 2–15 | Apoptosis from contact inhibition |
| **1** | Isolation Tolerance (IT) | Min alive neighbors before cell dies | 2–15 (≤ gene 0) | Death from lack of trophic factors |
| **2** | Birth Threshold (BT) | Exact neighbor count needed to reproduce | 2–15 | Morphogen concentration for mitosis |
| **3** | Mutation Rate (MR) | Probability of gene mutation | 0–100 (somatic: MR/100000, birth: MR/1000) | DNA repair fidelity |
| **4** | Dendrite Size (WG) | Number of synaptic weights | 9, 25, 49, or 81 (perfect squares) | Dendritic arbor complexity |
| **5** | Bias Range (BR) | Initial bias magnitude | 0.001 or 0.01 (discrete) | Resting membrane potential range |
| **6** | Fan-In (AW) | Weight initialization scaling | Count of connected upstream cells | Synaptic normalization factor |
| **7** | Charge Delta (CD) | Threshold for "significant" activity | 0.000001–0.01 | Activity-dependent survival signal |
| **8** | Weight Decay (WD) | L2 regularization strength | 1e-6 to 1e-4 | Synaptic protein turnover rate |
| **9** | Learning Rate (LR) | Synaptic plasticity speed | 0.003–0.05 | Hippocampal vs cortical plasticity |
| **10** | Gradient Threshold (GT) | Pruning survival sensitivity | 1e-8 to 1e-4 (log-uniform) | Neurotrophic factor receptor density |
| **11** | Activation Slope (AS) | Leaky ReLU negative slope | 0.01–0.3 | Neuron selectivity / response curve |

### Why Genes 9–11 Matter

**Gene 9 (Learning Rate)** is the single most impactful gene. In biology, synaptic plasticity varies enormously: hippocampal synapses are highly plastic (fast learning), while primary visual cortex synapses are more stable. By making learning rate a per-cell gene, evolution can discover that deep-layer cells should learn slowly while output cells learn fast — all through natural selection, not manual tuning.

**Gene 10 (Gradient Threshold)** controls how sensitive a cell is to pruning. Cells with a low threshold survive even with minimal learning signal; cells with a high threshold must be actively learning or they die. This creates Darwinian selection pressure: only cells that contribute to the network survive.

**Gene 11 (Activation Slope)** controls the neuron's selectivity. A low slope (0.01) means the neuron strongly suppresses negative inputs — it's highly selective, only responding to positive signals (like classic ReLU). A high slope (0.3) means the neuron is more permissive, passing more of the signal through even when negative. This doesn't create "inhibitory neurons" — all neurons still use the same weighted-sum-plus-bias computation. Instead, it controls *how picky* each neuron is about its inputs. Evolution can discover the right mix of selective vs permissive neurons for optimal feature detection.

### Why Log-Uniform Distributions Matter

Genes 7 (Charge Delta), 8 (Weight Decay), and 10 (Gradient Threshold) span multiple orders of magnitude — for example, gene 10 ranges from 0.00000001 to 0.0001. If you use a simple uniform random distribution over that range, 99.99% of values would land between 0.00009 and 0.0001 — the bottom of the range is effectively invisible. **Log-uniform sampling** (`10^uniform(-8, -4)`) ensures equal probability across *each order of magnitude*, so a cell is equally likely to get a threshold of 1e-7 as 1e-5. This produces genuine diversity in survival sensitivity, decay rate, and activity thresholds — which is exactly what evolution needs to work with.

### Gene 4 (Dendrite Size) Detail

Gene 4 determines `reach = (√genes[4] − 1) / 2`:

| Gene 4 Value | Weight Matrix | Reach | Receptive Field |
|-------------|---------------|-------|-----------------|
| 9 | 3×3 | 1 | 1 cell in each direction |
| 25 | 5×5 | 2 | 2 cells in each direction |
| 49 | 7×7 | 3 | 3 cells in each direction |
| 81 | 9×9 | 4 | 4 cells in each direction |

### The 5 Proteins

Proteins are the dynamic state that changes every forward/backward pass.

| Protein | What It Is | How It Changes | Biological Analogy |
|---------|-----------|----------------|-------------------|
| **Charge** | The cell's activation signal | Forward pass: `leaky_ReLU(bias + sum(upstream_charge × weight))`. Clipped to [−10, 10] | Membrane potential / firing rate |
| **Error** | Backpropagation error signal | Backward pass: accumulated from downstream errors × weights | Retrograde signaling molecules |
| **Bias** | Baseline offset added before activation | Initialized centered at 0, updated by gradient descent: `bias -= lr × error` | Resting membrane potential |
| **Weights** | Synaptic connection strengths (1D array) | He-initialized: `randn × √(2/fan_in)`, updated by `w -= lr × gradient + decay × w` | Synaptic receptor density |
| **Gradient** | Most recent learning signal | `error × upstream_charge`, clipped | Calcium/CaMKII activity level |

### How Genes and Proteins Interact

The key insight is that **genes set the structural constraints, proteins do the work:**

- Gene 4 determines *how many* weights a cell has → Proteins (weights) fill that array and are trained
- Gene 7 determines the *threshold* for significant activity → Protein (charge) is measured against it
- Gene 8 determines *how fast* weights decay → Protein (weights) shrink by that factor each update
- Gene 9 determines *how fast* the cell learns → Protein (weights) update at that rate
- Gene 11 determines the *response curve* → Protein (charge) passes through that activation function
- Genes 0–2 determine *who lives and dies* → The population of cells is shaped by these rules

This creates a two-timescale system:
- **Fast timescale:** Proteins change every training step (gradient descent)
- **Slow timescale:** Genes change across generations (evolution and mutation)

---

## Cell Autonomy

The `U` key toggles `autonomous_network_genes`:

- **Off (default):** All cells share the same network gene values from global config. This is like training a traditional network — uniform architecture and hyperparameters.
- **On:** Each cell has its own random gene values, subject to evolution. This is the bio-inspired mode — cells evolve independently, producing a heterogeneous network.

| Gene | Autonomous Off | Autonomous On |
|------|---------------|---------------|
| 0–2 (breeding) | Always per-cell | Always per-cell |
| 3 (mutation rate) | Same for all cells | Random per cell |
| 4 (dendrite size) | Same for all cells | Random: 9, 25, or 49 |
| 5–8 (network) | Same for all cells | Random per cell |
| 9 (learning rate) | Same `config.learning_rate` | Random 0.003–0.05 |
| 10 (gradient threshold) | Same `config.gradient_threshold` | Log-uniform 1e-8 to 1e-4 |
| 11 (activation slope) | Same `config.activation_slope` | Random 0.01–0.3 |

---

## The Life Cycle of a Network

### Phase 1: Growth (Andromida Mode)
Starting from a sparse grid, cells reproduce according to their birth genes. A cell is born at an empty location if its parent-derived gene 2 matches the local neighbor count. Offspring inherit genes from two parents via crossover, with mutation controlled by gene 3.

### Phase 2: Learning (Training Mode)
Input layer cells are loaded with MNIST pixel data. Charge propagates forward through dendritic weights. Error propagates backward through reversed weight indices. Each cell updates its own weights and bias using its own learning rate (gene 9).

### Phase 3: Pruning (Environmental Selection)
Cells that don't contribute to the network are removed:
- **Activity-based pruning (P key):** Cells whose charge doesn't change significantly across training samples (gene 7 threshold) are killed
- **Gradient-based pruning (O key):** Cells with average gradient below their survival threshold (gene 10) are killed
- **Overcrowding/isolation (Andromida mode):** Cells die if neighbor count exceeds gene 0 or falls below gene 1

### Phase 4: Regrowth
After pruning, evolution can restart. New cells fill gaps, potentially with mutated genes that produce different dendrite sizes, learning rates, or activation slopes. The cycle repeats: grow → learn → prune → regrow.

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
| **Space** | Toggle running mode (enables evolution loop: Andromida birth/death + pruning) |
| **A** | Toggle Andromida mode (genetic birth/death using cell genes) |
| **T** | Toggle training mode (forward pass + optional backprop) |
| **B** | Toggle backpropagation (error signal + weight updates) |
| **F / R** | Forward / Reverse charge flow direction |
| **P** | Toggle activity-based pruning (kills cells with low charge change) |
| **O** | Toggle gradient-based pruning (kills cells with low gradient) |
| **=** | Toggle prune logic between AND/OR |
| **C** | Change charge delta and gradient threshold values |
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
| **Mouse** | Left-click: place/remove cells, Right-click/Ctrl+click: inspect cell, Drag: paint cells |

## Typical Workflow

1. Launch: `python3 -m neurosim.main`
2. Load a saved network (`L`) or draw cells manually
3. Load training data (`M`) — choose MNIST digits or Fashion-MNIST
4. Set forward direction (`F`), enable backprop (`B`), start training (`T`)
5. Optionally enable evolution (`Space` + `A`) between training cycles
6. Toggle pruning (`P` for activity, `O` for gradient) to remove dead cells
7. Disable display (`D`) for faster training
8. Watch accuracy climb — save good networks (`S`)

## How Forward/Backward Pass Works

**Forward pass:** Each cell computes its charge as `leaky_ReLU(bias + sum(upstream_cell.charge × weight))` for all cells within dendrite reach in the layer above. The bias acts as the neuron's resting potential — a baseline signal present even with no input. The leaky ReLU activation (gene 11) introduces nonlinearity, allowing the network to learn non-linear relationships. Weight index: `(dx + reach) × matrix_width + (dy + reach)`.

**Backward pass:** Error propagates through the same dendritic connections in reverse. The reversed weight index `len(weights) − 1 − weight_index` maps to the (−dx, −dy) connection. This is mathematically equivalent to standard backprop through a transposed weight matrix.

**Weight update:** Each cell updates its own weights using its own learning rate (gene 9): `w -= lr × (error × upstream_charge) + weight_decay × w`. No external optimizer — each cell runs its own gradient descent.

## Module Structure

```
neurosim/
  __init__.py           Package init
  config.py             SimConfig dataclass + grid constants
  state.py              SimState dataclass + neighbor cache
  cell.py               Cell class: 12 genes, 5 proteins, forward/backward/die
  training.py           Forward/backward propagation loops
  evolution.py          Andromida mode: breeding, crossover, mutation, death
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

## Results

- **MNIST digits:** 100/100 correct on 6-layer network with 25 weights/cell
- **Fashion-MNIST:** 98/100 on similar architecture
- Networks survive save/reload and can continue training or evolution

## Author

Jonathan Marc Rothberg

## License

MIT
