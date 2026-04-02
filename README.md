# JMR Genetic Game of Neural Network Life

A bio-inspired neural network simulator where neurons are living cells with **14 genes**, **5 proteins**, and **6 cell-memory fields** that grow, connect, learn, and die on a 28×28 grid. Networks self-assemble through genetic rules, learn via backpropagation with weights stored *inside* each cell's dendrites, and are sculpted by environmental pruning — combining Conway's Game of Life mechanics with gradient descent.

**100/100 on MNIST. 98/100 on Fashion-MNIST.**

---

## Why This Is Different

Traditional neural networks are engineered: you choose the layer sizes, connectivity, and hyperparameters, then train a fixed weight matrix. **This simulator inverts that.** Cells are autonomous agents that:

1. **Grow their own connections** — each cell's dendrite size is encoded in its genes, not set by an architect
2. **Are born and die** — new cells emerge when conditions match their birth gene; cells die from overcrowding, isolation, or environmental pruning
3. **Carry heritable genes** — when two cells breed, offspring inherit genes via crossover and mutation
4. **Store weights locally** — weights live in each cell's dendrite array, not in a separate layer matrix
5. **Set their own learning rate** — each cell can evolve its own plasticity speed, gradient sensitivity, and activation curve
6. **Set their own pruning sensitivity** — each cell carries its own thresholds for weight-magnitude and contribution-score pruning (genes 12–13), allowing evolution to favor cells that resist or accept pruning pressure
7. **Are sculpted by the environment** — cells that don't contribute (low charge change, low gradient, near-zero weights, or low contribution score) are pruned, mimicking synaptic pruning during brain development

The result is a network that **self-organizes its own topology** through genetic evolution, then learns through backpropagation, then gets pruned by environmental pressure — a cycle of growth, learning, and selection that mirrors biological neural development.

---

## The Cell: Genes, Proteins, and Cell Memory

Every cell has three types of information, inspired by molecular biology:

- **Genes (14 values)** — inherited, mostly stable parameters that define the cell's identity, structure, and pruning sensitivity. These are the cell's *genotype*.
- **Proteins (5 values)** — dynamic, mutable state that changes every training step: charge, error, bias, weights, gradient. These are the cell's *phenotype*.
- **Cell Memory (6 fields)** — rolling statistics computed from training history, stored inside the cell. Used for pruning decisions and contribution scoring.

---

### The 14 Genes

Genes 0–2 control **survival and reproduction** (Game of Life rules).
Genes 3–8 control **network structure and regularization**.
Genes 9–11 control **learning dynamics**.
Genes 12–13 control **pruning sensitivity**.

| Gene | Name | Symbol | Controls | Default (non-autonomous) | Autonomous Range | Biological Analogy |
|------|------|--------|----------|--------------------------|------------------|--------------------|
| **0** | Overcrowding Tolerance | OT | Max alive neighbors before cell dies | 2–15 (random) | 2–15 (random) | Apoptosis from contact inhibition |
| **1** | Isolation Tolerance | IT | Min alive neighbors before cell dies | 2–15 (≤ gene 0) | 2–15 (≤ gene 0) | Death from lack of trophic factors |
| **2** | Birth Threshold | BT | Exact neighbor count to reproduce | 2–15 (random) | 2–15 (random) | Morphogen concentration for mitosis |
| **3** | Mutation Rate | MR | Probability of gene mutation | From config | 0–99 (random) | DNA repair fidelity |
| **4** | Dendrite Size | WG | Number of synaptic weights | From config (9, 25, 49, or 81) | 9 or 25 or 49 | Dendritic arbor complexity |
| **5** | Bias Range | BR | Initial bias magnitude | From config | 0.001 or 0.01 | Resting membrane potential range |
| **6** | Fan-In | AW | Weight initialization scaling (He) | From config | Count of connected upstream cells | Synaptic normalization factor |
| **7** | Charge Delta | CD | Threshold for "significant" activity | From config (0.01) | 10^uniform(−6,−2) | Activity-dependent survival signal |
| **8** | Weight Decay | WD | L2 regularization strength | From config (1e-5) | 10^uniform(−6,−4) | Synaptic protein turnover rate |
| **9** | Learning Rate | LR | Synaptic plasticity speed | From config (0.01) | uniform(0.003, 0.05) | Hippocampal vs cortical plasticity |
| **10** | Gradient Threshold | GT | Gradient-pruning survival threshold | From config (1e-4) | 10^uniform(−8,−4) | Neurotrophic factor receptor density |
| **11** | Activation Slope | AS | Leaky ReLU negative slope | From config (0.01) | uniform(0.01, 0.3) | Neuron selectivity / response curve |
| **12** | Weight Prune Threshold | WPT | Min max-|weight| to survive pruning | From config (0.01) | 10^uniform(−3,−1) | Synaptic maintenance threshold |
| **13** | Min Contribution Score | MCS | Min contribution score to survive | From config (0, off) | 10^uniform(−6,−2) | Activity-dependent trophic requirement |

#### Gene Groups Explained

**Genes 0–2 (Breeding):** Always per-cell, even in non-autonomous mode. They govern the Conway's Game of Life dynamics — when cells are born, when they die from overcrowding or isolation.

**Genes 3–8 (Network Structure):** Define the physical architecture of each cell — how many dendrites it has, how weights are initialized and decay, and the threshold for "significant" charge activity.

**Genes 9–11 (Learning Dynamics):** Control how each cell learns. Gene 9 (Learning Rate) is the single most impactful gene — in autonomous mode, evolution can discover that deep-layer cells should learn slowly while output cells learn fast. Gene 11 (Activation Slope) controls neuron selectivity: low slope (0.01) = highly selective, suppresses negative signals; high slope (0.3) = permissive, passes more signal through.

**Genes 12–13 (Pruning Sensitivity):** New in this version. These genes put pruning thresholds *inside* the cell. In non-autonomous mode, all cells share the same global values from config. In autonomous mode, each cell evolves its own sensitivity — cells can become more or less resilient to pruning pressure through natural selection. This is analogous to cells expressing different levels of trophic factor receptors: a cell with a low MCS (gene 13) is "easy to satisfy" and survives with minimal contribution, while a cell with a high MCS is under stronger pressure to contribute or die.

### Gene 4 (Dendrite Size) Detail

Gene 4 determines `reach = (√genes[4] − 1) / 2`:

| Gene 4 Value | Weight Matrix | Reach | Receptive Field |
|-------------|---------------|-------|-----------------|
| 9 | 3×3 | 1 | 1 cell in each direction |
| 25 | 5×5 | 2 | 2 cells in each direction |
| 49 | 7×7 | 3 | 3 cells in each direction |
| 81 | 9×9 | 4 | 4 cells in each direction |

### Why Log-Uniform Distributions Matter

Genes 7, 8, 10, 12, and 13 span multiple orders of magnitude. If you use a uniform random distribution over (say) 1e-8 to 1e-4, 99.99% of values land near the top. **Log-uniform sampling** (`10^uniform(a, b)`) ensures equal probability across each order of magnitude, producing genuine diversity in survival sensitivity, decay rate, and activity thresholds — exactly what evolution needs.

---

### The 5 Proteins

Proteins are the dynamic state that changes every forward/backward pass. They are the cell's *expressed behavior*.

| Protein | Symbol | What It Is | How It Changes | Range | Biological Analogy |
|---------|--------|-----------|----------------|-------|--------------------|
| **Charge** | — | Activation signal | Forward: `leaky_ReLU(bias + Σ(upstream_charge × weight))` | [−10, 10] | Membrane potential / firing rate |
| **Error** | — | Backprop error signal | Backward: accumulated from downstream errors × reversed weights | [−10, 10] | Retrograde signaling molecules |
| **Bias** | — | Baseline offset before activation | `bias -= lr × error` each backward step | Initialized near 0 | Resting membrane potential |
| **Weights** | — | Synaptic connection strengths (1D array, size = gene 4) | He-init: `randn × √(2/fan_in)`. Update: `w -= lr × gradient + decay × w` | Unconstrained | Synaptic receptor density |
| **Gradient** | — | Most recent learning signal | `error × upstream_charge`, clipped to [−clip, +clip] | [−clip, clip] | Calcium/CaMKII activity level |

---

### The 6 Cell Memory Fields

Cell memory stores rolling statistics derived from training. These are computed from proteins but persist across samples. All pruning decisions read from cell memory — the cell carries its own history.

| Field | What It Tracks | How It's Updated | Used By |
|-------|---------------|-----------------|---------|
| **max_charge_diff_forward** | Max charge swing across forward-pass training samples | Each forward pass: push charge, track running max − min | Activity pruning (P key), contribution score |
| **max_charge_diff_reverse** | Max charge swing across reverse-pass training samples | Each reverse pass: push charge, track running max − min | Activity pruning (P key), contribution score |
| **avg_gradient_magnitude** | Rolling average of |gradient| over recent samples | Each backward pass: push |gradient| to history window (size = training data count), compute mean | Gradient pruning (O key), contribution score |
| **contributionScore** | Combined activity + learning signal | `max(max_charge_diff_fwd, max_charge_diff_rev) × avg_gradient_magnitude` | Contribution-score pruning (gene 13), percentile pruning, 3D color mode |
| **significant_charge_change_forward** | Sticky flag: has forward charge ever exceeded gene 7 | Set to `true` when `max_charge_diff_forward > gene[7]`, never cleared (until explicit reset) | Conway death protection only (shouldDieGenetic) |
| **significant_charge_change_reverse** | Sticky flag: has reverse charge ever exceeded gene 7 | Set to `true` when `max_charge_diff_reverse > gene[7]`, never cleared | Conway death protection only (shouldDieGenetic) |

**Key design principle:** All rolling metrics (`max_charge_diff_*`, `avg_gradient_magnitude`, `contributionScore`) are *live* — they reflect recent training and are used for pruning decisions. The *sticky* flags (`significant_charge_change_*`, `significant_gradient_change`) are only used to protect cells from Conway-style genetic death, ensuring that cells that have ever contributed are not killed by overcrowding/isolation rules.

---

### How Genes, Proteins, and Cell Memory Interact

The three information layers create a two-timescale system:

| Timescale | What Changes | Mechanism |
|-----------|-------------|-----------|
| **Per-sample** (fast) | Proteins: charge, error, weights, bias, gradient | Forward/backward pass, gradient descent |
| **Per-sample** (fast) | Cell memory: rolling charge diffs, gradient history, contribution score | Updated inside `updateCharge()` and `updateGradientImportance()` |
| **Per-generation** (slow) | Genes 0–13 | Crossover, mutation, natural selection |

- Gene 4 determines *how many* weights a cell has → Proteins (weights) fill that array and are trained
- Gene 7 determines the *threshold* for significant activity → Cell memory (charge diffs) is measured against it
- Gene 8 determines *how fast* weights decay → Protein (weights) shrink by that factor each update
- Gene 9 determines *how fast* the cell learns → Protein (weights) update at that rate
- Gene 10 determines the *gradient survival threshold* → Cell memory (avg_gradient_magnitude) is compared against it
- Gene 11 determines the *response curve* → Protein (charge) passes through that activation function
- Gene 12 determines *weight-magnitude pruning sensitivity* → Protein (max |weight|) is compared against it
- Gene 13 determines *contribution-score pruning sensitivity* → Cell memory (contributionScore) is compared against it
- Genes 0–2 determine *who lives and dies* → The population of cells is shaped by these rules

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
| 5 (bias range) | Same for all cells | 0.001 or 0.01 |
| 6 (fan-in) | Same for all cells | Measured per cell |
| 7 (charge delta) | Same `config.charge_delta` | Log-uniform 1e-6 to 1e-2 |
| 8 (weight decay) | Same `config.weight_decay` | Log-uniform 1e-6 to 1e-4 |
| 9 (learning rate) | Same `config.learning_rate` | Uniform 0.003–0.05 |
| 10 (gradient threshold) | Same `config.gradient_threshold` | Log-uniform 1e-8 to 1e-4 |
| 11 (activation slope) | Same `config.activation_slope` | Uniform 0.01–0.3 |
| 12 (weight prune threshold) | Same `config.weight_prune_threshold` | Log-uniform 1e-3 to 1e-1 |
| 13 (min contribution score) | Same `config.min_contribution_score` | Log-uniform 1e-6 to 1e-2 |

---

## Pruning: Four Complementary Strategies

Pruning removes cells that don't contribute, mimicking synaptic pruning during brain development. The simulator implements four strategies that can be combined:

### Strategy 1: Activity-Based Pruning (P key)

Cells whose charge doesn't change significantly across training samples are killed. Uses cell memory (`max_charge_diff_forward`, `max_charge_diff_reverse`) compared against gene 7 (Charge Delta).

- **AND logic (`=` key):** Cell must show significant change in *both* forward and reverse passes to survive. Strict — requires bidirectional contribution.
- **OR logic (`=` key):** Cell survives if it shows significant change in *either* direction. More lenient.

### Strategy 2: Gradient-Based Pruning (O key)

Cells with average gradient magnitude below their survival threshold are killed. Uses cell memory (`avg_gradient_magnitude`) compared against gene 10 (Gradient Threshold).

### Strategy 3: Weight-Magnitude Pruning (active when P is on)

Cells whose maximum absolute weight falls below their weight prune threshold are killed. Uses protein data (max |weight|) compared against gene 12 (Weight Prune Threshold). Biologically: a synapse that has decayed to near zero carries no signal.

### Strategy 4: Contribution-Score Pruning (active when P is on)

Cells whose contribution score (`max(charge_diff_fwd, charge_diff_rev) × avg_gradient_magnitude`) falls below their minimum contribution score are killed. Uses cell memory (`contributionScore`) compared against gene 13 (Min Contribution Score). Biologically: combines "is this cell active?" with "is it learning?" into a single survival test.

### Strategy 5: Percentile Pruning (automatic at epoch boundary)

At the end of each epoch, the bottom N% of cells (ranked by contribution score) are killed. Configured via `prune_percentile` (0 = off, set via C key). Unlike strategies 1–4 which use per-cell thresholds, this is a relative/competitive mechanism: cells must outperform their peers to survive.

### Pruning Summary Table

| Strategy | Trigger | What's Compared | Threshold Source | Biological Analogy |
|----------|---------|----------------|-----------------|-------------------|
| Activity (P) | P key on | `max_charge_diff_*` | Gene 7 (per-cell) | Activity-dependent survival |
| Gradient (O) | O key on | `avg_gradient_magnitude` | Gene 10 (per-cell) | Neurotrophic factor requirement |
| Weight-magnitude | P key on | `max(|weight|)` | Gene 12 (per-cell) | Synaptic maintenance threshold |
| Contribution score | P key on | `contributionScore` | Gene 13 (per-cell) | Combined activity + learning requirement |
| Percentile | End of epoch | Rank of `contributionScore` | Config `prune_percentile` (global) | Competitive survival pressure |

---

## The Life Cycle of a Network

### Phase 1: Growth (Andromida Mode)
Starting from a sparse grid, cells reproduce according to their birth genes. A cell is born at an empty location if its parent-derived gene 2 matches the local neighbor count. Offspring inherit all 14 genes from two parents via crossover, with mutation controlled by gene 3.

### Phase 2: Learning (Training Mode)
Input layer cells are loaded with MNIST pixel data. Charge propagates forward through dendritic weights. Error propagates backward through reversed weight indices. Each cell updates its own weights and bias using its own learning rate (gene 9).

### Phase 3: Pruning (Environmental Selection)
Cells that don't contribute to the network are removed through five mechanisms (see Pruning section above). In autonomous mode, each cell's tolerance for pruning pressure is encoded in its genes (7, 10, 12, 13) — evolution discovers which sensitivity levels lead to useful networks.

### Phase 4: Regrowth
After pruning, evolution can restart. New cells fill gaps, potentially with mutated genes that produce different dendrite sizes, learning rates, activation slopes, or pruning sensitivities. The cycle repeats: grow → learn → prune → regrow.

---

## Installation

```bash
git clone https://github.com/jmrothberg/3D-neuron-game-of-life-simulator.git
cd 3D-neuron-game-of-life-simulator
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, pygame, numpy, Pillow. Optional: PyOpenGL (for 3D view), tensorflow/matplotlib (for data prep scripts in `old_code/`).

## Running (Python Desktop Version)

```bash
python3 -m neurosim.main
```

## Keyboard Controls

### All Keys (Desktop + Web)

| Key | Action |
|-----|--------|
| **Space** | Toggle running mode (enables evolution loop: Andromida birth/death + pruning) |
| **A** | Toggle Andromida mode (genetic birth/death using cell genes) |
| **T** | Toggle training mode (forward pass + optional backprop) |
| **B** | Toggle backpropagation (error signal + weight updates) |
| **F / R** | Forward / Reverse charge flow direction |
| **P** | Toggle activity-based pruning (kills cells with low charge change, low weight magnitude, low contribution score) |
| **O** | Toggle gradient-based pruning (kills cells with low gradient) |
| **=** | Toggle prune logic between AND/OR (for charge-based pruning) |
| **C** | Change pruning parameters: charge delta, gradient threshold, min contribution score, prune percentile |
| **U** | Toggle autonomous cell genes (per-cell vs global parameters) |
| **M** | Load MNIST or Fashion-MNIST training data |
| **K** | Set gradient minibatch size |
| **G** | 2D: switch genes/proteins display. 3D: cycle color mode (Charge → Error → Gradient → Weight Strength → Contribution) |
| **V** | Cycle statistics views (Settings + Pruning Readiness / Averages / Cell Types) |
| **E** | Edit all parameters interactively (with network-measured suggestions) |
| **I** | Change learning rate (with suggestion based on fan-in) |
| **X** | Reset network genes/proteins (auto-computes He initialization scaling) |
| **N** | Nuke all hidden layer cells |
| **S / L** | Save / Load network state (JSON) |
| **W** | Reset gradient tracking for all cells |
| **D** | Toggle display updating (training + plots always run; only expensive cell rendering is toggled) |
| **3** | Toggle 3D Three.js view |
| **4** | Toggle 3D backprop visualization |
| **Q** | Dump per-layer telemetry report to stats pane |
| **H** | Cycle help screens (2D only) |
| **?** | Scroll help to top |
| **Mouse** | Left-click: place/remove cells, Right-click/Ctrl+click: inspect cell, Drag: paint cells |

### Web-Only Key Differences

| Key | Web Behavior |
|-----|-------------|
| **M** | Load training data — **J** (local file picker), **D** (fetch demo from server), or **M** (synthetic) |
| **V** | Cycle statistics views → output goes to the dedicated stats pane below the help panel |
| **Q** | Dump per-layer telemetry → stats pane |
| **E** | Edit all parameters with `[net suggests: X]` hints from live network analysis |

---

## 3D Visualization Color Modes

Press **G** in 3D view to cycle through five color modes:

| Mode | What It Shows | Color Mapping |
|------|--------------|---------------|
| **Charge** | Cell activation level | Dark (low) → Bright (high charge) |
| **Error** | Backprop error signal | Blue (negative) → Dark (zero) → Red (positive) |
| **Gradient** | Learning signal strength | Dark (low |gradient|) → Green (high |gradient|) |
| **Weight Strength** | Average absolute weight | Dark (near-zero weights) → Yellow (strong weights) |
| **Contribution** | Combined activity × learning score | Dark (low) → Cyan (medium) → White (high contribution) |

---

## V Key: Statistics Views

Press **V** to cycle through three statistics screens:

### Screen 0: Current Settings + Pruning Readiness
Shows all current config values alongside network-measured suggestions (`← net: X`). Includes:
- Key learning parameters (LR, bias range, weight decay, etc.)
- Pruning thresholds with `[gene 12]` and `[gene 13]` labels
- Network measurements (cells, avg fan-in, median gradient/weight/charge/error)
- **Pruning Readiness:** "Would die" counts for each pruning strategy at current settings
- Contribution score percentile distribution (p10, p25, p50, p75, p90, max)

### Screen 1: Per-Layer Averages
Detailed per-layer statistics: average charge, error, gradient, weight, fan-in, dead neuron count, weight utilization.

### Screen 2: Cell Types
Cell type census and per-digit accuracy breakdown.

---

## Typical Workflow

1. Launch: `python3 -m neurosim.main` (desktop) or open the web version
2. Load a saved network (`L`) or draw cells manually
3. Load training data (`M`) — choose MNIST digits or Fashion-MNIST
4. Set forward direction (`F`), enable backprop (`B`), start training (`T`)
5. Check **V** screen for pruning readiness before enabling pruning
6. Toggle pruning (`P` for activity/weight/contribution, `O` for gradient) to remove dead cells
7. Use **C** to tune pruning thresholds (suggestions shown from network analysis)
8. Disable display (`D`) for faster training
9. Watch accuracy climb — save good networks (`S`)

---

## How Forward/Backward Pass Works

### Key Difference from Traditional Neural Networks

In a traditional NN, weights live in **layer-to-layer matrices** owned by the network. Here, weights live **inside each cell's dendrites** — a flat 1D array (`self.weights`) that the cell owns, carries, and updates independently. There is no global weight matrix.

### The Grid

```
         Layer 0       Layer 1       Layer 2      ...   Layer N-2     Layer N-1
        (Input)       (Hidden)      (Hidden)           (Output)      (Desired)
       ┌────────┐    ┌────────┐    ┌────────┐         ┌────────┐    ┌────────┐
       │ 28×28  │    │ 28×28  │    │ 28×28  │         │ 28×28  │    │ 28×28  │
       │ pixels │───>│ cells  │───>│ cells  │───>...─>│ cells  │    │ labels │
       │(MNIST) │    │        │    │        │         │        │    │        │
       └────────┘    └────────┘    └────────┘         └────────┘    └────────┘
                     ◄──── charge flows forward (F key) ────►
                     ◄──── error flows backward (B key) ────►
```

Layer 0 is loaded with MNIST pixel data. Layer N-1 holds desired output labels.
Layers 1 through N-2 are the hidden layers where cells live, learn, and die.

---

### Forward Pass (one cell's perspective)

Each cell looks at the layer **above** it (closer to input), gathers charges from
nearby cells within its dendrite reach, and computes its own charge.

```
    LAYER ABOVE (layer - 1)                    THIS CELL (layer)
    ┌─────────────────────────┐
    │  .  .  .  .  .  .  .   │
    │  .  .  .  .  .  .  .   │
    │  .  . [A] [B] [C] .  . │                ┌───────────┐
    │  .  . [D] [E] [F] .  . │ ──charges──>   │  Cell X    │
    │  .  . [G] [H] [I] .  . │                │  at (x,y)  │
    │  .  .  .  .  .  .  .   │                └───────────┘
    │  .  .  .  .  .  .  .   │
    └─────────────────────────┘
          3×3 dendrite reach
          (gene 4 = 9 weights)
```

**Step 1: Gather upstream cells.** Cell X checks every position within its `reach`
(determined by gene 4) in the layer above. With gene 4 = 9, reach = 1, so it
checks a 3×3 area centered on its own (x,y) position. It collects the charges
from cells A through I (skipping any empty positions).

**Step 2: Compute weighted sum + bias.**

```
    charge = bias + (A.charge × w[0]) + (B.charge × w[1]) + (C.charge × w[2])
                  + (D.charge × w[3]) + (E.charge × w[4]) + (F.charge × w[5])
                  + (G.charge × w[6]) + (H.charge × w[7]) + (I.charge × w[8])
```

The bias is the cell's resting potential — a baseline signal present even with
no input. It is learned via gradient descent, just like the weights.

**Step 3: Apply leaky ReLU activation.**

```
    if charge > 0:  charge = charge           (pass through)
    if charge ≤ 0:  charge = slope × charge   (gene 11 controls slope)
```

Low slope (0.01) = selective neuron, suppresses negative signals.
High slope (0.3) = permissive neuron, passes more signal through.

**Step 4: Clip to [-10, 10] and store as the cell's new charge.**

**Step 5: Update cell memory.** The new charge is pushed to the cell's charge history array. Rolling `max_charge_diff` is recomputed (max − min of recent charges). `contributionScore` is recomputed as `max(charge_diff_fwd, charge_diff_rev) × avg_gradient_magnitude`.

### Weight Indexing

Weights are stored in a flat 1D array. The 2D offset (dx, dy) from the cell
to each upstream neighbor maps to an array index:

```
    weight_index = (dx + reach) × matrix_width + (dy + reach)

    For a 3×3 dendrite (reach=1, matrix_width=3):

    Upstream position:    (-1,-1) (-1, 0) (-1,+1)      Weight index:  0  1  2
                          ( 0,-1) ( 0, 0) ( 0,+1)  →                  3  4  5
                          (+1,-1) (+1, 0) (+1,+1)                      6  7  8
```

For a 5×5 dendrite (gene 4 = 25, reach = 2), the same formula produces
indices 0–24 over a 5×5 grid of upstream positions.

---

### Backward Pass (one cell's perspective)

Backpropagation has two jobs: (1) compute this cell's error signal, and
(2) update this cell's weights. It works from the output layer back toward
the input layer.

#### Job 1: Compute Error Signal

**Output layer cells** (layer N-2): error = how wrong we are.

```
    error = (my_charge − desired_output) × leaky_ReLU_derivative(my_charge)
```

**Hidden layer cells**: error = share of blame from cells below.

```
    THIS CELL (layer)                          LAYER BELOW (layer + 1)
    ┌───────────┐                              ┌─────────────────────────┐
    │  Cell X    │                              │  .  .  .  .  .  .  .   │
    │  error = ? │ <──errors + weights────      │  .  . [1] [2] [3] .  . │
    └───────────┘                               │  .  . [4] [5] [6] .  . │
                                                │  .  . [7] [8] [9] .  . │
                                                │  .  .  .  .  .  .  .   │
                                                └─────────────────────────┘
                                                 These cells BELOW have
                                                 already computed their
                                                 errors (we work backward)
```

Cell X accumulates error from each cell below that connects to it:

```
    error = Σ (cell_below.error × cell_below.weights[reversed_index])
            × leaky_ReLU_derivative(my_charge)
```

**Why `reversed_index`?** In the forward pass, Cell 5 (below) uses Cell X's
charge via weight_index to compute Cell 5's charge. In the backward pass,
we need the reverse: Cell X needs Cell 5's error via the **same connection**,
but looked up from Cell 5's weight array in the opposite direction:

```
    reversed_index = len(cell_below.weights) − 1 − weight_index

    Forward (Cell 5 looks UP at Cell X):         Backward (Cell X looks DOWN at Cell 5):
    weight_index = (dx + reach) × m + (dy + reach)
                                                  reversed_index = len(w) - 1 - weight_index

    For 3×3:  forward index 0 ↔ reversed index 8     (-1,-1) ↔ (+1,+1)
              forward index 1 ↔ reversed index 7     (-1, 0) ↔ (+1, 0)
              forward index 4 ↔ reversed index 4     ( 0, 0) ↔ ( 0, 0)  (center stays)
              forward index 8 ↔ reversed index 0     (+1,+1) ↔ (-1,-1)

    This maps (dx, dy) → (−dx, −dy), which is equivalent to transposing
    the weight matrix — exactly what standard backprop does.
```

#### Job 2: Update Weights

Once Cell X knows its error, it updates each of its own dendritic weights
using the upstream cell charges (same cells used in the forward pass):

```
    For each upstream cell in the layer above:
        gradient = my_error × upstream_cell.charge
        gradient = clip(gradient, −clip_range, +clip_range)

        weight[index] −= learning_rate × gradient + weight_decay × weight[index]
                         ╰──────── learn ────────╯   ╰──── regularize ────────╯

    bias −= learning_rate × my_error
```

In autonomous mode, `learning_rate` comes from gene 9 and `weight_decay`
from gene 8 — each cell runs its own gradient descent at its own speed.

**Step 3: Update cell memory.** After weight updates, `updateGradientImportance()` pushes |gradient| to the rolling history, recomputes `avg_gradient_magnitude`, and recomputes `contributionScore`.

---

### Complete Forward + Backward Example

```
    Layer 0 (Input)    Layer 1 (Hidden)    Layer 2 (Output)    Layer 3 (Desired)
    ┌──────────┐       ┌──────────┐        ┌──────────┐        ┌──────────┐
    │          │       │          │        │          │        │          │
    │  MNIST   │──F──> │  Cells   │──F───> │  Cells   │        │  Labels  │
    │  pixels  │       │  learn   │        │  predict │        │  (0-9)   │
    │          │       │          │ <─B─── │          │ <─B─── │          │
    └──────────┘       └──────────┘        └──────────┘        └──────────┘

    F = forward pass: charge flows left to right, layer by layer
    B = backward pass: error flows right to left, layer by layer

    1. Load MNIST image into Layer 0 (pixel values become cell charges)
    2. Load one-hot label into Layer 3 (desired output)
    3. FORWARD: Layer 1 cells compute charges from Layer 0
                Layer 2 cells compute charges from Layer 1
    4. BACKWARD: Layer 2 error = (Layer 2 charge − Layer 3 desired)
                 Layer 2 updates its weights using Layer 1 charges
                 Layer 1 error = accumulated from Layer 2 errors
                 Layer 1 updates its weights using Layer 0 charges
    5. Cell memory updated: charge diffs, gradient history, contribution scores
    6. Repeat for next training sample
```

---

## Module Structure (Python Desktop)

```
neurosim/
  __init__.py           Package init
  config.py             SimConfig dataclass + grid constants
  state.py              SimState dataclass + neighbor cache
  cell.py               Cell class: 14 genes, 5 proteins, 6 cell memory fields, forward/backward/die
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

---

## Browser Version (JavaScript / Single-File HTML)

### Try it now — no install required

**[▶ Run in your browser](https://jmrothberg.github.io/3D-neuron-game-of-life-simulator/neurosim_web.html)**

The link opens the simulator on GitHub Pages. Press **M** → **D** → **OK** to auto-fetch the bundled 500-sample MNIST demo, then **T** to start training. Everything runs client-side in your browser — no server, no Python, no downloads.

---

### Overview

The web version is a full reimplementation of the desktop `neurosim/` simulator in JavaScript + HTML Canvas + Three.js. All simulation logic lives in **`neurosim_web.js`**; the build script `build_neurosim_web.py` inlines it (plus help text from `get_help_defs.py`) into the standalone **`neurosim_web.html`**.

### Key Features (Web Version)

| Feature | Details |
|---------|---------|
| **Full neural-network training** | Forward pass, backpropagation, weight/bias updates, cross-entropy loss — identical algorithm to the Python version |
| **14 heritable genes** | All 14 genes (breeding, network, learning, pruning) with autonomous/non-autonomous modes |
| **5 pruning strategies** | Activity, gradient, weight-magnitude, contribution-score, and percentile pruning — all configurable via C key |
| **Epoch-based training loop** | One epoch = one full pass over all loaded samples; epochs are counted and displayed in the status bar |
| **Minibatch gradient accumulation** | Configurable batch size **K** (gradient_minibatch_size); gradients accumulate over K samples before one weight update |
| **Epoch shuffling** | Training samples are Fisher-Yates shuffled at the start of each epoch |
| **Dual loss plots** | Separate scrolling plots for per-minibatch loss (blue) and per-epoch loss (green), each with an independent Y-axis scale slider |
| **Live status bar** | Shows: display toggle, current epoch with progress, total samples, minibatch K, epoch loss, batch loss, correct predictions, max correct |
| **Dedicated statistics pane** | Right panel split into Help (top) and Stats (bottom) with draggable splitter; stats show pruning readiness, per-layer metrics, contribution score distribution |
| **Smart defaults with suggestions** | `suggestParams()` scans the network and suggests LR, bias range, weight decay, pruning thresholds based on live measurements. Shown in E, I, C dialogs and V screen |
| **5 color modes in 3D** | Charge, Error, Gradient, Weight Strength, Contribution — cycle with G key |
| **He initialization** | Weight reset (X key) measures average effective fan-in and scales as `randn × √(2/fan_in)` |
| **Compact MNIST format** | Training JSON stores only pixel charges + label per sample (~2 MB / 500 samples); auto-detected on load |
| **Demo data fetch** | **M** → **D** fetches `mnist_demo_500.json` from the server (works on GitHub Pages and local HTTP) |
| **3D visualization** | Three.js orbit-camera view with round point sprites, visible connection lines, and backprop highlight mode (4 key) |
| **Save / Load** | **S** exports full network state as JSON; **L** imports it; auto-downloads a snapshot when accuracy hits 100% |
| **Display toggle** | **D** key toggles only expensive cell rendering; training, graphs, and status bar always update |

### Building the Standalone HTML

```bash
python3 build_neurosim_web.py
```

This reads `neurosim_web.js` and `get_help_defs.py`, injects help text as a `HELP_SCREEN` JSON object, and produces `neurosim_web.html`. Re-run after editing `neurosim_web.js`.

### Running Locally

**Option A — direct open:** Double-click `neurosim_web.html`. Works for most features, but Chromium may block some `file://` loads (see troubleshooting below).

**Option B — local HTTP server (recommended):**

```bash
python3 -m http.server 8765
# then open http://localhost:8765/neurosim_web.html
```

This avoids all `file://` security restrictions and enables the **M** → **D** demo-fetch path.

### Loading MNIST Training Data

**Quick (demo, 500 samples):** Press **M** → **D** → **OK**. The 500-sample compact JSON is fetched automatically.

**Full dataset (up to 5000 samples):** Generate a local file with the conversion script:

```bash
python3 mnist_to_neurosim_web_json.py --dataset mnist --count 5000 -o mnist_training_web.json
# Fashion-MNIST: --dataset fashion
# SSL issues: add --no-verify-ssl
```

Then press **M** → **J** → select the file. Options: `--offset`, `--count`, `--cache-dir`.

The compact format (default) stores only charge values and labels — ~22 MB for 5000 samples vs ~1.9 GB verbose. Use `--verbose-cells` for the old format if needed.

### Troubleshooting

**`file://` security errors:** Chromium may log *"Unsafe attempt to load URL … 'file:' URLs are treated as unique security origins"*. Each `file://` path is treated as its own origin, so subframe loads and fetch calls can fail. **Fix:** serve the repo over HTTP (`python3 -m http.server 8765`) and open via `http://localhost:8765/neurosim_web.html`.

**Large JSON files crashing the browser:** If you generated verbose-format JSON for thousands of samples, the file can be 1+ GB. Use the compact format (default in `mnist_to_neurosim_web_json.py`) or reduce `--count`.

### Known Gaps vs Desktop

- Pickle **saved_states/** and per-run **.png** icons are desktop-only; **L** loads JSON exported with **S**
- MNIST in pygame reads prepared pickle folders (**M** → **M** vs **F**); the browser uses **J**/**D** + JSON
- `state.timing` evolution prints are not wired to the UI
- Float RNG differs from NumPy; 3D backprop highlight differs from OpenGL; tab close replaces pygame quit

## Author

Jonathan Marc Rothberg

## License

MIT
