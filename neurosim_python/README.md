# Python desktop simulator (`neurosim`)

This directory holds the **pygame + OpenGL** desktop implementation. The **primary, actively developed app** is the browser simulator — see the [main README](../README.md) for `neurosim_web.html`, GitHub Pages, training data, and the full genes / proteins / pruning model as used in the UI.

---

## Why this project exists (shared idea)

Traditional NNs use fixed layer matrices. Here, **weights live in each cell’s dendrites**, cells are born and die on a 28×28 grid, and **14 genes** + **5 proteins** + **cell memory** drive structure, learning, and pruning. The same conceptual model is documented in the root README for the web build; the sections below focus on **running and navigating this Python tree**.

---

## Installation

```bash
git clone https://github.com/jmrothberg/3D-neuron-game-of-life-simulator.git
cd 3D-neuron-game-of-life-simulator
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, pygame, numpy, Pillow. Optional: PyOpenGL (3D view), tensorflow/matplotlib (some data prep under `old_code/`).

---

## Running the desktop app

Imports use the package name **`neurosim`**, but this folder is named **`neurosim_python`**. From the **repository root**, create a one-time symlink so Python can resolve `neurosim`:

```bash
ln -sf neurosim_python neurosim
export PYTHONPATH=.
python3 -m neurosim.main
```

If you prefer not to symlink, copy or rename `neurosim_python` to `neurosim` (keep a backup of whichever layout you use).

---

## Module layout (this directory)

| File | Role |
|------|------|
| `__init__.py` | Package exports |
| `config.py` | `SimConfig`, grid constants |
| `state.py` | `SimState`, neighbor cache |
| `cell.py` | Cell: genes, proteins, memory, forward/backward, death |
| `training.py` | Forward/backward loops |
| `evolution.py` | Andromida mode, breeding, pruning, mutation |
| `io_manager.py` | Save/load, MNIST loading |
| `visualization.py` | 2D grid + stats overlay |
| `visualization_3d.py` | OpenGL 3D |
| `ui.py` | Dialogs, side panel |
| `telemetry.py` | Per-layer checks |
| `main.py` | Event loop, entry point |
| `smoke_test.py` | Light regression checks |

---

## Keyboard (desktop + shared with web)

Most keys match the web app. Differences: desktop **M** uses pygame paths for MNIST/Fashion pickles; web **M** uses **J** (file) / **D** (demo fetch) / synthetic. Web stats go to the split Help/Stats pane (**V**, **Q**).

| Key | Action |
|-----|--------|
| **Space** | Toggle running (evolution + pruning loop) |
| **A** | Toggle Andromida (genetic birth/death) |
| **T** | Toggle training |
| **B** | Toggle backprop |
| **F / R** | Forward / reverse charge flow |
| **P** | Activity + weight + contribution pruning |
| **O** | Gradient pruning |
| **=** | AND/OR for charge-based pruning |
| **C** | Pruning parameters (charge delta, gradient threshold, contribution, percentile) |
| **U** | Autonomous network genes |
| **M** | Load training data (desktop: MNIST/Fashion pickles) |
| **K** | Gradient minibatch size |
| **G** | 2D: genes/proteins; 3D: color mode |
| **V** | Statistics views (web: stats pane) |
| **E** | Edit parameters |
| **I** | Learning rate |
| **X** | Reset weights/biases |
| **N** | Nuke hidden layers |
| **S / L** | Save / load JSON |
| **W** | Reset gradient tracking |
| **D** | Toggle heavy cell drawing |
| **3** | 3D view (web: Three.js) |
| **4** | Backprop-oriented 3D coloring (web: full net, Error default) |
| **Q** | Telemetry |
| **H** | Help screens (2D) |

---

## Typical workflow (desktop)

1. `python3 -m neurosim.main` (after symlink / `PYTHONPATH` above).
2. Load a network (**L**) or place cells with the mouse.
3. Load training data (**M**) from prepared pickle folders.
4. **F**, **B**, **T** to train; **V** / side panel for stats; **P** / **O** to prune.
5. **S** to save JSON (same idea as web; paths differ).

---

## The life cycle of a network (algorithm)

### Phase 1: Growth (Andromida)

Cells reproduce when local neighbor count matches gene 2 (birth threshold). Offspring get crossover from two parents; **germline** mutation can re-randomize genes at birth (rate tied to gene 3).

### Phase 2: Learning

Forward charge flow, backward errors, per-cell weight/bias updates (gene 9 learning rate, etc.).

### Phase 3: Pruning

Environmental removal using charge-diff, gradient, weight magnitude, contribution score, and optional percentile cut — see root README for the five strategies.

### Phase 4: Regrowth

Evolution fills gaps; **somatic** mutation can rarely re-randomize an existing cell’s genes (same gene 3, lower probability than at birth).

---

## How forward / backward pass works (detail)

### Grid layout

```
         Layer 0       Layer 1       Layer 2      ...   Layer N-2     Layer N-1
        (Input)       (Hidden)      (Hidden)           (Output)      (Desired)
       ┌────────┐    ┌────────┐    ┌────────┐         ┌────────┐    ┌────────┐
       │ 28×28  │    │ 28×28  │    │ 28×28  │         │ 28×28  │    │ 28×28  │
       │ pixels │───>│ cells  │───>│ cells  │───>...─>│ cells  │    │ labels │
       └────────┘    └────────┘    └────────┘         └────────┘    └────────┘
```

Layer 0 = input charges; layer N−1 = desired labels; hidden layers = 1 … N−2.

### Forward (per cell)

1. Gather upstream charges within dendrite **reach** (from gene 4 / weight count).
2. `charge = bias + Σ(weight × upstream_charge)`.
3. Leaky ReLU with slope from gene 11; clip.
4. Push charge into rolling history; update `max_charge_diff_*` and contribution score.

### Weight indexing

Flat index: `(dx + reach) * matrix_width + (dy + reach)` over the upstream patch.

### Backward (per cell)

1. Output error vs desired; hidden error from downstream errors × reversed weight index.
2. `gradient = error × upstream_charge` (clipped).
3. `weight -= lr * gradient + decay * weight`; `bias -= lr * error`.
4. Update gradient history → `avg_gradient_magnitude` → contribution score.

**Reversed index:** `len(weights) - 1 - forward_index` (transpose of the local receptive field).

---

## Data preparation (pickle pipeline)

The desktop loader expects **per-image pickle folders**. Helper scripts live under **`old_code/`** (not in this directory), for example:

- `JMR_fashion_mnist_to_cell_Oct_3_from_webdata.py` — Fashion-MNIST  
- `JMR_pick_mnist_to_cell_Oct_23.py` — MNIST digits  
- `importMNEST_Save_local.py` — raw MNIST download  

For the **browser**, use `mnist_to_neurosim_web_json.py` and compact JSON (see main README).

---

## Results (reported runs)

- **MNIST:** 100/100 on a 6-layer-style setup with 25 weights/cell (desktop experiments).  
- **Fashion-MNIST:** 98/100 on a similar architecture.  

Networks can be saved and resumed. Exact parity with the web RNG is not guaranteed.

---

## See also

- **[Main README](../README.md)** — `neurosim_web.html`, GitHub Pages, genes vs proteins vs memory, pruning UI, plots, save filenames, troubleshooting.  
- **`README.md`** — canonical manual; the web help panel is this file rendered at build time (`build_neurosim_web.py`).
