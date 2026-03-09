# JMR Genetic Game of Neural Network Life

A bio-inspired neural network simulator where neurons are cells with genetic parameters that evolve, reproduce, and die following Game of Life mechanics -- while simultaneously learning via backpropagation with weights stored in cell dendrites, not separate layer matrices.

100/100 on MNIST. 98/100 on Fashion-MNIST.

## What Makes This Different

In traditional neural networks, weights exist in layer-to-layer matrices. Here, each cell owns its own dendritic weight array -- a flat vector whose size is determined by its genes. Cells are born, mutate, and die on a 28x28 grid across up to 16 layers, governed by 9 genes:

| Gene | Controls |
|------|----------|
| Overcrowding Tolerance (OT) | Max neighbors before death |
| Isolation Tolerance (IT) | Min neighbors before death |
| Birth Threshold (BT) | Exact neighbor count to reproduce |
| Mutation Rate (MR) | Chance of gene mutation |
| Weights per Cell (WG) | Dendrite size (9, 25, 49, 81) |
| Bias Range (BR) | Initial bias magnitude |
| Average Weights (AW) | Weight initialization scaling |
| Charge Delta (CD) | Threshold for "significant" activity |
| Weight Decay (WD) | L2 regularization per cell |

Cells can share global parameters or evolve independently (toggle with `U` key).

## Versions

### Current: Modular Package (`neurosim/`)

Refactored into 12 modules with proper separation of concerns, debugging tools, and cell autonomy methods.

```bash
python3 -m neurosim.main
```

### Legacy: Single File (`3D_NeuroSim_BACKUP.py`)

The original 2,400-line monolithic file. Preserved for reference. The current entry point (`3D_NeuroSim_Oct_29_back3D_visualization_cell.py`) is a thin wrapper that calls the modular version.

```bash
python3 3D_NeuroSim_BACKUP.py
```

## Installation

```bash
git clone https://github.com/jmrothberg/3D-neuron-game-of-life-simulator.git
cd 3D-neuron-game-of-life-simulator
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, pygame, numpy, Pillow. Optional: PyOpenGL (for 3D view), tensorflow/matplotlib (for data prep scripts).

## Keyboard Controls

| Key | Action |
|-----|--------|
| **Space** | Toggle Game of Life evolution |
| **A** | Toggle Andromida mode (genetic birth/death) |
| **T** | Toggle training mode (requires loaded data) |
| **B** | Toggle backpropagation |
| **F / R** | Forward / Reverse charge flow |
| **M** | Load MNIST or Fashion-MNIST data |
| **P** | Toggle activity-based pruning |
| **O** | Toggle gradient-based pruning |
| **U** | Toggle autonomous cell genes |
| **G** | Switch genes/proteins display |
| **V** | Cycle statistics views |
| **E** | Edit all parameters |
| **I** | Change learning rate |
| **X** | Reset network genes/proteins |
| **N** | Nuke all hidden layer cells |
| **S / L** | Save / Load network state |
| **3** | Toggle 3D view |
| **4** | Toggle 3D backprop visualization |
| **Q** | Dump telemetry/validation report |
| **W** | Reset gradient tracking |
| **D** | Toggle display updating (faster training) |
| **H** | Cycle help screens |
| **Mouse** | Left-click to place/remove cells, right-click to inspect |

## Module Structure (`neurosim/`)

```
neurosim/
  __init__.py           - Package init, exports SimConfig/SimState/Cell
  config.py             - SimConfig dataclass + constants
  state.py              - SimState dataclass + neighbor cache
  cell.py               - Cell class (genes, proteins, dendrite weights,
                          forward/backward/validate/should_i_die)
  training.py           - Forward/backward propagation, training loop
  evolution.py          - Andromida mode, pruning, mutation
  io_manager.py         - Save/load files, MNIST data loading
  visualization.py      - 2D cell rendering, statistics
  visualization_3d.py   - OpenGL 3D rendering (optional)
  ui.py                 - Input dialogs, side panel
  telemetry.py          - Per-layer telemetry and NaN detection
  main.py               - Event loop, entry point
  smoke_test.py         - Regression tests
```

## How It Works

**Forward pass:** Each cell computes its charge from upper-layer cells within dendrite reach. Weight index: `(dx + reach) * matrix_width + (dy + reach)`.

**Backpropagation:** Error signals propagate through the same dendritic connections in reverse. The reversed weight index `len(weights) - 1 - weight_index` maps to the (-dx, -dy) connection -- mathematically equivalent to standard backprop through a shared weight matrix.

**Evolution:** Between training cycles, cells can reproduce (inheriting genes from two parents with crossover), mutate, or die based on neighbor count and their genetic thresholds. Cells showing significant charge or gradient activity are protected from pruning.

## Typical Workflow

1. Launch the simulator
2. Draw cells or load a saved network (`L`)
3. Load training data (`M`) -- choose MNIST digits or Fashion
4. Press `F` for forward direction, `B` for backprop, `T` to start training
5. Optionally press `Space` + `A` for evolution between training cycles
6. Press `D` to disable display for faster training
7. Watch accuracy climb in the bottom status bar

## Data Preparation

MNIST and Fashion-MNIST data must be preprocessed into per-image pickle files. Use the included scripts:

- `JMR_fashion_mnist_to_cell_Oct_3_from_webdata.py` -- Fashion-MNIST
- `JMR_pick_mnist_to_cell_Oct_23.py` -- MNIST digits
- `importMNEST_Save_local.py` -- Raw MNIST download

These create directories of `.pkl` files (one per image) at paths configured in `neurosim/io_manager.py`.

## Results

- **MNIST digits:** 100/100 correct on 8-layer network with 49 weights/cell
- **Fashion-MNIST:** 98/100 on similar architecture
- Networks can be saved and reloaded to continue training or evolution

## Author

Jonathan Marc Rothberg

## License

MIT
