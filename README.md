# 🧬 JMR Genetic Game of Neural Network Life

A bio-inspired neural network simulation combining cellular automata, evolutionary algorithms, and backpropagation. This project implements a novel approach where neurons are organized as cells with genetic parameters that can evolve, reproduce, and die based on Conway's Game of Life-inspired rules.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/topics/neural-network)

**🚀 Quick Start**
```bash
git clone https://github.com/yourusername/JMRGeneticsGameOfLife.git
cd JMRGeneticsGameOfLife
pip install -r requirements.txt
python 3D_NeuroSim_Feb_4_25_list_of_cells.py
```

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [📋 Complete File Listing & Purpose](#-complete-file-listing--purpose)
- [🛠️ Installation & Setup](#️-installation--setup)
- [🚀 Usage](#-usage)
- [📊 Data Acquisition Guide](#-data-acquisition-guide)
- [🧬 Gene System](#-gene-system)
- [🏗️ Architecture](#️-architecture)
- [Results](#results)
- [Development History](#development-history)
- [Related Work](#related-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project introduces **Cellular Neural Networks (CNN)** - a hybrid learning system that combines:

- **Cellular Automata**: Cells can be born, survive, or die based on neighbor counts and genetic rules
- **Evolutionary Algorithms**: Cells reproduce with genetic inheritance and mutation
- **Neural Networks**: Each cell contains dendritic weights trained via backpropagation
- **3D Visualization**: Interactive OpenGL-based visualization of network state and learning

### Key Innovation

Unlike traditional neural networks where weights are associated with output connections, each cell in this system stores a **dendritic weight matrix** representing incoming synaptic connections. The cell's structure and behavior are determined by:

- **Genes** (0-8): Control cell lifecycle, network architecture, and learning parameters
- **Proteins**: Represent runtime state (charge, error, gradient) during forward/backward passes

## Features

- **Interactive 2D/3D Visualization** with Pygame and OpenGL
- **Real-time Training** on MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 datasets
- **Genetic Control** of network architecture through 9 distinct genes
- **Dynamic Network Topology** - cells can be born, die, and mutate during training
- **Forward and Reverse Propagation** with configurable direction
- **Pruning Mechanisms** based on charge and gradient thresholds
- **Save/Load Network States** for experiment reproducibility

## 📋 Complete File Listing & Purpose

Here's every file in the repository and what it does:

### 🏗️ **CORE SIMULATION FILES**
| File | Purpose | When to Use | Key Features |
|------|---------|-------------|--------------|
| **`3D_NeuroSim_Feb_4_25_list_of_cells.py`** | **🎯 MAIN SIMULATION** | **Start here!** Latest complete version | Active cell optimization, 3D visualization, gradient pruning, full training |
| `3D_NeuroSim_Oct_29_back3D_visualization_cell.py` | 3D Network Viewer | For 3D visualization features | OpenGL 3D rendering, backpropagation visualization |
| `3D_NeuroSim_Sept_7_settings.py` | Settings Display | For exploring parameters | Comprehensive 'V' key settings display |
| `JMRGameOfLifewith4Genes.py` | Classic Game of Life | Basic cellular automata (no neural net) | Original 4-gene implementation |

### 📊 **DATA PROCESSING & TRAINING SCRIPTS**
| File | Purpose | Input | Output | Usage |
|------|---------|-------|--------|-------|
| `JMR_fashion_mnist _to_cell_Oct_3_from_webdata.py` | **📥 DATASET CONVERTER** | Raw datasets | Cell format (.pkl) | Converts MNIST/Fashion-MNIST/CIFAR to training data |
| `JMR_pick_mnist _to_cell_Oct_23.py` | MNIST Processor | MNIST images | Cell training pairs | Alternative MNIST processing with plots |
| `importMNEST_Save_local.py` | Raw MNIST Downloader | None | Raw MNIST files | Downloads MNIST directly from source |

### 🔧 **UTILITIES & HELPERS**
| File | Purpose | Function | Notes |
|------|---------|----------|-------|
| `get_help_defs.py` | **📚 HELP SYSTEM** | Interactive documentation | Gene definitions, controls, network explanation |
| `timetestclip.py` | Performance Test | Sigmoid optimization | Benchmarks different activation functions |
| `visualization/render_3d_backprop.py` | 3D Rendering | OpenGL helpers | 3D visualization utilities |
| `requirements.txt` | **📦 DEPENDENCIES** | Python packages | Version-pinned requirements |
| `.gitignore` | **🗂️ GIT EXCLUSIONS** | Repository cleanup | Excludes artifacts, virtual env, data files |

### 📁 **DATA DIRECTORIES**
| Directory | Contents | Purpose | Size |
|-----------|----------|---------|------|
| `saved_states/` | `.pkl`, `.txt`, `.png` | **💾 NETWORK CHECKPOINTS** | Training snapshots, experiments |
| `CIFAR_10/` | `5000×.pkl` files | **📊 TRAINING DATA** | Preprocessed CIFAR-10 dataset |
| `CIFAR_10_PLOTS/` | `.png` plots | **📈 TRAINING VISUALS** | Learning progress charts |
| `June 2023 screen shots/` | Screenshots, videos | **📹 DEVELOPMENT DOCS** | Original development recordings |
| `neuro life sim movies and images/` | Videos, images | **🎬 DEMONSTRATIONS** | Simulation examples |
| `path/` | Older version | **📁 ARCHIVE** | Previous implementation |

### 📄 **DOCUMENTATION & ASSETS**
| File | Purpose | Format | Content |
|------|---------|--------|---------|
| `README.md` | **📖 MAIN GUIDE** | Markdown | This comprehensive documentation |
| `LICENSE` | **⚖️ LEGAL** | Text | MIT License terms |
| `neurogeneticsgameoflifedocs` | **📋 OVERVIEW** | Text | Technical project description |
| `JMRLIFEHTML.html` | **🌐 WEB DOCS** | HTML | HTML-formatted documentation |
| `Numbers_mnist_200.png` | **🖼️ MNIST VISUAL** | Image | 200×200 MNIST digit samples |
| `Fashion plot_1_2023-10-03.png` | **📊 FASHION RESULTS** | Image | Fashion-MNIST training plot |

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.8+** with pip
- **OpenGL support** for 3D visualization (included with most desktop systems)
- **Display environment** for Pygame GUI
- **Git** for cloning the repository

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/JMRGeneticsGameOfLife.git
   cd JMRGeneticsGameOfLife
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Linux/macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Obtain training data** (see Data Acquisition section below)

### Dependencies

| Package | Purpose |
|---------|---------|
| `pygame` | 2D visualization and user interface |
| `numpy` | Numerical computations |
| `Pillow` | Image processing |
| `tensorflow` | Dataset loading (MNIST, CIFAR) |
| `matplotlib` | Plotting and data visualization |
| `PyOpenGL` | 3D visualization |

## 🚀 Usage

### Running the Main Simulation

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Run the latest version
python 3D_NeuroSim_Feb_4_25_list_of_cells.py
```

**On first run**, you'll be prompted for network parameters:
- Number of layers (default: 16)
- Weights per cell (9, 25, 49, or 81)
- Learning rate, bias range, etc.

### Alternative Versions

```bash
# For 3D visualization features
python 3D_NeuroSim_Oct_29_back3D_visualization_cell.py

# For classic cellular automata (no neural network)
python JMRGameOfLifewith4Genes.py
```

## 📊 Data Acquisition Guide

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` | Toggle simulation running |
| `T` | Toggle training mode |
| `B` | Toggle backpropagation |
| `F` | Forward propagation |
| `R` | Reverse propagation direction |
| `3` | Toggle 3D view |
| `4` | Backprop 3D visualization |
| `K` | Toggle weight change view in 3D |
| `G` | Toggle gene/protein display |
| `V` | View all settings |
| `H` | Scroll through help screens |
| `L` | Load saved layers |
| `S` | Save current layers |
| `M` | Load MNIST training data |
| `E` | Enter global network parameters |
| `W` | Reset gradients and charges |
| `X` | Reset weights/biases to random |
| `P` | Toggle pruning |
| `C` | Set charge/gradient pruning level |
| `ESC` | Exit 3D view |

### Mouse Controls

- **Left Click/Drag**: Place cells
- **Right Click**: Display cell info and statistics
- **Mouse Wheel (3D)**: Zoom in/out
- **Mouse Drag (3D)**: Rotate view

## 🧬 Gene System

Each cell contains 9 genes that control its behavior:

| Gene | Index | Name | Description |
|------|-------|------|-------------|
| OT | 0 | Overcrowding Tolerance | Max neighbors before cell dies |
| IT | 1 | Isolation Tolerance | Min neighbors needed to survive |
| BT | 2 | Birth Threshold | Exact neighbor count for reproduction |
| MR | 3 | Mutation Rate | Probability of gene mutation |
| WPC | 4 | Weights Per Cell | Size of dendritic weight matrix (9, 25, 49, 81) |
| BR | 5 | Bias Range | Range for random bias initialization |
| AWC | 6 | Average Weights/Cell | Expected number of incoming connections |
| CD | 7 | Charge Delta | Threshold for pruning protection |
| WD | 8 | Weight Decay | Regularization factor |

### Survival Rules

A cell survives if: `IT <= neighbor_count <= OT`

A new cell is born if: `neighbor_count == BT` (inherits genes from parents with possible mutation)

## 🏗️ Architecture

### Dendritic Weight Matrix

Each cell's weight matrix represents incoming synaptic connections from its receptive field:

```
Relative Position (dx, dy) for reach=1:
    (-1,-1)  (-1, 0)  (-1, 1)
    ( 0,-1)  ( 0, 0)  ( 0, 1)
    ( 1,-1)  ( 1, 0)  ( 1, 1)

Flattened weight indices:
    [ 0, 1, 2 ]
    [ 3, 4, 5 ]
    [ 6, 7, 8 ]

Weight Index Formula: (dx + reach) * matrix_size + (dy + reach)
```

### Forward Pass

1. Input data loaded into Layer 0 (28×28 grid for MNIST)
2. Each cell computes weighted sum of charges from upper layer cells
3. Activation applied (ReLU or sigmoid)
4. Output compared against target in final layer

### Backpropagation

1. Error computed at output layer
2. Error signals propagate backward through layers
3. Weights updated based on: `Δw = learning_rate × error × upper_cell_charge`
4. Optional weight decay for regularization

## Results

The system has achieved:
- **100/100** accuracy on MNIST test set (8 layers, 49 weights/cell)
- **98/100** accuracy on Fashion-MNIST
- Demonstrated dynamic network adaptation during training

## Development History

- **May 2023**: Initial cellular automata implementation
- **July 2023**: MNIST integration and forward propagation
- **Sept 2023**: Full backpropagation, variable layers, 100% MNIST accuracy
- **Oct 2024**: 3D visualization, gene display, gradient pruning
- **Feb 2025**: Documentation, active cell list optimization

## Related Work

This project draws inspiration from:
- Conway's Game of Life
- Neuroevolution algorithms (NEAT)
- Hebbian learning
- Biological neural plasticity

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## License

*"Cellular Neural Networks: A Bio-Inspired Approach to Machine Learning"*

---

**Author:** Jonathan Marc Rothberg
