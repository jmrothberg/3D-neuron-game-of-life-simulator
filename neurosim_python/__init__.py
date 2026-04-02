# neurosim - Bio-inspired Neural Network Simulator
# Weights live in cell dendrites, not separate layers.
#
# Module structure:
#   config.py           - SimConfig dataclass + constants
#   state.py            - SimState dataclass
#   cell.py             - Cell class (genes, proteins, dendrite weights)
#   training.py         - Forward/backward propagation, training loop
#   evolution.py        - Andromida mode, pruning, mutation
#   io_manager.py       - Save/load files, MNIST data loading
#   visualization.py    - 2D cell rendering, statistics
#   visualization_3d.py - OpenGL 3D rendering
#   ui.py               - Input dialogs, side panel
#   telemetry.py        - Per-layer telemetry and NaN detection
#   main.py             - Event loop, entry point

from neurosim.config import SimConfig
from neurosim.state import SimState
from neurosim.cell import Cell
