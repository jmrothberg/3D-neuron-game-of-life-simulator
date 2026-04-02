"""SimConfig dataclass and constants for the neural simulator."""
from dataclasses import dataclass

# Color constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
INDIGO = (13, 0, 184)
VIOLET = (179, 0, 255)
PINK = (255, 192, 203)
LIGHT_GRAY = (200, 200, 200)

COLORS = [INDIGO, ORANGE, PINK, YELLOW, GREEN, RED, VIOLET, BLUE]

# Window layout constants
EXTENDED_WINDOW_WIDTH = 1508
WINDOW_WIDTH = 1008
WINDOW_HEIGHT = 1008
WINDOW_EXTENSION = 100
EXTENDED_WINDOW_HEIGHT = WINDOW_HEIGHT + WINDOW_EXTENSION
HELP_PANEL_WIDTH = 500
MAIN_SURFACE_WIDTH = EXTENDED_WINDOW_WIDTH - HELP_PANEL_WIDTH

CELL_SIZE = 9
WIDTH = WINDOW_WIDTH // CELL_SIZE // 4   # 28
HEIGHT = WINDOW_HEIGHT // CELL_SIZE // 4  # 28

UPPER_ALLELE_LIMIT = 28
ARRAY_LAYERS = 16

FASHION_LABELS = "0 T-shirt/top | 1 Trouser | 2 Pullover | 3 Dress | 4 Coat | 5 Sandal | 6 Shirt | 7 Sneaker | 8 Bag | 9 Ankle boot"


@dataclass
class SimConfig:
    """Configuration parameters -- changed only by explicit user input (E/I/C keys)."""
    # Network architecture
    num_layers: int = 8
    length_of_dendrite: int = 1
    weight_matrix: int = 3           # derived: 2*length_of_dendrite + 1
    number_of_weights: int = 9       # derived: weight_matrix^2

    # Cell genetics
    mutation_rate: int = 10
    lower_allele_range: int = 2
    upper_allele_range: int = 15
    autonomous_network_genes: bool = False

    # Training hyperparameters (these are global defaults; in autonomous mode,
    # genes 4-11 override per-cell: WG, BR, AW, CD, WD, LR, GT, AS)
    learning_rate: float = 0.01          # Gene 9 default (LR: synaptic plasticity speed)
    bias_range: float = 0.01             # Gene 5 default (BR: initial bias magnitude)
    avg_weights_cell: int = 5            # Gene 6 default (AW: fan-in for He init)
    weight_decay: float = 1e-6           # Gene 8 default (WD: L2 regularization)
    charge_delta: float = 0.001          # Gene 7 default (CD: activity significance threshold)
    gradient_threshold: float = 0.0000001  # Gene 10 default (GT: pruning survival sensitivity)
    gradient_clip_range: int = 1         # Max gradient magnitude (not yet a gene)
    weight_change_threshold: float = 0.005  # Not yet a gene
    activation_slope: float = 0.1        # Gene 11 default (AS: leaky ReLU negative slope)
    # FUTURE AUTONOMY CANDIDATES: gradient_clip_range, weight_change_threshold,
    # charge clipping range (hardcoded ±10), prune_logic (AND/OR),
    # charge_change_protection, memory window (how_much_training_data)

    # Data
    how_much_training_data: int = 20
    start_index: int = 0

    # Constants
    epsilon: float = 1e-8

    def update_derived(self):
        """Recompute derived fields after length_of_dendrite changes."""
        self.weight_matrix = 2 * self.length_of_dendrite + 1
        self.number_of_weights = self.weight_matrix * self.weight_matrix


def set_default_values():
    """Return a fresh SimConfig with default values."""
    config = SimConfig()
    config.update_derived()
    return config
