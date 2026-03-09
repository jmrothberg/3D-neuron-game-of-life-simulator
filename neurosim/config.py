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

    # Training hyperparameters
    learning_rate: float = 0.01
    bias_range: float = 0.01
    avg_weights_cell: int = 5
    weight_decay: float = 1e-6
    charge_delta: float = 0.001
    gradient_threshold: float = 0.0000001
    gradient_clip_range: int = 1
    weight_change_threshold: float = 0.005
    activation_slope: float = 0.1        # Leaky ReLU negative slope (gene 11 default)

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
