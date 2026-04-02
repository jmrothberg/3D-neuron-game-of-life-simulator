"""SimState dataclass for mutable simulation state."""
from dataclasses import dataclass, field
import numpy as np

from neurosim.config import WIDTH, HEIGHT, ARRAY_LAYERS


@dataclass
class SimState:
    """Mutable simulation state -- changes every frame/training cycle."""
    # The main cell array
    cells: np.ndarray = field(default=None)

    # Mode flags
    running: bool = False
    prune: bool = False
    gradient_prune: bool = False
    training_mode: bool = False
    andromida_mode: bool = False
    charge_change_protection: bool = True
    back_prop: bool = False
    training_data_loaded: bool = False
    display_updating: bool = True
    simulating: bool = True
    not_saved_yet: bool = True

    # Display state
    prune_logic: str = "OR"
    display: str = "proteins"
    direction_of_charge_flow: str = "+++++>>>>>"
    show_3d_view: bool = False
    show_backprop_view: bool = False
    show_training_stats: bool = False
    display_set: int = 0

    # Training metrics
    bingo_count: int = 0
    max_bingo_count: int = 0
    total_cells: int = 0
    total_loss: float = 0.0
    total_predictions: int = 0
    running_avg_loss: float = 0.0
    training_cycles: int = 0
    total_weights: float = 0.0
    total_weights_list: np.ndarray = field(default=None)
    points: list = field(default_factory=list)
    epochs: int = 1
    batch_size: int = 1

    # Training data (loaded from MNIST)
    training_data_layer_0: list = field(default_factory=list)
    training_data_num_layer_minus_1: list = field(default_factory=list)

    # 3D view state
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_angle: float = 0.0
    zoom: float = -15.0

    # UI state
    mouse_up: bool = True
    current_index: int = 0
    side_panel_text: list = field(default_factory=list)
    training_stats_buffer: dict = field(default_factory=dict)
    stats_update_frequency: int = 1

    # Timing
    timing: bool = False

    # Neighbor cache (for Phase 6 optimization)
    _max_reach_per_layer: dict = field(default_factory=dict)
    _cache_valid: bool = False

    # 3D visualization cache dirty flag
    _3d_dirty: bool = True

    def __post_init__(self):
        if self.cells is None:
            self.cells = np.full((WIDTH, HEIGHT, ARRAY_LAYERS), None, dtype=object)
        if self.total_weights_list is None:
            self.total_weights_list = np.zeros(20)

    def invalidate_neighbor_cache(self):
        self._max_reach_per_layer.clear()
        self._cache_valid = False
        self._3d_dirty = True

    def get_max_reach_for_layer(self, layer):
        """Cached max reach for a layer. Invalidated on birth/death."""
        if layer not in self._max_reach_per_layer:
            max_r = 0
            for x in range(self.cells.shape[0]):
                for y in range(self.cells.shape[1]):
                    cell = self.cells[x, y, layer]
                    if cell is not None:
                        max_r = max(max_r, cell.reach)
            self._max_reach_per_layer[layer] = max_r
        return self._max_reach_per_layer[layer]

    def reset_training_metrics(self):
        self.bingo_count = 0
        self.max_bingo_count = 0
        self.total_loss = 0.0
        self.total_predictions = 0
        self.running_avg_loss = 0.0
        self.training_cycles = 0
        self.points = []
