"""Cell class -- the fundamental unit of the neural simulator.
Weights are stored IN the cell's dendrites (self.weights), not in separate layers.
"""
import numpy as np
import random

from neurosim.config import COLORS, WIDTH, HEIGHT


class Cell:
    """A neuron-cell with genes, proteins (charge/error/bias/weights), and dendrites."""

    # Class-level config reference -- set once at startup, NOT pickled.
    _config = None

    @classmethod
    def set_config(cls, config):
        """Set the shared config reference. Call at startup and after loading."""
        cls._config = config

    def __init__(self, layer, x, y, weights_per_cell_possible, Bias_Range, Avg_Weights_Cell,
                 charge_delta, weight_decay, mutation_rate, genes=None):
        cfg = Cell._config
        epsilon = 1e-8

        # Coordinate validation
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            self.x = min(max(0, x), WIDTH - 1)
            self.y = min(max(0, y), HEIGHT - 1)
        else:
            self.x = x
            self.y = y
        self.layer = layer

        if genes is None:
            self.initalize_all_genes()
            self.initalize_breeding_genes()
            self.initalize_network_genes(weights_per_cell_possible, Bias_Range, Avg_Weights_Cell,
                                         charge_delta, weight_decay, mutation_rate)
        else:
            self.genes = genes.copy()

        self.color_genes()
        self.initialize_network_proteins()
        self.initalize_cell_memory()
        self.color_proteins()

    def initalize_cell_memory(self):
        self.forward_charges = []
        self.reverse_charges = []
        self.max_charge_diff_forward = 0
        self.max_charge_diff_reverse = 0
        self.significant_charge_change_forward = False
        self.significant_charge_change_reverse = False
        self.number_of_upper_layer_cells = 0
        self.number_of_lower_layer_cells = 0
        self.gradient_history = []
        self.avg_gradient_magnitude = 0
        self.significant_gradient_change = False

    def initalize_all_genes(self):
        self.genes = [0] * 9
        self.colors = [0] * 9
        self.protein_colors = [0] * 9

    def initalize_breeding_genes(self):
        cfg = Cell._config
        lo = cfg.lower_allele_range if cfg else 2
        hi = cfg.upper_allele_range if cfg else 15
        self.genes[0] = random.randint(lo, hi)
        self.genes[1] = random.randint(lo, hi)
        self.genes[2] = random.randint(lo, hi)
        if self.genes[0] < self.genes[1]:
            self.genes[0], self.genes[1] = self.genes[1], self.genes[0]
        self.color_genes()

    def initalize_network_genes(self, weights_per_cell_possible, Bias_Range, Avg_Weights_Cell,
                                charge_delta, weight_decay, mutation_rate, cells_array=None):
        cfg = Cell._config
        autonomous = cfg.autonomous_network_genes if cfg else False

        if not autonomous:
            self.genes[3] = mutation_rate
            self.genes[4] = int(weights_per_cell_possible)
            self.genes[5] = Bias_Range
            self.genes[6] = Avg_Weights_Cell
            self.genes[7] = charge_delta
            self.genes[8] = weight_decay
        else:
            self.genes[3] = np.random.randint(0, 100)
            self.genes[4] = (np.random.randint(1, 4) * 2 + 1) ** 2
            self.genes[5] = np.random.choice([0.001, 0.01])
            self.reach = (int(np.sqrt(self.genes[4])) - 1) // 2
            # Lamarcian: get actual upper layer cells count
            if cells_array is not None:
                upper = self.get_upper_layer_cells(cells_array, self.reach)
                if len(upper) > 0:
                    self.genes[6] = len(upper)
                else:
                    self.genes[6] = Avg_Weights_Cell
            else:
                self.genes[6] = Avg_Weights_Cell
            self.genes[7] = np.random.uniform(0.000001, 0.01)
            self.genes[8] = np.random.uniform(1e-6, 1e-4)

        self.color_genes()

    def initialize_network_proteins(self):
        epsilon = 1e-8
        self.charge = 0
        self.gradient = 0
        self.error = epsilon
        self.reach = (int(np.sqrt(self.genes[4])) - 1) // 2
        self.weights = np.clip(
            np.random.randn(int(self.genes[4])) / (np.sqrt(self.genes[6]) + 1e-8),
            -0.8, 0.8
        )
        self.bias = np.random.uniform(0, self.genes[5])

    def color_genes(self):
        self.colors = [
            tuple(max(0, min(int(gene * 255 // 15), 255)) for color_component in color)
            for gene, color in zip(self.genes[:4], COLORS)
        ]
        gene_4_scaled = max(0, min(int(((self.genes[4] - 9) / (81 - 9)) * 255), 255))
        gene_5_scaled = max(0, min(int(self.genes[5] * 255), 255))
        gene_6_scaled = max(0, min(int(((self.genes[6] - 5) / (30 - 5)) * 255), 255))
        gene_7_scaled = max(0, min(int(((self.genes[7] - 0.0001) / (0.01 - 0.0001)) * 255), 255))
        gene_8_scaled = max(0, min(int(((self.genes[8] - 1e-6) / (1e-4 - 1e-6)) * 255), 255))
        self.colors.extend([
            (gene_4_scaled, 0, gene_4_scaled),
            (0, gene_5_scaled, gene_5_scaled),
            (gene_6_scaled, 0, 0),
            (0, gene_7_scaled, 0),
            (0, 0, gene_8_scaled)
        ])

    def color_proteins(self):
        self.protein_colors = [0] * 9
        epsilon = 1e-8

        bias_range = self.genes[5]
        bias_scaled = max(0, min(int((self.bias / (bias_range + epsilon)) * 255), 255))
        self.protein_colors[0] = (bias_scaled, 0, bias_scaled)

        max_forward_charge_scaled = max(0, min(int((self.max_charge_diff_forward / (1.0 + epsilon)) * 255), 255))
        self.protein_colors[1] = (max_forward_charge_scaled, 0, 0)

        max_reverse_charge_scaled = max(0, min(int((self.max_charge_diff_reverse / (1.0 + epsilon)) * 255), 255))
        self.protein_colors[2] = (0, 0, max_reverse_charge_scaled)

        charge_scaled = max(0, min(int(self.charge * 255), 255))
        self.protein_colors[4] = (charge_scaled, 0, 0)

        error_magnitude = abs(self.error + epsilon)
        error_scaled = max(0, min(int(255 * (np.log10(error_magnitude) + 3) / 2), 255))
        self.protein_colors[6] = (0, 0, error_scaled)

        mean_weight = np.mean(np.abs(self.weights))
        weight_scaled = max(0, min(int((mean_weight / (1.0 + epsilon)) * 255), 255))
        self.protein_colors[7] = (0, weight_scaled, 0)

        gradient = self.charge * self.error
        gradient_scaled = max(0, min(int(np.log(abs(gradient + epsilon) * 55)), 255))
        self.protein_colors[8] = (0, 0, gradient_scaled)

        self.protein_colors[3] = (0, 0, 0)
        self.protein_colors[5] = (0, 0, 0)

    def __setstate__(self, state):
        """Restore from pickle with backward compatibility."""
        self.__dict__.update(state)

        if not hasattr(self, 'genes') or len(self.genes) < 9:
            self.genes = [0] * 9
            if not hasattr(self.genes[3], '__int__'):
                self.initalize_breeding_genes()
            if not hasattr(self.genes[4], '__int__'):
                cfg = Cell._config
                if cfg:
                    self.initalize_network_genes(cfg.number_of_weights, cfg.bias_range,
                                                 cfg.avg_weights_cell, cfg.charge_delta,
                                                 cfg.weight_decay, cfg.mutation_rate)
            if not hasattr(self.genes[5], '__float__'):
                self.genes[5] = Cell._config.bias_range if Cell._config else 0.01
            if not hasattr(self.genes[6], '__float__'):
                self.genes[6] = Cell._config.avg_weights_cell if Cell._config else 5
            if not hasattr(self.genes[7], '__float__'):
                self.genes[7] = Cell._config.charge_delta if Cell._config else 0.001
            if not hasattr(self.genes[8], '__float__'):
                self.genes[8] = Cell._config.weight_decay if Cell._config else 1e-6

        if not hasattr(self, 'reach'):
            self.reach = (int(np.sqrt(self.genes[4])) - 1) // 2
        if not hasattr(self, 'gradient'):
            self.gradient = 0
        if not hasattr(self, 'x'):
            self.x = 0
        if not hasattr(self, 'y'):
            self.y = 0
        if not hasattr(self, 'forward_charges'):
            self.forward_charges = []
        if not hasattr(self, 'reverse_charges'):
            self.reverse_charges = []
        if not hasattr(self, 'max_charge_diff_forward'):
            self.max_charge_diff_forward = 0
        if not hasattr(self, 'max_charge_diff_reverse'):
            self.max_charge_diff_reverse = 0
        if not hasattr(self, 'significant_charge_change_forward'):
            self.significant_charge_change_forward = False
        if not hasattr(self, 'significant_charge_change_reverse'):
            self.significant_charge_change_reverse = False
        if not hasattr(self, 'gradient_history'):
            self.gradient_history = []
        if not hasattr(self, 'avg_gradient_magnitude'):
            self.avg_gradient_magnitude = 0
        if not hasattr(self, 'significant_gradient_change'):
            self.significant_gradient_change = False

        if np.isnan(self.weights).any():
            print(f"Warning: NaN weights detected in cell at ({self.x}, {self.y}, {self.layer})")
            self.weights = np.random.randn(self.genes[4]) / np.sqrt(self.genes[6])

        if not hasattr(self, 'protein_colors'):
            self.color_proteins()

    # ---- Forward pass: weights are in THIS cell's dendrites ----

    def compute_total_charge(self, upper_layer_cells, reach):
        cfg = Cell._config
        if cfg and cfg.autonomous_network_genes:
            reach = self.reach
            NUMBER_OF_WEIGHTS = self.genes[4]
            WEIGHT_MATRIX = int(np.sqrt(NUMBER_OF_WEIGHTS))
        else:
            WEIGHT_MATRIX = 2 * reach + 1
        charge = 0
        for dx, dy, cell in upper_layer_cells:
            weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
            if 0 <= weight_index < len(self.weights):
                charge += cell.charge * self.weights[weight_index]
        self.charge = np.clip(charge, -10, 10)

    def compute_total_charge_reverse(self, lower_layer_cells, reach):
        cfg = Cell._config
        if cfg and cfg.autonomous_network_genes:
            pass  # each cell below uses its own reach
        else:
            WEIGHT_MATRIX = 2 * reach + 1
        charge = 0
        charge -= self.bias
        for dx, dy, cell in lower_layer_cells:
            if cfg and cfg.autonomous_network_genes:
                cell_reach = cell.reach
                cell_weight_matrix = int(np.sqrt(cell.genes[4]))
            else:
                cell_reach = reach
                cell_weight_matrix = WEIGHT_MATRIX
            relative_x = -dx + cell_reach
            relative_y = -dy + cell_reach
            weight_index = relative_y * cell_weight_matrix + relative_x
            if 0 <= weight_index < len(cell.weights):
                charge += cell.charge * cell.weights[weight_index]
        self.charge = np.clip(charge, -10, 10)

    # ---- Activation functions ----

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1.0, 0.1)  # Leaky ReLU derivative

    # ---- Neighbor lookups ----

    def get_upper_layer_cells(self, cells, reach):
        cfg = Cell._config
        if cfg and cfg.autonomous_network_genes:
            reach = self.reach
        upper_layer_cells = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                new_x, new_y = self.x + dx, self.y + dy
                if 0 <= new_x < cells.shape[0] and 0 <= new_y < cells.shape[1] and cells[new_x, new_y, self.layer - 1] is not None:
                    upper_layer_cells.append((dx, dy, cells[new_x, new_y, self.layer - 1]))
        self.number_of_upper_layer_cells = len(upper_layer_cells)
        return upper_layer_cells

    def get_layer_below_cells(self, cells, reach, max_reach_below=None):
        cfg = Cell._config
        if cfg and cfg.autonomous_network_genes:
            reach = self.reach
        layer_below_cells = []
        # Guard: if self.layer+1 is out of bounds, return empty
        if self.layer + 1 >= cells.shape[2]:
            self.number_of_lower_layer_cells = 0
            return layer_below_cells
        # Use cached max_reach if provided, otherwise compute it
        if max_reach_below is None:
            max_possible_reach = max(
                (cell.reach for row in cells[:, :, self.layer + 1] for cell in row if cell is not None),
                default=0
            )
        else:
            max_possible_reach = max_reach_below
        for dx in range(-max_possible_reach, max_possible_reach + 1):
            for dy in range(-max_possible_reach, max_possible_reach + 1):
                new_x, new_y = self.x + dx, self.y + dy
                if 0 <= new_x < cells.shape[0] and 0 <= new_y < cells.shape[1]:
                    cell_below = cells[new_x, new_y, self.layer + 1]
                    if cell_below is not None:
                        cell_below_reach = cell_below.reach if (cfg and cfg.autonomous_network_genes) else reach
                        if abs(dx) <= cell_below_reach and abs(dy) <= cell_below_reach:
                            layer_below_cells.append((dx, dy, cell_below))
        self.number_of_lower_layer_cells = len(layer_below_cells)
        return layer_below_cells

    # ---- Backpropagation: error traces back through dendrite weights ----
    # NOTE: reversed_index = len(weights) - 1 - weight_index is mathematically
    # equivalent to indexing at (-dx, -dy). DO NOT CHANGE THIS.

    def compute_error_signal(self, desired_output=None, connected_cells=None, reach=None):
        cfg = Cell._config
        epsilon = cfg.epsilon if cfg else 1e-8
        if cfg and cfg.autonomous_network_genes:
            reach = self.reach
        error_signal = epsilon

        # Output layer: error = (predicted - desired) * relu_derivative
        if desired_output is not None:
            error_signal = (self.charge - desired_output) * self.relu_derivative(self.charge)
            self.error = np.clip(error_signal, -10, 10)

        # Hidden layers: backpropagate error from layer below
        elif connected_cells is not None and reach is not None:
            for dx, dy, cell in connected_cells:
                if cfg and cfg.autonomous_network_genes:
                    cell_reach = cell.reach
                    cell_weight_matrix = int(np.sqrt(cell.genes[4]))
                else:
                    cell_reach = reach
                    cell_weight_matrix = 2 * reach + 1

                weight_index = (dx + cell_reach) * cell_weight_matrix + (dy + cell_reach)
                reversed_index = len(cell.weights) - 1 - weight_index

                if 0 <= reversed_index < len(cell.weights):
                    error_signal += cell.error * cell.weights[reversed_index] * self.relu_derivative(self.charge)

            self.error = np.clip(error_signal, -10, 10)
        else:
            self.error = epsilon

    def update_weights_and_bias(self, connected_cells, learning_rate, reach, weight_decay):
        cfg = Cell._config
        gradient_clip_range = cfg.gradient_clip_range if cfg else 1
        if cfg and cfg.autonomous_network_genes:
            reach = self.reach
            NUMBER_OF_WEIGHTS = self.genes[4]
            WEIGHT_MATRIX = int(np.sqrt(NUMBER_OF_WEIGHTS))
            weight_decay = self.genes[8]
        else:
            WEIGHT_MATRIX = 2 * reach + 1

        epsilon = cfg.epsilon if cfg else 1e-8
        if self.error is None:
            self.error = epsilon

        for dx, dy, cell in connected_cells:
            gradient = self.error * cell.charge
            gradient = np.clip(gradient, -gradient_clip_range, gradient_clip_range)
            self.gradient = gradient
            self.update_gradient_importance(gradient)
            weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
            if weight_index < len(self.weights):
                self.weights[weight_index] -= learning_rate * gradient + weight_decay * self.weights[weight_index]

        self.bias -= learning_rate * self.error

    # ---- Charge / gradient tracking ----

    def update_charge(self, new_charge, direction):
        cfg = Cell._config
        how_much = cfg.how_much_training_data if cfg else 20
        if direction == "forward":
            self.forward_charges.append(new_charge)
            if len(self.forward_charges) > how_much:
                self.forward_charges.pop(0)
            self.max_charge_diff_forward = max(self.forward_charges) - min(self.forward_charges)
            if self.max_charge_diff_forward > self.genes[7]:
                self.significant_charge_change_forward = True
        elif direction == "reverse":
            self.reverse_charges.append(new_charge)
            if len(self.reverse_charges) > how_much:
                self.reverse_charges.pop(0)
            self.max_charge_diff_reverse = max(self.reverse_charges) - min(self.reverse_charges)
            if self.max_charge_diff_reverse > self.genes[7]:
                self.significant_charge_change_reverse = True
        self.charge = new_charge

    def update_gradient_importance(self, new_gradient):
        cfg = Cell._config
        how_much = cfg.how_much_training_data if cfg else 20
        grad_threshold = cfg.gradient_threshold if cfg else 1e-7
        self.gradient_history.append(abs(new_gradient))
        if len(self.gradient_history) > how_much:
            self.gradient_history.pop(0)
        self.avg_gradient_magnitude = np.mean(self.gradient_history)
        if self.avg_gradient_magnitude > grad_threshold:
            self.significant_gradient_change = True

    def reset_directional_charge_history(self, direction):
        if direction == "+++++>>>>>":
            self.forward_charges.clear()
            self.max_charge_diff_forward = 0
            self.significant_charge_change_forward = False
        elif direction == "<<<<<-----":
            self.reverse_charges.clear()
            self.max_charge_diff_reverse = 0
            self.significant_charge_change_reverse = False
        self.gradient_history.clear()
        self.avg_gradient_magnitude = 0
        self.significant_gradient_change = False

    def reset_gradient_change(self):
        cfg = Cell._config
        epsilon = cfg.epsilon if cfg else 1e-8
        self.significant_gradient_change = False
        self.gradient_history.clear()
        self.gradient = 0
        self.error = epsilon

    # ---- Weight remapping (when dendrite size changes via mutation or user input) ----

    def remap_weights(self, reach):
        old_matrix = int(np.sqrt(len(self.weights)))
        new_matrix = 2 * reach + 1

        if old_matrix == new_matrix:
            return

        if len(self.weights) not in [9, 25, 49, 81, 121, 169, 225, 289, 361, 441, 529, 625]:
            print("bad mutant gene repaired")
            self.weights = np.random.randn(new_matrix ** 2) / (np.sqrt(self.genes[6]) + 1e-8)
            self.genes[4] = len(self.weights)
            return

        old_grid = np.reshape(self.weights, (old_matrix, old_matrix))
        new_grid = np.zeros((new_matrix, new_matrix))

        old_center = old_matrix // 2
        new_center = new_matrix // 2

        copy_range = min(old_matrix, new_matrix)
        start_old = old_center - copy_range // 2
        start_new = new_center - copy_range // 2

        new_grid[start_new:start_new + copy_range, start_new:start_new + copy_range] = \
            old_grid[start_old:start_old + copy_range, start_old:start_old + copy_range]

        if new_matrix > old_matrix:
            mask = new_grid == 0
            num_new_weights = np.sum(mask)
            new_weights = np.random.randn(num_new_weights) / (np.sqrt(self.genes[6]) + 1e-8)
            new_grid[mask] = new_weights

        self.weights = new_grid.flatten()
        self.genes[4] = len(self.weights)

    # ---- Cell autonomy methods (Phase 4) ----

    def forward(self, cells_array):
        """Self-contained forward pass for this cell."""
        cfg = Cell._config
        reach = self.reach if (cfg and cfg.autonomous_network_genes) else cfg.length_of_dendrite
        upper_cells = self.get_upper_layer_cells(cells_array, reach)
        self.compute_total_charge(upper_cells, reach)
        self.update_charge(self.charge, "forward")

    def backward(self, cells_array, learning_rate=None, max_reach_below=None):
        """Self-contained backward pass for this cell."""
        cfg = Cell._config
        lr = learning_rate or cfg.learning_rate
        reach = self.reach if (cfg and cfg.autonomous_network_genes) else cfg.length_of_dendrite
        wd = self.genes[8] if (cfg and cfg.autonomous_network_genes) else cfg.weight_decay

        # Compute error signal
        if self.layer == cfg.num_layers - 2:
            target_cell = cells_array[self.x, self.y, cfg.num_layers - 1]
            if target_cell is not None:
                self.compute_error_signal(desired_output=target_cell.charge)
        else:
            below_cells = self.get_layer_below_cells(cells_array, reach, max_reach_below=max_reach_below)
            self.compute_error_signal(connected_cells=below_cells, reach=reach)

        # Update weights
        upper_cells = self.get_upper_layer_cells(cells_array, reach)
        self.update_weights_and_bias(upper_cells, lr, reach, wd)

    def should_i_die(self, prune, gradient_prune, prune_logic):
        """Determine if this cell should die based on pruning rules."""
        if gradient_prune and not self.significant_gradient_change:
            return True
        if prune:
            if prune_logic == "AND":
                if not (self.significant_charge_change_forward and self.significant_charge_change_reverse):
                    return True
            elif prune_logic == "OR":
                if not (self.significant_charge_change_forward or self.significant_charge_change_reverse):
                    return True
        return False

    def should_i_die_genetic(self, num_alive, charge_change_protection):
        """Conway-style death check, respecting charge protection."""
        if charge_change_protection:
            protected = (self.significant_charge_change_forward or
                         self.significant_charge_change_reverse or
                         self.significant_gradient_change)
            if protected:
                return False
        return num_alive <= self.genes[1] or num_alive >= self.genes[0]

    # ---- Validation (Phase 3) ----

    def validate(self):
        """NaN/sanity checks. Returns list of issue strings (empty = OK)."""
        issues = []
        if np.isnan(self.weights).any():
            issues.append(f"NaN weights at ({self.x},{self.y},{self.layer})")
            self.weights = np.clip(
                np.random.randn(int(self.genes[4])) / (np.sqrt(self.genes[6]) + 1e-8),
                -0.8, 0.8
            )
        if np.isnan(self.charge):
            issues.append(f"NaN charge at ({self.x},{self.y},{self.layer})")
            self.charge = 0.0
        if np.isnan(self.error):
            issues.append(f"NaN error at ({self.x},{self.y},{self.layer})")
            self.error = 1e-8
        if np.isnan(self.bias):
            issues.append(f"NaN bias at ({self.x},{self.y},{self.layer})")
            self.bias = 0.0
        if len(self.weights) != int(self.genes[4]):
            issues.append(f"Weight/gene mismatch at ({self.x},{self.y},{self.layer}): "
                          f"len(weights)={len(self.weights)} genes[4]={self.genes[4]}")
        expected_reach = (int(np.sqrt(self.genes[4])) - 1) // 2
        if self.reach != expected_reach:
            issues.append(f"Reach mismatch at ({self.x},{self.y},{self.layer}): "
                          f"reach={self.reach} expected={expected_reach}")
            self.reach = expected_reach
        return issues

    # ---- Display ----

    def __str__(self):
        weights = [f'{w:.4f}' for w in self.weights]
        weights_with_newlines = []
        for i in range(0, len(weights), 7):
            if i > 0:
                weights_with_newlines.append('\n')
            weights_with_newlines.extend(weights[i:i + 7])
        weights_str = ', '.join(weights_with_newlines)
        bias = f'{self.bias:.4f}'
        error = f'{self.error:.4e}'
        charge = f'{self.charge:.4f}'
        gradient = f'{self.gradient:.4f}'
        reach = f'{self.reach}'
        max_charge_diff_forward = f'{self.max_charge_diff_forward:.4f}'
        significant_charge_change_forward = f'{self.significant_charge_change_forward}'
        max_charge_diff_reverse = f'{self.max_charge_diff_reverse:.4f}'
        significant_charge_change_reverse = f'{self.significant_charge_change_reverse}'
        gradient_average = f'{self.avg_gradient_magnitude:.4f}'
        significant_gradient_change = f'{self.significant_gradient_change}'
        return (
            f"Neuron: layer={self.layer} x={self.x} y={self.y}\n"
            f"Genes:\n"
            f"  OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]},\n"
            f"  WG={self.genes[4]}, BR={self.genes[5]}, AW={self.genes[6]}, CD={self.genes[7]}, WD={self.genes[8]}\n"
            f"charge={charge}, error={error}, bias={bias}, gradient={gradient}\n"
            f"max_charge_diff_forward={max_charge_diff_forward}, significant_charge_change_forward={significant_charge_change_forward}\n"
            f"max_charge_diff_reverse={max_charge_diff_reverse}, significant_charge_change_reverse={significant_charge_change_reverse}\n"
            f"gradient_average={gradient_average}, significant_gradient_change={significant_gradient_change}\n"
            f"reach={reach}\n"
            f"weights={weights_str}"
        )
