"""Cell class -- the fundamental unit of the neural simulator.

Each cell is an autonomous agent with two types of information:

  GENES (12 values) -- inherited, mostly stable parameters defining the cell's
  identity and structure.  Genes are the cell's *genotype*.  They change only
  through mutation or crossover during reproduction.

    Breeding genes (0-2):  control survival and reproduction (Game of Life rules)
      0  OT  Overcrowding Tolerance -- max neighbors before death
      1  IT  Isolation Tolerance    -- min neighbors before death
      2  BT  Birth Threshold        -- exact neighbor count to reproduce

    Network genes (3-11): control learning and connectivity
      3  MR  Mutation Rate           -- probability of gene mutation
      4  WG  Dendrite Size (Weights) -- number of synaptic weights (9,25,49,81)
      5  BR  Bias Range              -- initial bias magnitude
      6  AW  Fan-In (Avg Weights)    -- weight initialization scaling factor
      7  CD  Charge Delta            -- threshold for "significant" activity
      8  WD  Weight Decay            -- L2 regularization per cell
      9  LR  Learning Rate           -- synaptic plasticity speed (0.003-0.05)
     10  GT  Gradient Threshold      -- survival signal sensitivity for pruning (log-uniform 1e-8 to 1e-4)
     11  AS  Activation Slope        -- leaky ReLU negative slope (0.01-0.3, selectivity)

  PROTEINS (5 values) -- dynamic, mutable state that changes every training step.
  Proteins are the cell's *phenotype* -- the expressed behavior that results from
  genes interacting with the environment.

    charge   -- the cell's activation signal (like membrane potential)
    error    -- backpropagation error signal (like retrograde signaling)
    bias     -- offset to activation (like resting potential)
    weights  -- synaptic connection strengths (stored in dendrites, NOT in layer matrices)
    gradient -- most recent learning signal (like calcium/CaMKII activity)

Weights are stored IN the cell's dendrites (self.weights), not in separate layers.
"""
import numpy as np
import random

from neurosim.config import COLORS, WIDTH, HEIGHT


class Cell:
    """A neuron-cell with 12 genes, 5 proteins, and dendritic weights.

    Two timescales:
      - Fast: proteins change every training step (gradient descent)
      - Slow: genes change across generations (evolution and mutation)
    """

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
        # 12 genes: 0-2 breeding, 3-8 original network, 9-11 new autonomy genes
        self.genes = [0] * 12
        self.colors = [0] * 12
        self.protein_colors = [0] * 12

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
        """Initialize network genes (3-11).

        Two modes controlled by config.autonomous_network_genes:
          - Non-autonomous: all cells get the SAME values from global config
          - Autonomous: each cell gets RANDOM values, subject to evolution
        """
        cfg = Cell._config
        autonomous = cfg.autonomous_network_genes if cfg else False

        if not autonomous:
            # ---- Non-autonomous: stamp all network genes from global config ----
            self.genes[3] = mutation_rate                # MR: mutation frequency
            self.genes[4] = int(weights_per_cell_possible)  # WG: dendrite size
            self.genes[5] = Bias_Range                   # BR: bias initialization range
            self.genes[6] = Avg_Weights_Cell             # AW: fan-in for weight scaling
            self.genes[7] = charge_delta                 # CD: activity significance threshold
            self.genes[8] = weight_decay                 # WD: L2 regularization strength
            # New genes 9-11: stamp from global config
            self.genes[9] = cfg.learning_rate if cfg else 0.01       # LR: synaptic plasticity rate
            self.genes[10] = cfg.gradient_threshold if cfg else 1e-7  # GT: pruning survival sensitivity
            self.genes[11] = cfg.activation_slope if cfg else 0.1     # AS: leaky ReLU negative slope
        else:
            # ---- Autonomous: each cell gets its own random values ----
            # Gene 3 (MR): mutation frequency -- higher = more genetic drift
            self.genes[3] = np.random.randint(0, 100)
            # Gene 4 (WG): dendrite size -- determines receptive field
            # Like biological neurons with varying dendritic complexity
            self.genes[4] = (np.random.randint(1, 4) * 2 + 1) ** 2  # 9, 25, or 49
            # Gene 5 (BR): bias range -- resting potential variability
            self.genes[5] = np.random.choice([0.001, 0.01])
            self.reach = (int(np.sqrt(self.genes[4])) - 1) // 2
            # Gene 6 (AW): fan-in -- Lamarckian: count actual upstream connections
            # Like how synaptic normalization depends on actual input count
            if cells_array is not None:
                upper = self.get_upper_layer_cells(cells_array, self.reach)
                if len(upper) > 0:
                    self.genes[6] = len(upper)
                else:
                    self.genes[6] = Avg_Weights_Cell
            else:
                self.genes[6] = Avg_Weights_Cell
            # Gene 7 (CD): charge delta -- activity-dependent survival signal
            # Log-uniform: spans 4 orders of magnitude (1e-6 to 1e-2)
            self.genes[7] = 10 ** np.random.uniform(-6, -2)
            # Gene 8 (WD): weight decay -- synaptic protein turnover rate
            # Log-uniform: spans 2 orders of magnitude (1e-6 to 1e-4)
            self.genes[8] = 10 ** np.random.uniform(-6, -4)
            # Gene 9 (LR): learning rate -- synaptic plasticity speed
            # Like hippocampal (high) vs cortical (low) plasticity
            # Narrowed range: 0.003-0.05 avoids extreme instability or sluggishness
            self.genes[9] = np.random.uniform(0.003, 0.05)
            # Gene 10 (GT): gradient threshold -- survival signal sensitivity
            # Log-uniform: spans 4 orders of magnitude (1e-8 to 1e-4)
            # Without log-uniform, 99.99% of cells would get values near 1e-4
            self.genes[10] = 10 ** np.random.uniform(-8, -4)
            # Gene 11 (AS): activation slope -- neuron response curve
            # From near-ReLU (0.01, selective) to moderately leaky (0.3, permissive)
            # 0.3 max avoids extreme leakiness that weakens feature detection
            self.genes[11] = np.random.uniform(0.01, 0.3)

        self.color_genes()

    def initialize_network_proteins(self):
        """Initialize the 5 proteins from the cell's genes.

        Proteins are the dynamic state -- they change every training step.
        Genes set the structural constraints; proteins do the work.
        """
        epsilon = 1e-8
        # Protein 1: charge -- the cell's activation signal (like membrane potential)
        self.charge = 0
        # Protein 2: gradient -- most recent learning signal
        self.gradient = 0
        # Protein 3: error -- backprop error signal (like retrograde signaling)
        self.error = epsilon
        # Derived from gene 4: reach = how far the dendrites extend
        self.reach = (int(np.sqrt(self.genes[4])) - 1) // 2
        # Protein 4: weights -- synaptic strengths, stored in THIS cell's dendrites
        # Size determined by gene 4 (WG), scaling by gene 6 (AW / fan-in)
        # He initialization: randn * sqrt(2 / fan_in) -- optimal for ReLU/leaky ReLU
        fan_in = max(self.genes[6], 1)
        self.weights = np.clip(
            np.random.randn(int(self.genes[4])) * np.sqrt(2.0 / (fan_in + 1e-8)),
            -1.0, 1.0
        )
        # Protein 5: bias -- offset to activation (like resting potential)
        # Centered at 0 so the network can learn positive or negative offsets
        # Range determined by gene 5 (BR)
        self.bias = np.random.uniform(-self.genes[5], self.genes[5])

    def color_genes(self):
        """Map each gene to an RGB color for visualization.

        Genes 0-3 (breeding): mapped through COLORS palette
        Genes 4-8 (original network): individual color channels
        Genes 9-11 (new autonomy): cyan/magenta/yellow
        Note: only genes 0-8 are shown in the 3x3 cell display grid;
        genes 9-11 are visible via right-click inspect and telemetry.
        """
        # Breeding genes 0-3: use standard color palette
        self.colors = [
            tuple(max(0, min(int(gene * 255 // 15), 255)) for color_component in color)
            for gene, color in zip(self.genes[:4], COLORS)
        ]
        # Network genes 4-8: scale to 0-255 range
        gene_4_scaled = max(0, min(int(((self.genes[4] - 9) / (81 - 9)) * 255), 255))
        gene_5_scaled = max(0, min(int(self.genes[5] * 255), 255))
        gene_6_scaled = max(0, min(int(((self.genes[6] - 5) / (30 - 5)) * 255), 255))
        gene_7_scaled = max(0, min(int(((self.genes[7] - 0.0001) / (0.01 - 0.0001)) * 255), 255))
        gene_8_scaled = max(0, min(int(((self.genes[8] - 1e-6) / (1e-4 - 1e-6)) * 255), 255))
        self.colors.extend([
            (gene_4_scaled, 0, gene_4_scaled),     # Gene 4 WG: purple
            (0, gene_5_scaled, gene_5_scaled),     # Gene 5 BR: cyan
            (gene_6_scaled, 0, 0),                 # Gene 6 AW: red
            (0, gene_7_scaled, 0),                 # Gene 7 CD: green
            (0, 0, gene_8_scaled)                  # Gene 8 WD: blue
        ])
        # Autonomy genes 9-11: new color channels
        gene_9_scaled = max(0, min(int(((self.genes[9] - 0.003) / (0.05 - 0.003)) * 255), 255))
        gene_10_val = max(1e-8, min(self.genes[10], 1e-4))
        gene_10_scaled = max(0, min(int(((np.log10(gene_10_val) + 8) / 4) * 255), 255))
        gene_11_scaled = max(0, min(int(((self.genes[11] - 0.01) / (0.3 - 0.01)) * 255), 255))
        self.colors.extend([
            (0, gene_9_scaled, gene_9_scaled),     # Gene 9 LR: cyan
            (gene_10_scaled, 0, gene_10_scaled),   # Gene 10 GT: magenta
            (gene_11_scaled, gene_11_scaled, 0),   # Gene 11 AS: yellow
        ])

    def color_proteins(self):
        """Map protein values to colors for visualization (3x3 grid display)."""
        self.protein_colors = [0] * 12
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
        # Genes 9-11 don't have separate protein colors (they ARE gene parameters)
        self.protein_colors[9] = (0, 0, 0)
        self.protein_colors[10] = (0, 0, 0)
        self.protein_colors[11] = (0, 0, 0)

    def __setstate__(self, state):
        """Restore from pickle with backward compatibility.

        Handles old pickles with 9 genes by extending to 12 with sensible defaults.
        """
        self.__dict__.update(state)

        # Very old pickles: may have fewer than 9 genes
        if not hasattr(self, 'genes') or len(self.genes) < 9:
            self.genes = [0] * 12
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

        # Migrate 9-gene cells to 12-gene cells (added genes 9=LR, 10=GT, 11=AS)
        if len(self.genes) < 12:
            cfg = Cell._config
            while len(self.genes) < 12:
                idx = len(self.genes)
                if idx == 9:
                    self.genes.append(cfg.learning_rate if cfg else 0.01)        # LR
                elif idx == 10:
                    self.genes.append(cfg.gradient_threshold if cfg else 1e-7)   # GT
                elif idx == 11:
                    self.genes.append(getattr(cfg, 'activation_slope', 0.1) if cfg else 0.1)  # AS
                else:
                    self.genes.append(0)

        # Pad colors / protein_colors arrays to match gene count
        if hasattr(self, 'colors') and len(self.colors) < 12:
            while len(self.colors) < 12:
                self.colors.append((0, 0, 0))
        if hasattr(self, 'protein_colors') and len(self.protein_colors) < 12:
            while len(self.protein_colors) < 12:
                self.protein_colors.append((0, 0, 0))

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
            fan_in = max(self.genes[6], 1)
            self.weights = np.clip(
                np.random.randn(int(self.genes[4])) * np.sqrt(2.0 / (fan_in + 1e-8)),
                -1.0, 1.0
            )

        if not hasattr(self, 'protein_colors'):
            self.color_proteins()

    # ---- Forward pass: weights are in THIS cell's dendrites ----
    # Unlike traditional NNs where weights live in layer-to-layer matrices,
    # here each cell OWNS its weights in self.weights (a flat 1D array).
    # Forward: charge = sum(upstream_cell.charge * my_weight[index])
    # Weight index: (dx + reach) * matrix_width + (dy + reach)

    def compute_total_charge(self, upper_layer_cells, reach):
        """Forward pass: charge = bias + sum(upstream_charge * weight).

        Each cell computes its activation from the cells in the layer above,
        using its own dendritic weights.  Bias is added as a baseline signal
        (like a neuron's resting membrane potential).
        """
        cfg = Cell._config
        if cfg and cfg.autonomous_network_genes:
            reach = self.reach
            NUMBER_OF_WEIGHTS = self.genes[4]
            WEIGHT_MATRIX = int(np.sqrt(NUMBER_OF_WEIGHTS))
        else:
            WEIGHT_MATRIX = 2 * reach + 1
        charge = self.bias  # Start with bias (resting potential)
        for dx, dy, cell in upper_layer_cells:
            weight_index = (dx + reach) * WEIGHT_MATRIX + (dy + reach)
            if 0 <= weight_index < len(self.weights):
                charge += cell.charge * self.weights[weight_index]
        self.charge = np.clip(charge, -10, 10)

    def compute_total_charge_reverse(self, lower_layer_cells, reach):
        """Reverse pass: charge flows backward for visualization.

        Uses bias as baseline, same as forward pass.
        """
        cfg = Cell._config
        if cfg and cfg.autonomous_network_genes:
            pass  # each cell below uses its own reach
        else:
            WEIGHT_MATRIX = 2 * reach + 1
        charge = self.bias  # Start with bias (same as forward pass)
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
    # Gene 11 (AS) controls the negative-region slope of leaky ReLU.
    # Low slope (0.01) = selective neuron: strongly suppresses negative inputs.
    # High slope (0.3) = permissive neuron: passes more of the signal through.
    # In biology, different neuron types have different response curves --
    # this gene lets evolution discover what mix of selectivity works best.

    def relu(self, x):
        """Leaky ReLU activation. Negative slope from gene 11 (activation slope)."""
        cfg = Cell._config
        slope = self.genes[11] if (cfg and cfg.autonomous_network_genes) else (
            cfg.activation_slope if cfg else 0.1)
        return np.where(x > 0, x, slope * x)

    def relu_derivative(self, x):
        """Leaky ReLU derivative. Negative slope from gene 11 (activation slope)."""
        cfg = Cell._config
        slope = self.genes[11] if (cfg and cfg.autonomous_network_genes) else (
            cfg.activation_slope if cfg else 0.1)
        return np.where(x > 0, 1.0, slope)

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
    # Error propagates BACKWARD through the same dendritic connections.
    # The reversed weight index maps (dx,dy) to (-dx,-dy) -- mathematically
    # equivalent to standard backprop through a transposed weight matrix.
    # NOTE: reversed_index = len(weights) - 1 - weight_index. DO NOT CHANGE THIS.

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
        """Update this cell's dendritic weights and bias via gradient descent.

        In autonomous mode, learning_rate and weight_decay come from the cell's
        own genes (9 and 8), not from the global config.  This lets each cell
        evolve its own plasticity rate and regularization strength.
        """
        cfg = Cell._config
        gradient_clip_range = cfg.gradient_clip_range if cfg else 1
        if cfg and cfg.autonomous_network_genes:
            reach = self.reach
            NUMBER_OF_WEIGHTS = self.genes[4]
            WEIGHT_MATRIX = int(np.sqrt(NUMBER_OF_WEIGHTS))
            weight_decay = self.genes[8]       # Gene 8: per-cell L2 decay
            learning_rate = self.genes[9]      # Gene 9: per-cell learning rate
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
        """Track gradient magnitude to decide if this cell is learning.

        Gene 10 (GT) controls the survival threshold: cells with average gradient
        above this threshold are marked as "significantly learning" and protected
        from gradient-based pruning.  In autonomous mode, each cell sets its own
        threshold -- like neurotrophic factor receptor density.
        """
        cfg = Cell._config
        how_much = cfg.how_much_training_data if cfg else 20
        # Gene 10: per-cell gradient threshold (survival sensitivity)
        grad_threshold = self.genes[10] if (cfg and cfg.autonomous_network_genes) else (
            cfg.gradient_threshold if cfg else 1e-7)
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
            fan_in = max(self.genes[6], 1)
            self.weights = np.random.randn(new_matrix ** 2) * np.sqrt(2.0 / (fan_in + 1e-8))
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
            fan_in = max(self.genes[6], 1)
            new_weights = np.random.randn(num_new_weights) * np.sqrt(2.0 / (fan_in + 1e-8))
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
        """Self-contained backward pass for this cell.

        In autonomous mode, learning rate (gene 9), weight decay (gene 8),
        and reach (gene 4) all come from the cell's own genes.
        """
        cfg = Cell._config
        # Gene 9 (LR): per-cell learning rate in autonomous mode
        if cfg and cfg.autonomous_network_genes:
            lr = self.genes[9]
        else:
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
            fan_in = max(self.genes[6], 1)
            self.weights = np.clip(
                np.random.randn(int(self.genes[4])) * np.sqrt(2.0 / (fan_in + 1e-8)),
                -1.0, 1.0
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
        # Handle old cells that might have only 9 genes
        lr_str = f", LR={self.genes[9]:.4f}" if len(self.genes) > 9 else ""
        gt_str = f", GT={self.genes[10]:.2e}" if len(self.genes) > 10 else ""
        as_str = f", AS={self.genes[11]:.2f}" if len(self.genes) > 11 else ""
        return (
            f"Neuron: layer={self.layer} x={self.x} y={self.y}\n"
            f"Genes (breeding):\n"
            f"  OT={self.genes[0]}, IT={self.genes[1]}, BT={self.genes[2]}, MR={self.genes[3]}\n"
            f"Genes (network):\n"
            f"  WG={self.genes[4]}, BR={self.genes[5]}, AW={self.genes[6]}, CD={self.genes[7]}, WD={self.genes[8]}\n"
            f"  {lr_str.lstrip(', ')}{gt_str}{as_str}\n"
            f"Proteins:\n"
            f"  charge={charge}, error={error}, bias={bias}, gradient={gradient}\n"
            f"max_charge_diff_forward={max_charge_diff_forward}, significant_charge_change_forward={significant_charge_change_forward}\n"
            f"max_charge_diff_reverse={max_charge_diff_reverse}, significant_charge_change_reverse={significant_charge_change_reverse}\n"
            f"gradient_average={gradient_average}, significant_gradient_change={significant_gradient_change}\n"
            f"reach={reach}\n"
            f"weights={weights_str}"
        )
