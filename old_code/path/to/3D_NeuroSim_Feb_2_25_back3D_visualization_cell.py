class Cell:
    # The Cell class represents a neuron in our 3D neural network.
    def __init__(self, layer, x, y, weights_per_cell_possible, Bias_Range, Avg_Weights_Cell, 
                 charge_delta, weight_decay, mutation_rate, genes=None):
        global cells
        # Use the global EPSILON constant for consistency.
        # Add coordinate validation; always set self.layer even if x,y are adjusted.
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            print(f"Warning: Invalid coordinates: x={x}, y={y}. Must be within grid bounds (0-{WIDTH-1}, 0-{HEIGHT-1})")
            print_to_side_panel(f"Warning: Invalid coordinates: x={x}, y={y}. Must be within grid bounds (0-{WIDTH-1}, 0-{HEIGHT-1})")
            self.x = min(max(0, x), WIDTH-1)
            self.y = min(max(0, y), HEIGHT-1)
            self.layer = layer  # Ensure the layer is assigned even when correcting coordinates.
        else:
            self.x = x
            self.y = y
            self.layer = layer

        # NEW: Add explicit connection lists
        self.incoming_connections = []   # For storing (source_cell, weight_index)
        self.outgoing_connections = []   # For storing (target_cell, weight_index)
        
        self.alive = True  # Use this flag to mark cell death

        if genes is None: 
            self.initalize_all_genes()
            self.initalize_breeding_genes()
            self.initalize_network_genes(weights_per_cell_possible, Bias_Range, Avg_Weights_Cell, charge_delta, weight_decay, mutation_rate)
        else:  # genes are passed from parent cell 
            self.genes = genes.copy()
        
        self.color_genes()
        self.initialize_network_proteins()
        self.initalize_cell_memory()
        self.color_proteins()