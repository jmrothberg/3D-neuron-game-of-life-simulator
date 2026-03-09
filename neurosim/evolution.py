"""Cell evolution -- Andromida mode (Conway's Game of Life with genetics), pruning, mutation."""
import numpy as np
import random

from neurosim.cell import Cell


def update_cells(state, config):
    """Core evolution: pruning, somatic mutation, andromida birth/death/protection."""
    start_layer = 1
    stop_layer = config.num_layers - 1

    for layer in range(start_layer, stop_layer):
        for (x, y) in np.ndindex(state.cells.shape[:2]):
            # --- Pruning ---
            if state.cells[x, y, layer] is not None:
                cell = state.cells[x, y, layer]

                if cell.should_i_die(state.prune, state.gradient_prune, state.prune_logic):
                    state.cells[x, y, layer] = None
                    state.invalidate_neighbor_cache()
                    continue

            # --- Somatic mutation (happens even outside andromida mode) ---
            if state.cells[x, y, layer] is not None:
                cell = state.cells[x, y, layer]
                mutation_chance = cell.genes[3] / 100000
                if random.random() < mutation_chance:
                    if random.random() < 0.5:
                        cell.initalize_breeding_genes()
                    else:
                        cell.initalize_network_genes(
                            config.number_of_weights, config.bias_range,
                            config.avg_weights_cell, config.charge_delta,
                            config.weight_decay, config.mutation_rate,
                            cells_array=state.cells
                        )

            # --- Andromida mode: birth, death, protection ---
            if state.andromida_mode:
                if layer == 1:
                    neighbors = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                                 for dz in [0, 1] if dx != 0 or dy != 0 or dz != 0]
                elif layer == config.num_layers - 2:
                    neighbors = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                                 for dz in [0, -1] if dx != 0 or dy != 0 or dz != 0]
                else:
                    neighbors = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                                 for dz in [-1, 0, 1] if dx != 0 or dy != 0 or dz != 0]

                alive_neighbors = [
                    state.cells[x + dx, y + dy, layer + dz]
                    for dx, dy, dz in neighbors
                    if (0 <= x + dx < state.cells.shape[0]
                        and 0 <= y + dy < state.cells.shape[1]
                        and 0 <= layer + dz < state.cells.shape[2]
                        and state.cells[x + dx, y + dy, layer + dz] is not None)
                ]
                num_alive = len(alive_neighbors)

                # --- Cell Birth ---
                if state.cells[x, y, layer] is None and alive_neighbors:
                    potential_parents = alive_neighbors
                    if len(potential_parents) >= 2:
                        parent1, parent2 = random.sample(potential_parents, 2)
                        new_genes = [random.choice([parent1.genes[i], parent2.genes[i]]) for i in range(9)]
                    elif potential_parents:
                        parent1 = potential_parents[0]
                        new_genes = parent1.genes

                    if num_alive == new_genes[2]:
                        state.cells[x, y, layer] = Cell(
                            layer, x, y, config.number_of_weights, config.bias_range,
                            config.avg_weights_cell, config.charge_delta,
                            config.weight_decay, config.mutation_rate, new_genes
                        )
                        cell = state.cells[x, y, layer]
                        state.invalidate_neighbor_cache()

                        mutation_chance = cell.genes[3] / 1000
                        if random.random() < mutation_chance:
                            cell.initalize_breeding_genes()
                        if random.random() < mutation_chance:
                            cell.initalize_network_genes(
                                config.number_of_weights, config.bias_range,
                                config.avg_weights_cell, config.charge_delta,
                                config.weight_decay, config.mutation_rate,
                                cells_array=state.cells
                            )

                # --- Cell Death / Protection ---
                if state.cells[x, y, layer] is not None:
                    cell = state.cells[x, y, layer]
                    if cell.should_i_die_genetic(num_alive, state.charge_change_protection):
                        state.cells[x, y, layer] = None
                        state.invalidate_neighbor_cache()


def reset_all_gradient_changes(state, config):
    """Reset all gradient/charge tracking for all cells."""
    for layer in range(1, config.num_layers - 1):
        for (x, y) in np.ndindex(state.cells.shape[:2]):
            if state.cells[x, y, layer] is not None:
                state.cells[x, y, layer].reset_gradient_change()
                state.cells[x, y, layer].significant_charge_change_forward = False
                state.cells[x, y, layer].significant_charge_change_reverse = False


def reset_directional_charge_history(state, config, direction):
    """Reset charge history for a specific direction."""
    for layer in range(1, config.num_layers - 1):
        for (x, y) in np.ndindex(state.cells.shape[:2]):
            if state.cells[x, y, layer] is not None:
                state.cells[x, y, layer].reset_directional_charge_history(direction)
