"""2D visualization: cell rendering, statistics, prediction plots."""
import numpy as np
import pygame

from neurosim.config import (WINDOW_WIDTH, WINDOW_HEIGHT, CELL_SIZE, WHITE, BLACK, BLUE,
                              WINDOW_EXTENSION, WIDTH, HEIGHT)


def draw_grid(growthsurface):
    """Draw grid lines on the play surface."""
    for x in range(0, WINDOW_WIDTH, WINDOW_WIDTH // 4):
        pygame.draw.line(growthsurface, BLACK, (x, 0), (x, WINDOW_HEIGHT), 2)
    for y in range(0, WINDOW_HEIGHT, WINDOW_HEIGHT // 4):
        pygame.draw.line(growthsurface, BLACK, (0, y), (WINDOW_WIDTH, y), 2)


def draw_cells(state, config, growthsurface):
    """Render all cells as colored 3x3 grids showing genes or proteins."""
    if state.display == "genes":
        for layer in range(config.num_layers):
            for (x, y) in np.ndindex(state.cells.shape[:2]):
                if state.cells[x, y, layer] is not None:
                    state.cells[x, y, layer].color_genes()
                    for i in range(9):
                        color = state.cells[x, y, layer].colors[i]
                        pygame.draw.rect(
                            growthsurface, color,
                            pygame.Rect(
                                x * CELL_SIZE + (i % 3) * (CELL_SIZE // 3) + (layer % 4) * (WINDOW_WIDTH // 4),
                                y * CELL_SIZE + ((i // 3) % 3) * (CELL_SIZE // 3) + (layer // 4) * (WINDOW_HEIGHT // 4),
                                CELL_SIZE // 3, CELL_SIZE // 3
                            )
                        )
    else:  # proteins
        for layer in range(config.num_layers):
            for (x, y) in np.ndindex(state.cells.shape[:2]):
                if state.cells[x, y, layer] is not None:
                    state.cells[x, y, layer].color_proteins()
                    for i in range(9):
                        color = state.cells[x, y, layer].protein_colors[i]
                        pygame.draw.rect(
                            growthsurface, color,
                            pygame.Rect(
                                x * CELL_SIZE + (i % 3) * (CELL_SIZE // 3) + (layer % 4) * (WINDOW_WIDTH // 4),
                                y * CELL_SIZE + ((i // 3) % 3) * (CELL_SIZE // 3) + (layer // 4) * (WINDOW_HEIGHT // 4),
                                CELL_SIZE // 3, CELL_SIZE // 3
                            )
                        )


def update_cell_types(cells, config):
    """Categorize cells by their first 4 genes."""
    cell_types = {}
    for layer in range(1, config.num_layers - 1):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                cell_type = tuple(cells[x, y, layer].genes[:4])
                cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
    return cell_types


def update_phenotype_cell_types(cells, config):
    """Categorize cells by computed phenotypes."""
    epsilon = config.epsilon
    phenotype_cell_types = {}
    all_charges, all_biases, all_errors, all_weights = [], [], [], []
    all_gradients, all_max_fwd, all_max_rev = [], [], []

    count_pos = 0
    total_cells = 0

    for layer in range(1, config.num_layers - 1):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                if not all(weight in [0, epsilon] for weight in cells[x, y, layer].weights):
                    cell = cells[x, y, layer]
                    all_charges.append(cell.charge)
                    all_biases.append(cell.bias)
                    all_errors.append(cell.error)
                    all_weights.append(np.mean(cell.weights))
                    all_gradients.append(cell.gradient)
                    all_max_fwd.append(cell.max_charge_diff_forward)
                    all_max_rev.append(cell.max_charge_diff_reverse)
                    count_pos += 1

    if count_pos > 0:
        return (0, 0, {})

    charge_mean, charge_std = (np.mean(all_charges), np.std(all_charges)) if all_charges else (0, 0)
    bias_mean, bias_std = (np.mean(all_biases), np.std(all_biases)) if all_biases else (0, 0)
    error_mean, error_std = (np.mean(all_errors), np.std(all_errors)) if all_errors else (0, 0)
    weights_mean, weights_std = (np.mean(all_weights), np.std(all_weights)) if all_weights else (0, 0)
    gradient_mean, gradient_std = (np.mean(all_gradients), np.std(all_gradients)) if all_gradients else (0, 0)
    max_fwd_mean, max_fwd_std = (np.mean(all_max_fwd), np.std(all_max_fwd)) if all_max_fwd else (0, 0)
    max_rev_mean, max_rev_std = (np.mean(all_max_rev), np.std(all_max_rev)) if all_max_rev else (0, 0)

    for layer in range(1, config.num_layers - 1):
        for (x, y) in np.ndindex(cells.shape[:2]):
            if cells[x, y, layer] is not None:
                cell = cells[x, y, layer]
                weights_avg = np.mean(cell.weights)
                phenotype = (
                    'charge:' + ('+' if cell.charge > charge_mean else '-') + str(int(abs((cell.charge - charge_mean) / (charge_std + epsilon)))),
                    'bias:' + ('+' if cell.bias > bias_mean else '-') + str(int(abs((cell.bias - bias_mean) / (bias_std + epsilon)))),
                    'error:' + ('+' if cell.error > error_mean else '-') + str(int(abs((cell.error - error_mean) / (error_std + epsilon)))),
                    'weights:' + ('+' if weights_avg > weights_mean else '-') + str(int(abs((weights_avg - weights_mean) / (weights_std + epsilon)))),
                    'gradient:' + ('+' if cell.gradient > gradient_mean else '-') + str(int(abs((cell.gradient - gradient_mean) / (gradient_std + epsilon)))),
                    'max_fwd:' + ('+' if cell.max_charge_diff_forward > max_fwd_mean else '-') + str(int(abs((cell.max_charge_diff_forward - max_fwd_mean) / (max_fwd_std + epsilon)))),
                    'max_rev:' + ('+' if cell.max_charge_diff_reverse > max_rev_mean else '-') + str(int(abs((cell.max_charge_diff_reverse - max_rev_mean) / (max_rev_std + epsilon))))
                )
                if phenotype not in phenotype_cell_types:
                    phenotype_cell_types[phenotype] = []
                phenotype_cell_types[phenotype].append(cell)

    return (count_pos, total_cells, phenotype_cell_types)


def display_statistics(cell_types, ui_print_fn):
    """Display top 5 genotypes to side panel."""
    sorted_types = sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:5]
    statistics = ["[OT IT BT] CH WG ER BI: Overcrowding Isolation Birth"]
    statistics += [f"Type: {ct} | Count: {count}" for ct, count in sorted_types]
    total_mutation_freq = sum(int(ct[3]) for ct in cell_types.keys())
    avg_mutation_freq = total_mutation_freq / len(cell_types) if cell_types else 0
    statistics.append(f"\nAverage Mutation Frequency: {avg_mutation_freq:.2f}")
    ui_print_fn("\nDisplaying statistics for the top 5 cell types:", 10)
    ui_print_fn("\n".join(statistics))


def display_phenotype_statistics(phenotype_cell_types, ui_print_fn):
    """Display phenotype statistics."""
    count_pos, total_cells, phenotype_dict = phenotype_cell_types
    sorted_types = sorted([
        (ph, len(cl), np.mean([c.charge for c in cl]), np.mean([c.bias for c in cl]),
         np.mean([c.error for c in cl]), np.mean([np.mean(c.weights) for c in cl]),
         np.mean([c.gradient for c in cl]), np.mean([c.max_charge_diff_forward for c in cl]),
         np.mean([c.max_charge_diff_reverse for c in cl]))
        for ph, cl in phenotype_dict.items()
    ], key=lambda x: x[1], reverse=True)[:5]

    statistics = ["Phenotype: Avg Charge | Avg Bias | Avg Error | Avg Weights | Avg Gradient | Avg Max Fwd | Avg Max Rev"]
    for ph, count, ac, ab, ae, aw, ag, amf, amr in sorted_types:
        statistics.append(f"{ph} | Count: {count:4} \nCharge: {ac:.4f} | Bias: {ab:+.4f} | Error: {ae:.4e} | Weights: {aw:.4f} | Gradient: {ag:.4e} | MaxFwd: {amf:.4f} | MaxRev: {amr:.4f}")
    ui_print_fn("\nDisplaying statistics for the top 5 phenotypes:")
    ui_print_fn("\n".join(statistics))


def display_max_charge_diff(state, config, N, ui_print_fn):
    """Show top N cells by charge difference."""
    charge_diffs = []
    for layer in range(1, config.num_layers - 1):
        for x, y in np.ndindex(state.cells.shape[:2]):
            cell = state.cells[x, y, layer]
            if cell is not None:
                if state.direction_of_charge_flow == "+++++>>>>>":
                    cd = cell.max_charge_diff_forward
                else:
                    cd = cell.max_charge_diff_reverse
                charge_diffs.append((x, y, layer, cd))
    top_N = sorted(charge_diffs, key=lambda x: x[3], reverse=True)[:N]
    output = [f"Cell at ({x}, {y}, {l}) - Charge difference: {cd:.2f}" for x, y, l, cd in top_N]
    ui_print_fn("\nMax charge difference for top 5 cells:")
    ui_print_fn("\n".join(output))


def display_averages(state, config, ui_print_fn):
    """Display detailed layer-wise statistics."""
    epsilon = config.epsilon
    output = []
    try:
        nl = config.num_layers
        avg_gradient = np.zeros(nl)
        max_gradient = np.zeros(nl)
        min_gradient = np.zeros(nl)
        avg_error = np.zeros(nl)
        avg_charge = np.zeros(nl)
        avg_weights = np.zeros(nl)
        avg_weights_per_cell = np.zeros(nl)

        total_cells = 0
        active_cells = 0

        for layer in range(1, nl - 1):
            cells_in_layer = [state.cells[x, y, layer] for x, y in np.ndindex(state.cells.shape[:2]) if state.cells[x, y, layer] is not None]
            total_cells += len(cells_in_layer)
            active_in_layer = [c for c in cells_in_layer if not all(w in [0, epsilon] for w in c.weights)]
            active_cells += len(active_in_layer)

            if active_in_layer:
                gradients = [c.gradient for c in active_in_layer]
                avg_gradient[layer] = np.mean(gradients)
                max_gradient[layer] = np.max(gradients)
                min_gradient[layer] = np.min(gradients)
                avg_error[layer] = np.mean([abs(c.error) for c in active_in_layer])
                avg_charge[layer] = np.mean([abs(c.charge) for c in active_in_layer])
                avg_weights[layer] = np.mean([np.mean(abs(c.weights)) for c in active_in_layer])
                avg_weights_per_cell[layer] = np.mean([len(c.weights) for c in active_in_layer])

        output.append(f"Predictions: {state.total_predictions} | Average Loss: {state.running_avg_loss:.4e}")
        output.append(f"Active Cells: {active_cells}/{total_cells}")
        output.append("\nLayer-wise Statistics:")
        for layer in range(1, nl - 1):
            output.append(f"Layer {layer}:")
            output.append(f"  Gradient: {avg_gradient[layer]:.4e}, Max: {max_gradient[layer]:.4e}, Min: {min_gradient[layer]:.4e}")
            output.append(f"  Error: {avg_error[layer]:.4e}")
            output.append(f"  Charge: {avg_charge[layer]:.4f}")
            output.append(f"  Weights: {avg_weights[layer]:.4f}")
            output.append(f"  Avg Weights/Cell: {avg_weights_per_cell[layer]:.2f}")
        output.append("\nNetwork-wide (excluding input and output layers):")
        output.append(f"  Avg Gradient: {np.mean(avg_gradient[1:-1]):.4e}, Max: {np.max(max_gradient[1:-1]):.4e}, Min: {np.min(min_gradient[1:-1]):.4e}")
        output.append(f"  Avg Error: {np.mean(avg_error[1:-1]):.4e}")
        output.append(f"  Avg Charge: {np.mean(avg_charge[1:-1]):.4f}")
        output.append(f"  Avg Weights: {np.mean(avg_weights[1:-1]):.4f}")
        output.append(f"  Avg Weights/Cell: {np.mean(avg_weights_per_cell[1:-1]):.2f}")
    except Exception as e:
        output.append(f"Error in averages: {e}")

    ui_print_fn("\nNetwork Statistics:", 10)
    ui_print_fn("\n".join(output), 50)


def prediction_plot(state, bottom_caption_surface):
    """Plot training loss curve on bottom panel."""
    offset = 756
    y = WINDOW_EXTENSION - (state.running_avg_loss / 10) * WINDOW_EXTENSION
    state.points.append((len(state.points) + offset, y))
    if len(state.points) > 700:
        state.points.pop(0)
        state.points = [(x - 1, y) for x, y in state.points]
    for point in state.points:
        pygame.draw.circle(bottom_caption_surface, BLUE, point, 4)


def update_training_stats(state, config, ui_print_fn=None):
    """Update training statistics dictionary."""
    epsilon = config.epsilon
    nl = config.num_layers
    total_cells_count = sum(1 for layer in range(1, nl - 1) for cell in state.cells[:, :, layer].flatten() if cell is not None)
    total_weights_count = sum(len(cell.weights) for layer in range(1, nl - 1) for cell in state.cells[:, :, layer].flatten() if cell is not None)
    avg_wpc = total_weights_count / (total_cells_count + epsilon)

    count_pos, total_cells_pheno, phenotype_cell_types = update_phenotype_cell_types(state.cells, config)
    cell_types = update_cell_types(state.cells, config)
    top_genotypes = sorted(cell_types.items(), key=lambda x: x[1], reverse=True)[:5]

    total_neurons = total_cells_count
    active_neurons = sum(1 for layer in range(1, nl - 1) for cell in state.cells[:, :, layer].flatten() if cell is not None and np.any(cell.weights != 0))
    avg_connections = sum(np.count_nonzero(cell.weights) for layer in range(1, nl - 1) for cell in state.cells[:, :, layer].flatten() if cell is not None) / (total_neurons + epsilon)

    gradients = [abs(cell.gradient) for layer in range(1, nl - 1) for cell in state.cells[:, :, layer].flatten() if cell is not None]
    max_grad = max(gradients) if gradients else 0
    avg_grad = np.mean(gradients) if gradients else 0

    state.training_stats_buffer = {
        "Accuracy": f"{state.bingo_count}/{config.how_much_training_data} (Max: {state.max_bingo_count})",
        "Avg Loss": f"{state.running_avg_loss:.4e}",
        "Total Predictions": state.total_predictions,
        "Avg Weights/Cell": f"{avg_wpc:.2f}",
        "Total Weights": f"{total_weights_count:.2f}",
        "Total Cells": total_cells_count,
        "Active Cells": f"{count_pos}/{total_cells_count}",
        "Top Genotypes": top_genotypes,
        "Active Neurons": f"{active_neurons}/{total_neurons}",
        "Avg Connections/Neuron": f"{avg_connections:.2f}",
        "Max Gradient": f"{max_grad:.4e}",
        "Avg Gradient": f"{avg_grad:.4e}",
    }


def display_training_stats(state, ui_print_fn):
    """Render training stats to side panel."""
    buf = state.training_stats_buffer
    stats_text = "Training Statistics:\n"
    stats_text += f"Accuracy: {buf['Accuracy']}\n"
    stats_text += f"Avg Loss: {buf['Avg Loss']}\n"
    stats_text += f"Total Predictions: {buf['Total Predictions']}\n"
    stats_text += f"Avg Weights/Cell: {buf['Avg Weights/Cell']}\n"
    stats_text += f"Total Weights: {buf['Total Weights']}\n"
    stats_text += f"Total Cells: {buf['Total Cells']}\n"
    stats_text += f"Active Cells: {buf['Active Cells']}\n"
    stats_text += "Top Genotypes:\n"
    for genotype, count in buf['Top Genotypes']:
        stats_text += f"  {genotype}: {count}\n"
    stats_text += f"Active Neurons: {buf['Active Neurons']}\n"
    stats_text += f"Avg Connections/Neuron: {buf['Avg Connections/Neuron']}\n"
    stats_text += f"Max Gradient: {buf['Max Gradient']}\n"
    stats_text += f"Avg Gradient: {buf['Avg Gradient']}\n"
    ui_print_fn(stats_text, 10)
