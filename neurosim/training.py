"""Training loop, forward propagation, reverse propagation, and backpropagation."""
import numpy as np


def forward_propagation(state, config):
    """Forward pass: layer 1 to Num_Layers-2. Each cell computes charge from upper layer."""
    for layer in range(1, config.num_layers - 1):
        for (x, y) in np.ndindex(state.cells.shape[:2]):
            cell = state.cells[x, y, layer]
            if cell is not None:
                cell.forward(state.cells)


def reverse_forward_propagation(state, config):
    """Reverse pass: charge flows backward for visualization. No weight updates."""
    for layer in range(config.num_layers - 2, 0, -1):
        for (x, y) in np.ndindex(state.cells.shape[:2]):
            cell = state.cells[x, y, layer]
            if cell is not None:
                if layer == config.num_layers - 2:
                    if state.cells[x, y, config.num_layers - 1] is not None:
                        cell.update_charge(state.cells[x, y, config.num_layers - 1].charge, "reverse")
                else:
                    reach = cell.reach if config.autonomous_network_genes else config.length_of_dendrite
                    lower_layer_cells = cell.get_layer_below_cells(state.cells, reach)
                    cell.compute_total_charge_reverse(lower_layer_cells, reach)
                    cell.update_charge(cell.charge, "reverse")


def back_propagation(state, config, render_backprop_fn=None):
    """Backpropagation: compute errors and update weights, layer by layer in reverse.

    render_backprop_fn: optional callback(layer, x, y) for 3D backprop visualization.
    """
    for layer in range(config.num_layers - 2, 0, -1):
        # Cache max_reach for this layer's below-layer (Phase 6 optimization)
        max_reach_below = None
        if layer < config.num_layers - 2:
            max_reach_below = state.get_max_reach_for_layer(layer + 1)

        for (x, y) in np.ndindex(state.cells.shape[:2]):
            cell = state.cells[x, y, layer]
            if cell is not None:
                cell.backward(state.cells, learning_rate=config.learning_rate,
                              max_reach_below=max_reach_below)

                # Optional 3D backprop visualization
                if render_backprop_fn and state.show_3d_view and state.show_backprop_view:
                    render_backprop_fn(current_layer=layer, current_pos=(x, y))


def prediction_to_actual(state, config):
    """Compare predictions to actual labels and update metrics."""
    cells = state.cells
    num_layers = config.num_layers

    one_hot_label_guess = np.array([
        cell.charge if cell is not None else 0
        for cell in cells[9:19, 14, num_layers - 2]
    ])
    one_hot_label_actual = np.array([
        cell.charge if cell is not None else 0
        for cell in cells[9:19, 14, num_layers - 1]
    ])

    predictions_clipped = np.clip(one_hot_label_guess, 1e-7, 1 - 1e-7)
    loss = -np.sum(one_hot_label_actual * np.log(predictions_clipped))

    pos_guess = np.argmax(one_hot_label_guess)
    pos_actual = np.argmax(one_hot_label_actual)

    state.total_loss += loss
    state.total_predictions += 1
    state.running_avg_loss = state.total_loss / state.total_predictions

    if pos_guess == pos_actual:
        state.bingo_count += 1
        if state.bingo_count > state.max_bingo_count:
            state.max_bingo_count = state.bingo_count


def train_network(state, config, render_backprop_fn=None):
    """Single epoch: forward or reverse pass, then optionally backprop."""
    for epoch in range(state.epochs):
        if state.direction_of_charge_flow == "+++++>>>>>":
            forward_propagation(state, config)
            if state.back_prop:
                back_propagation(state, config, render_backprop_fn=render_backprop_fn)
            prediction_to_actual(state, config)
        if state.direction_of_charge_flow == "<<<<<-----":
            reverse_forward_propagation(state, config)


def training(state, config, draw_fn=None, render_3d_fn=None, render_backprop_fn=None):
    """Main training loop: iterate through training data, run train_network per sample."""
    import copy
    state.bingo_count = 0
    set_size = config.how_much_training_data

    for i in range(set_size):
        state.cells[:, :, 0] = state.training_data_layer_0[i]
        state.cells[:, :, config.num_layers - 1] = state.training_data_num_layer_minus_1[i]
        state.total_weights = sum(
            np.sum(np.abs(cell.weights))
            for layer_idx in range(1, config.num_layers - 1)
            for cell in state.cells[:, :, layer_idx].flatten()
            if cell is not None
        )
        state.total_weights_list[i] = state.total_weights

        train_network(state, config, render_backprop_fn=render_backprop_fn)

        if state.display_updating and draw_fn:
            draw_fn()
