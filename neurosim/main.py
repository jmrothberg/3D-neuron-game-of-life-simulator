"""Main entry point: pygame initialization, event loop, simulation orchestration."""
import sys
import time
import numpy as np
import pygame

from neurosim.config import (SimConfig, set_default_values,
                              WHITE, BLACK, GREEN, RED, LIGHT_GRAY,
                              WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_EXTENSION,
                              EXTENDED_WINDOW_WIDTH, EXTENDED_WINDOW_HEIGHT,
                              HELP_PANEL_WIDTH, MAIN_SURFACE_WIDTH,
                              CELL_SIZE, WIDTH, HEIGHT, ARRAY_LAYERS)
from neurosim.state import SimState
from neurosim.cell import Cell
from neurosim.training import training as run_training
from neurosim.evolution import update_cells, reset_all_gradient_changes
from neurosim.io_manager import save_file, load_file, load_training_data_main
from neurosim.visualization import (draw_grid, draw_cells, update_cell_types,
                                     update_phenotype_cell_types, display_statistics,
                                     display_phenotype_statistics, display_max_charge_diff,
                                     display_averages, prediction_plot,
                                     update_training_stats, display_training_stats)
from neurosim.visualization_3d import setup_3d_view, render_3d_network, render_3d_backprop
from neurosim.ui import (render_help_text, print_to_side_panel, pygame_input,
                          get_user_input, get_user_input_float, convert_x_y_to_index,
                          get_all_settings)
from neurosim.telemetry import compute_telemetry, format_telemetry

# Import help definitions
sys.path.insert(0, '.')
from get_help_defs import get_defs


def main():
    # ---- Initialize config and state ----
    config = set_default_values()
    state = SimState()
    state.total_weights_list = np.zeros(config.how_much_training_data)

    # Set Cell class-level config reference
    Cell.set_config(config)

    # ---- Pygame setup ----
    pygame.init()
    screen = pygame.display.set_mode((EXTENDED_WINDOW_WIDTH, EXTENDED_WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("JMR's Game of Life with Genetics & Neural Network")
    font = pygame.font.SysFont(None, 24)
    font_small = pygame.font.SysFont(None, 20)
    font_directory = pygame.font.SysFont(None, 16)

    bottom_caption_surface = pygame.Surface((EXTENDED_WINDOW_WIDTH, WINDOW_EXTENSION)).convert()
    help_surface = pygame.Surface((HELP_PANEL_WIDTH - 2, WINDOW_HEIGHT)).convert()

    subsurface_rect = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    growthsurface = screen.subsurface(subsurface_rect).convert()
    growthsurface.fill(WHITE)

    # ---- Help screens ----
    jmr_defs, jmr_defs2, conways_defs, how_network_works, forward_pass, how_backprop_works, how_backprop_works2, controls = get_defs()
    help_screen = {
        "jmr_defs": jmr_defs, "jmr_defs2": jmr_defs2, "conways_defs": conways_defs,
        "how_network_works": how_network_works, "forward_pass": forward_pass,
        "how_backprop_works": how_backprop_works, "how_backprop_works2": how_backprop_works2,
        "controls": controls
    }
    help_text = help_screen["controls"]
    help_surface.fill(LIGHT_GRAY)
    render_help_text(help_surface, help_text, font_small, BLACK, 10, 10, 20)
    screen.blit(help_surface, (MAIN_SURFACE_WIDTH + 3, 0))

    growthsurface.fill(WHITE)
    draw_grid(growthsurface)
    screen.blit(growthsurface, (0, 0))
    pygame.display.flip()

    # ---- Helper closures that capture local surfaces ----
    def ui_print(text, position=None):
        print_to_side_panel(text, state, screen, help_surface, font_small, state.show_3d_view, position)

    def do_pygame_input(prompt, default_value=None):
        return pygame_input(prompt, screen, bottom_caption_surface, font, clock, default_value)

    def do_get_user_input(prompt, default_value):
        return get_user_input(prompt, default_value, screen, bottom_caption_surface, font, clock)

    def do_get_user_input_float(prompt, default_value):
        return get_user_input_float(prompt, default_value, screen, bottom_caption_surface, font, clock)

    def do_draw():
        """Standard 2D draw cycle."""
        if state.show_3d_view:
            render_3d_network(state, config)
        else:
            growthsurface.fill(WHITE)
            draw_cells(state, config, growthsurface)
            draw_grid(growthsurface)
            screen.blit(growthsurface, (0, 0))
        pygame.display.flip()

    def do_render_backprop(current_layer, current_pos):
        """3D backprop visualization callback."""
        render_3d_backprop(state, config, current_layer, current_pos)
        pygame.display.flip()
        pygame.time.wait(10)

    # ---- Simulation loop ----
    start_time = time.time()
    epsilon = config.epsilon

    while state.simulating:

        # Autosave perfect networks
        if state.max_bingo_count == config.how_much_training_data and state.not_saved_yet:
            save_file(state, config, "-perfect", ui_print_fn=ui_print)
            state.not_saved_yet = False

        for event in pygame.event.get():

            # ---- H: cycle help screens ----
            if event.type == pygame.KEYDOWN and event.key == pygame.K_h and not state.show_3d_view:
                help_keys = list(help_screen.keys())
                state.current_index = (state.current_index + 1) % len(help_keys)
                help_text = help_screen[help_keys[state.current_index]]
                help_surface.fill(LIGHT_GRAY)
                render_help_text(help_surface, help_text, font_small, BLACK, 10, 10, 20)
                screen.blit(help_surface, (MAIN_SURFACE_WIDTH + 3, 0))

            if event.type == pygame.KEYDOWN:
                # ---- U: toggle autonomous network genes ----
                if event.key == pygame.K_u:
                    config.autonomous_network_genes = not config.autonomous_network_genes
                    msg = f"Autonomous network genes mode: {config.autonomous_network_genes}"
                    print(msg)
                    ui_print(msg)

                # ---- SPACE: toggle running ----
                elif event.key == pygame.K_SPACE:
                    state.running = not state.running
                    print(f"Running={state.running}")

                # ---- P: toggle pruning ----
                elif event.key == pygame.K_p:
                    state.prune = not state.prune
                    print(f"Prune={state.prune}")

                # ---- O: toggle gradient pruning ----
                elif event.key == pygame.K_o:
                    state.gradient_prune = not state.gradient_prune
                    print(f"Gradient Prune={state.gradient_prune}")

                # ---- =: toggle prune logic ----
                elif event.key == pygame.K_EQUALS:
                    state.prune_logic = "AND" if state.prune_logic == "OR" else "OR"
                    print(f"Prune Logic={state.prune_logic}")

                # ---- C: change charge delta and gradient threshold ----
                elif event.key == pygame.K_c:
                    if state.charge_change_protection:
                        old_delta = config.charge_delta
                        old_gt = config.gradient_threshold
                        try:
                            config.charge_delta = do_get_user_input_float(
                                f"Enter charge delta ({config.charge_delta}): ", config.charge_delta)
                            config.gradient_threshold = do_get_user_input_float(
                                f"Enter gradient threshold ({config.gradient_threshold}): ", config.gradient_threshold)
                            ui_print(f"Charge delta: {config.charge_delta}, Gradient threshold: {config.gradient_threshold}")
                        except Exception:
                            config.charge_delta = old_delta
                            config.gradient_threshold = old_gt

                # ---- D: toggle display updating ----
                elif event.key == pygame.K_d:
                    state.display_updating = not state.display_updating
                    print(f"Display_updating={state.display_updating}")

                # ---- N: nuke all cells ----
                elif event.key == pygame.K_n:
                    try:
                        confirm = do_pygame_input("Are you sure you want to Nuke all cells? (y/n): ")
                        if confirm and confirm.lower() == 'y':
                            for i in range(1, config.num_layers - 1):
                                state.cells[:, :, i] = None
                            state.not_saved_yet = True
                            state.reset_training_metrics()
                            state.invalidate_neighbor_cache()
                            ui_print("All cells nuked.")
                        else:
                            ui_print("Nuclear option cancelled.")
                    except Exception as e:
                        ui_print(f"An error occurred: {e}")

                # ---- I: change learning rate ----
                elif event.key == pygame.K_i:
                    old_lr = config.learning_rate
                    try:
                        config.learning_rate = do_get_user_input_float(
                            f"Enter learning rate ({old_lr:.4f}): ", old_lr)
                        ui_print(f"Learning rate updated to: {config.learning_rate:.4f}")
                    except ValueError:
                        config.learning_rate = old_lr

                # ---- E: enter parameter resets ----
                elif event.key == pygame.K_e:
                    old_lod = config.length_of_dendrite
                    # Get new values via input
                    config.num_layers = do_get_user_input(f"Enter number of layers ({config.num_layers}, 4-16): ", config.num_layers)
                    if config.num_layers < 3 or config.num_layers > 16:
                        config.num_layers = 8
                    config.length_of_dendrite = do_get_user_input(f"Enter dendrite length ({config.length_of_dendrite}, 1-4): ", config.length_of_dendrite)
                    config.mutation_rate = do_get_user_input(f"Mutation rate/100k ({config.mutation_rate}): ", config.mutation_rate)
                    config.lower_allele_range = do_get_user_input(f"Lower allele ({config.lower_allele_range}): ", config.lower_allele_range)
                    config.upper_allele_range = do_get_user_input(f"Upper allele ({config.upper_allele_range}): ", config.upper_allele_range)
                    config.weight_change_threshold = do_get_user_input_float(f"Weight change threshold ({config.weight_change_threshold:.3f}): ", config.weight_change_threshold)
                    config.avg_weights_cell = do_get_user_input(f"Avg weights/cell ({config.avg_weights_cell}): ", config.avg_weights_cell)
                    config.weight_decay = do_get_user_input_float(f"Weight decay ({config.weight_decay:.3f}): ", config.weight_decay)
                    config.bias_range = do_get_user_input_float(f"Bias range ({config.bias_range:.3f}): ", config.bias_range)
                    config.learning_rate = do_get_user_input_float(f"Learning rate ({config.learning_rate:.4f}): ", config.learning_rate)
                    config.charge_delta = do_get_user_input_float(f"Charge delta ({config.charge_delta:.3f}): ", config.charge_delta)
                    config.gradient_threshold = do_get_user_input_float(f"Gradient threshold ({config.gradient_threshold:.3f}): ", config.gradient_threshold)
                    config.activation_slope = do_get_user_input_float(f"Activation slope ({config.activation_slope:.3f}): ", config.activation_slope)

                    new_wm = 2 * config.length_of_dendrite + 1
                    new_now = new_wm * new_wm
                    if new_now != config.number_of_weights:
                        msg = f"Updating connections: old={config.number_of_weights}, new={new_now}"
                        print(msg)
                        ui_print(msg)
                        for layer in range(1, config.num_layers - 1):
                            for x, y in np.ndindex(state.cells.shape[:2]):
                                if state.cells[x, y, layer] is not None:
                                    state.cells[x, y, layer].remap_weights(config.length_of_dendrite)
                        state.invalidate_neighbor_cache()

                    config.update_derived()
                    msg = f"Updated: dendrite={config.length_of_dendrite}, matrix={config.weight_matrix}, weights={config.number_of_weights}"
                    print(msg)
                    ui_print(msg)

                # ---- X: reset network genes/proteins ----
                elif event.key == pygame.K_x:
                    for z in range(1, config.num_layers - 1):
                        for x, y in np.ndindex(state.cells.shape[:2]):
                            if state.cells[x, y, z] is not None:
                                state.cells[x, y, z].initalize_network_genes(
                                    config.number_of_weights, config.bias_range,
                                    config.avg_weights_cell, config.charge_delta,
                                    config.weight_decay, config.mutation_rate,
                                    cells_array=state.cells
                                )
                                state.cells[x, y, z].color_genes()
                                state.cells[x, y, z].initialize_network_proteins()
                                state.cells[x, y, z].color_proteins()
                    if not config.autonomous_network_genes:
                        ui_print(f"Reset: weights={config.number_of_weights} bias={config.bias_range} avg_w={config.avg_weights_cell} delta={config.charge_delta}")
                    else:
                        ui_print("Autonomous genes: weights/biases set from genes")
                    state.not_saved_yet = True
                    state.reset_training_metrics()

                # ---- S: save ----
                elif event.key == pygame.K_s:
                    save_file(state, config, "", ui_print_fn=ui_print)

                # ---- L: load ----
                elif event.key == pygame.K_l and not state.show_3d_view:
                    load_file(state, config, screen, growthsurface, bottom_caption_surface,
                              font_directory, font, lambda: draw_cells(state, config, growthsurface),
                              lambda: draw_grid(growthsurface), do_pygame_input, ui_print_fn=ui_print)

                # ---- M: load MNIST ----
                elif event.key == pygame.K_m and not state.show_3d_view:
                    load_training_data_main(state, config, do_pygame_input,
                                            draw_fn=do_draw, ui_print_fn=ui_print)

                # ---- F: forward propagation direction ----
                elif event.key == pygame.K_f:
                    state.direction_of_charge_flow = "+++++>>>>>"
                    print(f"Direction={state.direction_of_charge_flow}")

                # ---- R: reverse propagation direction ----
                elif event.key == pygame.K_r:
                    state.direction_of_charge_flow = "<<<<<-----"
                    print(f"Direction={state.direction_of_charge_flow}")

                # ---- B: toggle backprop ----
                elif event.key == pygame.K_b:
                    state.back_prop = not state.back_prop
                    print(f"Back_prop={state.back_prop}")

                # ---- T: toggle training ----
                elif event.key == pygame.K_t:
                    if not state.training_data_loaded:
                        print(f"T pressed but no training data loaded (press M first)")
                    else:
                        state.training_mode = not state.training_mode
                        print(f"training_mode={state.training_mode} | direction={state.direction_of_charge_flow} | back_prop={state.back_prop} | data={config.how_much_training_data} samples")

                # ---- A: toggle andromida ----
                elif event.key == pygame.K_a:
                    state.andromida_mode = not state.andromida_mode
                    print(f"andromida_mode={state.andromida_mode}")

                # ---- W: reset gradients ----
                elif event.key == pygame.K_w:
                    reset_all_gradient_changes(state, config)
                    print("All gradient changes reset")

                # ---- G: toggle genes/proteins display ----
                elif event.key == pygame.K_g and not state.show_3d_view:
                    state.display = "proteins" if state.display == "genes" else "genes"
                    msg = f"display = {state.display}"
                    print(msg)
                    ui_print(msg)

                # ---- V: cycle display sets ----
                elif event.key == pygame.K_v and not state.show_3d_view:
                    state.display_set = (state.display_set + 1) % 3
                    if state.display_set == 0:
                        settings = get_all_settings(state, config)
                        help_surface.fill(WHITE)
                        render_help_text(help_surface, settings, font_small, BLACK, 10, 10, 20)
                        screen.blit(help_surface, (MAIN_SURFACE_WIDTH + 3, 0))
                    elif state.display_set == 1:
                        display_averages(state, config, ui_print)
                    else:
                        ct = update_cell_types(state.cells, config)
                        pt = update_phenotype_cell_types(state.cells, config)
                        display_statistics(ct, ui_print)
                        display_phenotype_statistics(pt, ui_print)
                        display_max_charge_diff(state, config, 5, ui_print)
                    state.show_training_stats = not state.show_training_stats
                    pygame.display.flip()

                # ---- 3: toggle 3D view ----
                elif event.key == pygame.K_3:
                    state.show_3d_view = not state.show_3d_view
                    if state.show_3d_view:
                        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
                        setup_3d_view()
                    else:
                        screen = pygame.display.set_mode((EXTENDED_WINDOW_WIDTH, EXTENDED_WINDOW_HEIGHT))

                # ---- 4: toggle backprop 3D view ----
                elif event.key == pygame.K_4:
                    state.show_3d_view = True
                    state.show_backprop_view = not state.show_backprop_view
                    if state.show_3d_view:
                        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
                        setup_3d_view()
                    else:
                        screen = pygame.display.set_mode((EXTENDED_WINDOW_WIDTH, EXTENDED_WINDOW_HEIGHT))

                # ---- Q: dump telemetry/validation report ----
                elif event.key == pygame.K_q and not state.show_3d_view:
                    telem = compute_telemetry(state, config)
                    report = format_telemetry(telem)
                    print(report)
                    ui_print(report, 10)

            # ---- 3D mouse controls ----
            if state.show_3d_view:
                if event.type == pygame.MOUSEMOTION:
                    if event.buttons[0]:
                        state.rotation_y += event.rel[0]
                        state.rotation_x += event.rel[1]
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        state.zoom += 1
                    elif event.button == 5:
                        state.zoom -= 1

            # ---- Mouse up ----
            if event.type == pygame.MOUSEBUTTONUP:
                state.mouse_up = True

            # ---- Mouse click: create/delete/inspect cells ----
            if event.type == pygame.MOUSEBUTTONDOWN and not state.show_3d_view:
                mx, my = event.pos
                if my < WINDOW_HEIGHT and mx < WINDOW_WIDTH:
                    cell_x, cell_y, layer = convert_x_y_to_index(mx, my)
                    if event.button == 3 or (event.button == 1 and pygame.key.get_mods() & pygame.KMOD_CTRL):
                        if state.cells[cell_x, cell_y, layer] is not None:
                            ct = update_cell_types(state.cells, config)
                            cp, tc, pt = update_phenotype_cell_types(state.cells, config)
                            ui_print(f"Cells {tc} | Positive {cp} | Fraction {cp / (tc + epsilon):.2f}")
                            ui_print(str(state.cells[cell_x, cell_y, layer]))
                        else:
                            ui_print("No cell at this location")
                    else:
                        if state.cells[cell_x, cell_y, layer] is None:
                            state.cells[cell_x, cell_y, layer] = Cell(
                                layer, cell_x, cell_y, config.number_of_weights,
                                config.bias_range, config.avg_weights_cell,
                                config.charge_delta, config.weight_decay, config.mutation_rate
                            )
                        else:
                            state.cells[cell_x, cell_y, layer] = None
                        state.mouse_up = False
                        state.invalidate_neighbor_cache()

            # ---- Mouse drag: paint cells ----
            if event.type == pygame.MOUSEMOTION and not state.show_3d_view:
                if not state.mouse_up:
                    mx, my = event.pos
                    if my < WINDOW_HEIGHT and mx < WINDOW_WIDTH:
                        cell_x, cell_y, layer = convert_x_y_to_index(mx, my)
                        if state.cells[cell_x, cell_y, layer] is None:
                            state.cells[cell_x, cell_y, layer] = Cell(
                                layer, cell_x, cell_y, config.number_of_weights,
                                config.bias_range, config.avg_weights_cell,
                                config.charge_delta, config.weight_decay, config.mutation_rate
                            )
                            state.invalidate_neighbor_cache()

            # ---- QUIT ----
            if event.type == pygame.QUIT:
                try:
                    confirm = input("Are you sure you want to quit? (y/n): ")
                    if confirm.lower() == 'y':
                        state.simulating = False
                        continue
                except Exception:
                    pass

        # ---- Evolution ----
        if state.running:
            update_cells(state, config)
            if state.timing:
                print(f"Evolution tick: andromida={state.andromida_mode} prune={state.prune}")

        # ---- Training ----
        if state.training_mode:
            state.total_loss = 0
            state.total_predictions = 0
            state.training_cycles += 1
            try:
                run_training(state, config, draw_fn=do_draw, render_backprop_fn=do_render_backprop)
                print(f"Cycle {state.training_cycles}: correct={state.bingo_count}/{config.how_much_training_data} loss={state.running_avg_loss:.4f} max={state.max_bingo_count}")
                state._3d_dirty = True
            except Exception as e:
                import traceback
                print(f"Training error: {e}")
                traceback.print_exc()
                ui_print(f"Training error: {e}")
            bottom_caption_surface.fill(BLACK)

            try:
                if state.training_cycles % state.stats_update_frequency == 0:
                    if state.show_training_stats:
                        update_training_stats(state, config, ui_print)
                        display_training_stats(state, ui_print)
            except Exception as e:
                ui_print(f"Stats error: {e}")

        # ---- Display ----
        if state.display_updating:
            if state.show_3d_view:
                if not state.show_backprop_view:
                    render_3d_network(state, config)
                pygame.display.flip()
            else:
                growthsurface.fill(WHITE)
                draw_cells(state, config, growthsurface)
                draw_grid(growthsurface)
                screen.blit(growthsurface, (0, 0))

        # ---- Bottom status bar ----
        if not state.show_3d_view:
            end_time = time.time()
            elapsed_time = end_time - start_time

            if not state.training_mode:
                bottom_caption_surface.fill(BLACK)
            prune_color = GREEN if state.running else WHITE
            if state.prune:
                prune_color = RED

            text_surface = font.render(
                f"Running = {state.running} | Andromida = {state.andromida_mode} | "
                f"Prune = {state.prune}: {state.prune_logic} | "
                f"Charge = {state.charge_change_protection}: {config.charge_delta:.2e}, "
                f"Gradient = {state.gradient_prune}: {config.gradient_threshold:.2e}",
                True, prune_color
            )
            text_surface1 = font.render(
                f"Training = {state.training_mode}: {config.learning_rate:.4f} | "
                f"{state.direction_of_charge_flow} | Back_prop = {state.back_prop} | "
                f"Auto Genes = {config.autonomous_network_genes} | "
                f"Learning Rate: {config.learning_rate:.4f} | Cycles: {state.training_cycles}",
                True, WHITE
            )
            text_surface2 = font.render(
                f"Elapsed: {elapsed_time:.2f} | Training_data: {config.how_much_training_data} | "
                f"Loss: {state.running_avg_loss:.4f} | Correct: {state.bingo_count} | "
                f"Max Correct: {state.max_bingo_count}",
                True, WHITE
            )

            bottom_caption_surface.fill(BLACK)
            bottom_caption_surface.blit(text_surface, (10, 10))
            bottom_caption_surface.blit(text_surface1, (10, 40))
            bottom_caption_surface.blit(text_surface2, (10, 70))
            if state.training_mode:
                prediction_plot(state, bottom_caption_surface)

            screen.blit(bottom_caption_surface, (0, WINDOW_HEIGHT))
            pygame.display.flip()

            start_time = time.time()

    pygame.quit()


if __name__ == '__main__':
    main()
