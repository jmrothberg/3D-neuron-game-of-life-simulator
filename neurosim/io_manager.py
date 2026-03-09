"""File I/O: save/load simulation states, load MNIST training data."""
import os
import sys
import platform as sys_platform
import pickle
import copy
import datetime
import math
import numpy as np
import pygame
from PIL import Image

from neurosim.config import (WIDTH, HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT, CELL_SIZE,
                              WHITE, BLACK, FASHION_LABELS)
from neurosim.cell import Cell


# Platform-specific training data paths
system = sys_platform.system()
if system == "Darwin":
    input_path_training_Digits = '/Users/jonathanrothberg/MNIST_5000_0_15_Cells'
    input_path_training_Fashion = '/Users/jonathanrothberg/Fashion_MNIST_5000'
elif system == "Linux":
    if "Ubuntu" in sys_platform.version():
        input_path_training_Digits = '/data/MNIST_5000_0_15_Cells'
        input_path_training_Fashion = '/data/Fashion_MNIST_5000'
    else:
        input_path_training_Digits = '/data/MNIST_5000_0_15_Cells'
        input_path_training_Fashion = '/data/Fashion_MNIST_5000'
else:
    input_path_training_Digits = './MNIST_5000_0_15_Cells'
    input_path_training_Fashion = './Fashion_MNIST_5000'


class CompatUnpickler(pickle.Unpickler):
    """Custom unpickler that handles old pickle files where Cell was in __main__."""
    def find_class(self, module, name):
        if name == 'Cell':
            return Cell
        return super().find_class(module, name)


def create_icon(filename, cell_size=1):
    """Create a pygame surface icon from a pickled cell array."""
    with open(filename, 'rb') as f:
        cells = CompatUnpickler(f).load()

    icon = np.ones((112 * cell_size, 112 * cell_size))
    number_of_layers = cells.shape[2]
    for k in range(number_of_layers):
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                if cells[i, j, k] is not None:
                    icon_x = (k // 4) * 28 * cell_size + j * cell_size
                    icon_y = (k % 4) * 28 * cell_size + i * cell_size
                    icon[icon_x:icon_x + cell_size, icon_y:icon_y + cell_size] = 0

    img = Image.fromarray((icon * 255).astype(np.uint8))
    img = img.convert("RGB")
    return pygame.image.fromstring(img.tobytes(), img.size, img.mode)


def parse_file_name(file_name):
    """Extract num_layers and num_weights from filename."""
    try:
        parts = file_name.split('_')
        num_layers = int(parts[1])
        num_weights = int(parts[2])
    except Exception as e:
        print("Error in parse_file_name", e)
        print(file_name)
        num_layers = 8
        num_weights = 25
    return num_layers, num_weights


def save_file(state, config, tag, ui_print_fn=None):
    """Save cells to .pkl, icon .png, and metadata .txt."""
    timestamp = datetime.datetime.now().strftime("%d-%H-%M-%S")
    file_dir = "./saved_states/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    filename = (f"{file_dir}sim_{config.num_layers}_{config.number_of_weights}_"
                f"-{timestamp}_{state.bingo_count}_{config.how_much_training_data}"
                f"-{config.start_index}{tag}")
    pkl_filename = f"{filename}.pkl"
    txt_filename = f"{filename}.txt"
    icon_filename = f"{filename}.png"

    with open(pkl_filename, 'wb') as f:
        pickle.dump(state.cells, f)
    msg = f"Simulation state saved to {pkl_filename}!"
    print(msg)
    if ui_print_fn:
        ui_print_fn(msg)

    icon = create_icon(pkl_filename, 3)
    pygame.image.save(icon, icon_filename)

    epsilon = config.epsilon
    with open(txt_filename, 'w') as f:
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"Cell Settings:\n")
        f.write(f"mutation_rate: {config.mutation_rate}\n")
        f.write(f"lower_allele_range: {config.lower_allele_range}\n")
        f.write(f"upper_allele_range: {config.upper_allele_range}\n")
        f.write(f"Simulation Settings:\n")
        f.write(f"NUM_LAYERS: {config.num_layers}\n")
        f.write(f"LENGTH_OF_DENDRITE: {config.length_of_dendrite}\n")
        f.write(f"Bias_Range: {config.bias_range}\n")
        f.write(f"Weight Range based on estimate Avg weights/cell: {config.avg_weights_cell}\n")
        f.write(f"Weight decay: {config.weight_decay}\n")
        f.write(f"Charge delta (pruning/protection threshold): {config.charge_delta}\n")
        f.write(f"Gradient threshold: {config.gradient_threshold}\n")
        f.write(f"Activation slope (leaky ReLU): {config.activation_slope}\n")
        f.write(f"learning_rate: {config.learning_rate}\n")
        f.write(f"how_much_training_data: {config.how_much_training_data}\n")
        f.write(f"start_index: {config.start_index}\n")
        f.write(f"training_cycles: {state.training_cycles}\n")
        f.write(f"Bingo count: {state.bingo_count}\n")
        f.write(f"max_bingo_count: {state.max_bingo_count}\n")
        f.write(f"Loss: {state.running_avg_loss}\n")
        f.write(f"Number of cells: {state.total_cells}\n")
        f.write(f"Number of weights: {state.total_weights_list[0]}\n")
        f.write(f"Weight/Cell: {state.total_weights_list[0] / (state.total_cells + epsilon):.2f}\n")


def update_cell_coordinates(cells):
    """Fix cell x/y coordinates after loading (for backward compat)."""
    for layer in range(cells.shape[2]):
        for x in range(cells.shape[0]):
            for y in range(cells.shape[1]):
                if cells[x, y, layer] is not None:
                    if not hasattr(cells[x, y, layer], 'x') or cells[x, y, layer].x == 0:
                        cells[x, y, layer].x = x
                    if not hasattr(cells[x, y, layer], 'y') or cells[x, y, layer].y == 0:
                        cells[x, y, layer].y = y


def load_file(state, config, screen, growthsurface, bottom_caption_surface, font_directory, font,
              draw_cells_fn, draw_grid_fn, pygame_input_fn, ui_print_fn=None):
    """Interactive file browser UI for loading saved states."""
    file_dir = "./saved_states/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_list = [f for f in os.listdir(file_dir) if f.startswith("sim") and f.endswith(".pkl")]

    if len(file_list) == 0:
        msg = "\nNo saved simulation states found!"
        print(msg)
        if ui_print_fn:
            ui_print_fn(msg)
        return

    growthsurface.fill(WHITE)
    bottom_caption_surface.fill(BLACK)
    screen.blit(growthsurface, (0, 0))
    pygame.display.flip()

    page = 0
    cell_size = 1
    no_file_selected = True
    new_page = True
    NUM_LAY = [0] * 1000
    num_wei = [0] * 1000

    while no_file_selected:
        if new_page:
            growthsurface.fill(WHITE)
            bottom_caption_surface.fill(BLACK)
            draw_grid_fn()
            for i, file_name in enumerate(file_list[page * 16:(page + 1) * 16]):
                real_i = i + page * 16
                icon = create_icon(os.path.join(file_dir, file_name), cell_size)
                icon_x = (i % 4) * 252 * cell_size + 75
                icon_y = (i // 4) * 252 * cell_size + 50
                screen.blit(icon, (icon_x, icon_y))
                pygame.draw.rect(screen, WHITE, (icon_x - 65, icon_y + 112 * cell_size - 5, 230, 70))
                NUM_LAY[real_i], num_wei[real_i] = parse_file_name(file_name)

                max_width = 220
                truncated_name = file_name
                while font_directory.size(truncated_name)[0] > max_width:
                    truncated_name = truncated_name[:-1]
                if truncated_name != file_name:
                    truncated_name += "..."

                text1 = font_directory.render(truncated_name, True, (0, 0, 0))
                text2 = font_directory.render(f"Layers: {NUM_LAY[real_i]} | Weights {num_wei[real_i]}", True, (0, 0, 0))
                screen.blit(text1, (icon_x - 65, icon_y + 112 * cell_size + 20))
                screen.blit(text2, (icon_x - 65, icon_y + 112 * cell_size + 45))

            text_surface1 = font.render("Click on a file to load it", True, WHITE)
            bottom_caption_surface.blit(text_surface1, (50, 10))
            text_surface2 = font.render("Use Left & Right Arrow keys to scroll pages", True, WHITE)
            bottom_caption_surface.blit(text_surface2, (50, 40))
            text_surface3 = font.render("Press ESC to return", True, WHITE)
            bottom_caption_surface.blit(text_surface3, (50, 70))

            from neurosim.config import EXTENDED_WINDOW_HEIGHT, WINDOW_EXTENSION
            screen.blit(bottom_caption_surface, (0, EXTENDED_WINDOW_HEIGHT - WINDOW_EXTENSION))
            pygame.display.flip()

        new_page = False

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for i in range(16):
                    icon_x = (i % 4) * 252 * cell_size + 75
                    icon_y = (i // 4) * 252 * cell_size + 50
                    if icon_x <= mouse_x < icon_x + 252 * cell_size and icon_y <= mouse_y < icon_y + 252 * cell_size:
                        selection = i + page * 16
                        if selection < len(file_list):
                            file_path = os.path.join(file_dir, file_list[selection])
                            try:
                                with open(file_path, 'rb') as f:
                                    state.cells = CompatUnpickler(f).load()
                                # Restore Cell config reference after unpickling
                                Cell.set_config(config)

                                config.num_layers = NUM_LAY[selection]
                                config.number_of_weights = num_wei[selection]
                                config.weight_matrix = int(math.sqrt(config.number_of_weights))
                                config.length_of_dendrite = int((config.weight_matrix - 1) / 2)

                                update_cell_coordinates(state.cells)

                                # Count actual active layers
                                active_layers = [l for l in range(state.cells.shape[2])
                                                 if any(state.cells[x2, y2, l] is not None
                                                        for x2 in range(state.cells.shape[0])
                                                        for y2 in range(state.cells.shape[1]))]
                                msg = (f"Loaded from {file_path} | Layers: {config.num_layers} | "
                                       f"Weights: {config.number_of_weights} | Dendrite: {config.length_of_dendrite} | "
                                       f"Active layers: {active_layers}")
                                print(msg)
                                if ui_print_fn:
                                    ui_print_fn(msg)

                                state.not_saved_yet = True
                                state.max_bingo_count = 0
                                state.bingo_count = 0
                                state.total_loss = 0
                                state.total_predictions = 0
                                state.running_avg_loss = 0
                                state.training_cycles = 0
                                state.points = []
                                state.invalidate_neighbor_cache()
                                no_file_selected = False
                            except Exception as e:
                                print(f"Error loading file: {e}")
                                if ui_print_fn:
                                    ui_print_fn(f"Error loading file: {e}")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    no_file_selected = False
                elif event.key == pygame.K_RIGHT:
                    if (page + 1) * 16 < len(file_list):
                        page += 1
                        new_page = True
                elif event.key == pygame.K_LEFT:
                    if page > 0:
                        page -= 1
                        new_page = True

    growthsurface.fill(WHITE)
    bottom_caption_surface.fill(BLACK)
    screen.blit(growthsurface, (0, 0))
    pygame.display.flip()


def load_layers(input_path_training, image_index, state, config):
    """Load a single MNIST training pair into layers 0 and Num_Layers-1."""
    file_path = os.path.join(input_path_training, f'simulation_state_layers_0_and_15_image_{image_index}.pkl')
    with open(file_path, 'rb') as f:
        state.cells[:, :, 0], state.cells[:, :, config.num_layers - 1] = pickle.load(f)
    state.cells[:, :, 0] = np.flip(np.rot90(state.cells[:, :, 0], 3), 1)
    state.cells[:, :, config.num_layers - 1] = np.flip(np.rot90(state.cells[:, :, config.num_layers - 1], 3), 1)


def load_training_data(input_path_training, state, config, draw_fn=None, ui_print_fn=None):
    """Batch load training data into state."""
    state.training_data_layer_0 = []
    state.training_data_num_layer_minus_1 = []
    print(f"Loading training data from: {input_path_training}")
    if ui_print_fn:
        ui_print_fn(f"Loading training data from: {input_path_training}")

    for k in range(config.start_index, config.start_index + config.how_much_training_data):
        try:
            load_layers(input_path_training, k, state, config)
            state.training_data_layer_0.append(copy.deepcopy(state.cells[:, :, 0]))
            state.training_data_num_layer_minus_1.append(copy.deepcopy(state.cells[:, :, config.num_layers - 1]))
            if draw_fn:
                draw_fn()
        except Exception as e:
            print(f"Error loading data at index {k}: {e}")
            if ui_print_fn:
                ui_print_fn(f"Error loading data at index {k}: {e}")
            continue


def load_training_data_main(state, config, pygame_input_fn, draw_fn=None, ui_print_fn=None):
    """UI wrapper for loading MNIST data."""
    default_training_data = config.how_much_training_data
    default_start_index = config.start_index
    try:
        which_data_set = pygame_input_fn("Enter which data set (M for MNIST or F for Fashion MNIST): ", "M")
        if which_data_set.lower() == "f":
            training_data = input_path_training_Fashion
            print(FASHION_LABELS)
        else:
            training_data = input_path_training_Digits
        config.how_much_training_data = int(pygame_input_fn("Enter training set size (20, 1 to 1000): ", 20))
        config.start_index = int(pygame_input_fn("Start index (0, 0 to 999): ", 0))
    except ValueError:
        msg = "Invalid input. No New Data loaded"
        print(msg)
        if ui_print_fn:
            ui_print_fn(msg)
    else:
        if config.start_index + config.how_much_training_data > 5000:
            config.how_much_training_data = default_training_data
            config.start_index = default_start_index
            msg = f"Total can't exceed 5000. Returning to defaults {config.how_much_training_data} {config.start_index}"
            print(msg)
            if ui_print_fn:
                ui_print_fn(msg)
        else:
            try:
                print(f"Loading from {training_data}, start={config.start_index}, count={config.how_much_training_data}")
                load_training_data(training_data, state, config, draw_fn=draw_fn, ui_print_fn=ui_print_fn)
                state.total_weights_list = np.zeros(config.how_much_training_data)
                state.training_data_loaded = True
                state.not_saved_yet = True
                state.reset_training_metrics()
                print(f"Training data loaded OK: {len(state.training_data_layer_0)} samples, training_data_loaded={state.training_data_loaded}")
            except Exception as e:
                msg = f"Error loading training data: {str(e)}"
                print(msg)
                if ui_print_fn:
                    ui_print_fn(msg)
