"""UI functions: input dialogs, side panel, help text rendering."""
import pygame

from neurosim.config import (BLACK, WHITE, LIGHT_GRAY, WINDOW_WIDTH, WINDOW_HEIGHT,
                              WINDOW_EXTENSION, EXTENDED_WINDOW_HEIGHT,
                              HELP_PANEL_WIDTH, MAIN_SURFACE_WIDTH, CELL_SIZE)


def render_help_text(surface, text, font, color, x, y, line_height):
    """Render multi-line help text to a surface."""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        text_surface = font.render(line.strip(), True, color)
        surface.blit(text_surface, (x, y + i * line_height))


def print_to_side_panel(text, state, screen, help_surface, font_small, show_3d_view, position=None):
    """Print scrolling text to the right side panel."""
    if show_3d_view:
        return
    if position is None:
        lines = text.split('\n')
        state.side_panel_text.extend(lines)
        state.side_panel_text = state.side_panel_text[-50:]

        help_surface.fill(WHITE)
        y = 10
        for line in state.side_panel_text:
            text_surface = font_small.render(line, True, BLACK)
            help_surface.blit(text_surface, (10, y))
            y += 20
    else:
        state.side_panel_text = []
        help_surface.fill(WHITE, (0, position, HELP_PANEL_WIDTH, WINDOW_HEIGHT))
        lines = text.split('\n')
        for i, line in enumerate(lines):
            text_surface = font_small.render(line, True, BLACK)
            help_surface.blit(text_surface, (10, position + i * 20))

    screen.blit(help_surface, (MAIN_SURFACE_WIDTH + 3, 0))
    pygame.display.update(pygame.Rect(MAIN_SURFACE_WIDTH + 3, 0, HELP_PANEL_WIDTH, WINDOW_HEIGHT))


def pygame_input(prompt, screen, bottom_caption_surface, font, clock, default_value=None):
    """Interactive text input dialog at bottom of screen."""
    input_box = pygame.Rect(50, 10, 200, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_active
    active = True
    text = str(default_value) if default_value is not None else ""
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return default_value
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        done = True
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode

        bottom_caption_surface.fill(BLACK)
        txt_surface = font.render(prompt + text, True, color)
        width = max(200, txt_surface.get_width() + 10)
        input_box.w = width
        pygame.draw.rect(bottom_caption_surface, color, input_box, 2)
        bottom_caption_surface.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        screen.blit(bottom_caption_surface, (0, EXTENDED_WINDOW_HEIGHT - WINDOW_EXTENSION))
        pygame.display.flip()
        clock.tick(30)

    return text if text else default_value


def get_user_input(prompt, default_value, screen, bottom_caption_surface, font, clock):
    """Get integer input from user."""
    try:
        user_input = pygame_input(prompt, screen, bottom_caption_surface, font, clock, default_value)
        if user_input == "":
            return default_value
        return int(user_input)
    except:
        print("Invalid input")
        return default_value


def get_user_input_float(prompt, default_value, screen, bottom_caption_surface, font, clock):
    """Get float input from user."""
    try:
        user_input = pygame_input(prompt, screen, bottom_caption_surface, font, clock, default_value)
        if user_input == "":
            return default_value
        return float(user_input)
    except:
        print("Invalid input")
        return default_value


def convert_x_y_to_index(x, y):
    """Convert pixel coordinates to cell grid coordinates and layer."""
    layer_x = x // (WINDOW_WIDTH // 4)
    layer_y = y // (WINDOW_HEIGHT // 4)
    layer = layer_x + layer_y * 4

    adjusted_x = x - layer_x * (WINDOW_WIDTH // 4)
    adjusted_y = y - layer_y * (WINDOW_HEIGHT // 4)

    from neurosim.config import WIDTH, HEIGHT
    cell_x = min(adjusted_x // CELL_SIZE, WIDTH - 1)
    cell_y = min(adjusted_y // CELL_SIZE, HEIGHT - 1)
    return cell_x, cell_y, layer


def get_all_settings(state, config):
    """Return formatted string of all current settings."""
    return f"""Current Settings:
    Num_Layers: {config.num_layers}
    LENGTH_OF_DENDRITE: {config.length_of_dendrite}
    WEIGHT_MATRIX: {config.weight_matrix}
    NUMBER_OF_WEIGHTS: {config.number_of_weights}
    Mutation Rate Per 10,000 Cycles: {config.mutation_rate}
    Lower Allele Range: {config.lower_allele_range}
    Upper Allele Range: {config.upper_allele_range}
    Weight Change Threshold: {config.weight_change_threshold}
    Learning Rate: {config.learning_rate}
    Bias Range: {config.bias_range}
    Avg Weights per Cell: {config.avg_weights_cell}
    Weight Decay: {config.weight_decay}
    Charge Delta: {config.charge_delta}
    Gradient Clip Range: {config.gradient_clip_range}
    Training Data Size: {config.how_much_training_data}
    Start Index: {config.start_index}
    Display Mode: {state.display}
    Direction of Charge Flow: {state.direction_of_charge_flow}
    Prune Logic: {state.prune_logic}
    """
