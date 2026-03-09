"""3D OpenGL visualization of the neural network."""
import pygame
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

from neurosim.config import WINDOW_WIDTH, WINDOW_HEIGHT, WIDTH, HEIGHT


def setup_3d_view():
    """Initialize OpenGL viewport and perspective."""
    if not HAS_OPENGL:
        print("OpenGL not available — 3D view disabled")
        return
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)


def render_3d_network(state, config):
    """Render neurons as points and connections as lines."""
    if not HAS_OPENGL:
        return
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, state.zoom)
    glRotatef(state.rotation_x, 1, 0, 0)
    glRotatef(state.rotation_y, 0, 1, 0)

    # Draw neurons
    glPointSize(10)
    glBegin(GL_POINTS)
    for layer in range(config.num_layers):
        z = layer * 2
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if state.cells[x, y, layer] is not None:
                    charge_intensity = state.cells[x, y, layer].charge
                    ci = min(int(abs(charge_intensity) * 255), 255)
                    glColor3f(ci / 255, 0, 0)
                    glVertex3f(x - WIDTH / 2, y - HEIGHT / 2, z - config.num_layers)
    glEnd()

    # Draw connections
    glLineWidth(1)
    glBegin(GL_LINES)
    for layer in range(1, config.num_layers):
        z = layer * 2
        prev_z = (layer - 1) * 2
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if state.cells[x, y, layer] is not None:
                    if layer == config.num_layers - 1:
                        if state.cells[x, y, layer - 1] is not None:
                            glColor3f(0, 1, 0)
                            glVertex3f(x - WIDTH / 2, y - HEIGHT / 2, z - config.num_layers)
                            glVertex3f(x - WIDTH / 2, y - HEIGHT / 2, prev_z - config.num_layers)
                    else:
                        reach = config.length_of_dendrite
                        for dx in range(-reach, reach + 1):
                            for dy in range(-reach, reach + 1):
                                if 0 <= x + dx < WIDTH and 0 <= y + dy < HEIGHT:
                                    if state.cells[x + dx, y + dy, layer - 1] is not None:
                                        weight_index = (dx + reach) * config.weight_matrix + (dy + reach)
                                        if weight_index < len(state.cells[x, y, layer].weights):
                                            weight = state.cells[x, y, layer].weights[weight_index]
                                            if weight > 0:
                                                glColor3f(0, min(weight, 1), 0)
                                            else:
                                                glColor3f(min(-weight, 1), 0, 0)
                                            glVertex3f(x - WIDTH / 2, y - HEIGHT / 2, z - config.num_layers)
                                            glVertex3f(x + dx - WIDTH / 2, y + dy - HEIGHT / 2, prev_z - config.num_layers)
    glEnd()


def render_3d_backprop(state, config, current_layer, current_pos):
    """Render backprop visualization with error/gradient coloring."""
    if not HAS_OPENGL:
        return
    try:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, state.zoom)
        glRotatef(state.rotation_x, 1, 0, 0)
        glRotatef(state.rotation_y, 0, 1, 0)

        # Draw all neurons
        glPointSize(10)
        glBegin(GL_POINTS)
        try:
            for layer in range(config.num_layers):
                z = layer * 2
                for x in range(WIDTH):
                    for y in range(HEIGHT):
                        if state.cells[x, y, layer] is not None:
                            try:
                                if layer == current_layer and (x, y) == current_pos:
                                    glColor3f(1.0, 1.0, 0.0)
                                elif layer == current_layer + 1:
                                    ei = min(max(abs(state.cells[x, y, layer].error), 0.0), 1.0)
                                    glColor3f(ei, 0, ei)
                                elif layer == current_layer - 1:
                                    gi = min(max(abs(state.cells[x, y, layer].gradient), 0.0), 1.0)
                                    glColor3f(1.0, gi, 0.0)
                                else:
                                    glColor3f(0.2, 0.2, 0.4)
                                ci = min(abs(state.cells[x, y, layer].charge), 1.0)
                                r, g, b = 0.2 + 0.8 * ci, 0.2 + 0.8 * ci, 0.4
                                glColor3f(r, g, b)
                                glVertex3f(float(x - WIDTH / 2), float(y - HEIGHT / 2), float(z - config.num_layers))
                            except Exception as e:
                                continue
        finally:
            glEnd()

        # Draw connections for current neuron
        x, y = current_pos
        if state.cells[x, y, current_layer] is not None:
            current_cell = state.cells[x, y, current_layer]
            z = current_layer * 2
            reach = config.length_of_dendrite

            glLineWidth(2)
            glBegin(GL_LINES)
            try:
                if current_layer < config.num_layers - 1:
                    for dx in range(-reach, reach + 1):
                        for dy in range(-reach, reach + 1):
                            try:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                                    next_cell = state.cells[nx, ny, current_layer + 1]
                                    if next_cell is not None:
                                        wi = (dx + reach) * config.weight_matrix + (dy + reach)
                                        if wi < len(next_cell.weights):
                                            weight = next_cell.weights[wi]
                                            error = next_cell.error
                                            w_int = min(max(abs(weight), 0.0), 1.0)
                                            e_int = min(max(abs(error), 0.0), 1.0)
                                            if weight > 0:
                                                glColor3f(0, w_int, e_int)
                                            else:
                                                glColor3f(w_int, 0, e_int)
                                            glVertex3f(float(x - WIDTH / 2), float(y - HEIGHT / 2), float(z - config.num_layers))
                                            glVertex3f(float(nx - WIDTH / 2), float(ny - HEIGHT / 2), float((current_layer + 1) * 2 - config.num_layers))
                            except Exception:
                                continue

                try:
                    if current_layer > 0:
                        gi = min(max(abs(current_cell.gradient), 0.0), 1.0)
                        glColor3f(1.0, 0.5 * gi, 0.0)
                        arrow_length = 0.5
                        glVertex3f(float(x - WIDTH / 2), float(y - HEIGHT / 2), float(z - config.num_layers))
                        glVertex3f(float(x - WIDTH / 2), float(y - HEIGHT / 2), float(z - config.num_layers - arrow_length))
                except Exception:
                    pass
            finally:
                glEnd()
    except Exception as e:
        print(f"Error in render_3d_backprop: {e}")


def render_3d_network_no_connections(state, config):
    """Simplified 3D render without connections."""
    if not HAS_OPENGL:
        return
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, state.zoom)
    glRotatef(state.rotation_x, 1, 0, 0)
    glRotatef(state.rotation_y, 0, 1, 0)

    glPointSize(5)
    glBegin(GL_POINTS)
    for layer in range(config.num_layers):
        z = layer * 2
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if state.cells[x, y, layer] is not None:
                    glColor3fv([c / 255 for c in state.cells[x, y, layer].colors[0][:3]])
                    glVertex3f(x - WIDTH / 2, y - HEIGHT / 2, z - config.num_layers)
    glEnd()
    pygame.display.flip()
