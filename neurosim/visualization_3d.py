"""3D OpenGL visualization — cached vertex arrays, layer colors, HUD, connection filtering."""
import numpy as np
import pygame

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

from neurosim.config import WINDOW_WIDTH, WINDOW_HEIGHT, WIDTH, HEIGHT

# ---------------------------------------------------------------------------
# Module-level vertex cache — rebuilt only when state._3d_dirty is True
# ---------------------------------------------------------------------------
_neuron_verts = None       # [N, 3] float32
_neuron_colors = None      # [N, 3] float32
_conn_verts = None         # [M*2, 3] float32 (pairs of endpoints)
_conn_colors = None        # [M*2, 3] float32
_input_range = (0, 0)      # (start_index, count) into neuron arrays
_hidden_range = (0, 0)
_output_range = (0, 0)
_cell_count = 0
_conn_count = 0

# HUD texture cache
_hud_texture_id = None
_hud_text_cache = ""
_hud_w = 0
_hud_h = 0
_hud_font = None

# Layer label textures: list of (texture_id, w, h, x, y, z)
_label_textures = []
_label_cache_layers = -1

# Weight threshold — connections weaker than this are not drawn
WEIGHT_DRAW_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# HSV → RGB (inline, no imports)
# ---------------------------------------------------------------------------
def _hsv_to_rgb(h, s, v):
    """Convert HSV (h in 0-1, s in 0-1, v in 0-1) to RGB tuple."""
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)


def _layer_color(layer, num_layers, charge):
    """Color a neuron by its layer position and charge magnitude.

    Hue ramp: blue (input) → cyan → green (mid hidden) → yellow → red (output).
    Brightness scales with |charge|.
    """
    if num_layers <= 1:
        t = 0.5
    else:
        t = layer / (num_layers - 1)
    # Hue: 0.6 (blue) → 0.0 (red)
    h = 0.6 * (1.0 - t)
    s = 0.85
    v = 0.3 + 0.7 * min(abs(charge), 1.0)
    return _hsv_to_rgb(h, s, v)


# ---------------------------------------------------------------------------
# Cache rebuild
# ---------------------------------------------------------------------------
def rebuild_3d_cache(state, config):
    """Build flat numpy arrays of all neuron positions/colors and connection
    line endpoints/colors.  Called once per dirty flag, NOT every frame."""
    global _neuron_verts, _neuron_colors, _conn_verts, _conn_colors
    global _input_range, _hidden_range, _output_range
    global _cell_count, _conn_count

    cells = state.cells
    nl = config.num_layers

    # --- Pass 1: count neurons per category ---
    n_input = n_hidden = n_output = 0
    for layer in range(nl):
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if cells[x, y, layer] is not None:
                    if layer == 0:
                        n_input += 1
                    elif layer == nl - 1:
                        n_output += 1
                    else:
                        n_hidden += 1

    total_neurons = n_input + n_hidden + n_output
    _cell_count = total_neurons

    if total_neurons == 0:
        _neuron_verts = np.empty((0, 3), dtype=np.float32)
        _neuron_colors = np.empty((0, 3), dtype=np.float32)
        _conn_verts = np.empty((0, 3), dtype=np.float32)
        _conn_colors = np.empty((0, 3), dtype=np.float32)
        _input_range = (0, 0)
        _hidden_range = (0, 0)
        _output_range = (0, 0)
        _conn_count = 0
        state._3d_dirty = False
        return

    # Allocate neuron arrays — sorted: input, hidden, output
    nv = np.empty((total_neurons, 3), dtype=np.float32)
    nc = np.empty((total_neurons, 3), dtype=np.float32)

    # Estimate connection count (will trim later)
    conn_list_v = []
    conn_list_c = []

    idx_input = 0
    idx_hidden = n_input
    idx_output = n_input + n_hidden

    half_w = WIDTH / 2.0
    half_h = HEIGHT / 2.0

    for layer in range(nl):
        z = float(layer * 2 - nl)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                cell = cells[x, y, layer]
                if cell is None:
                    continue

                px = float(x) - half_w
                py = float(y) - half_h

                r, g, b = _layer_color(layer, nl, cell.charge)

                if layer == 0:
                    idx = idx_input
                    idx_input += 1
                elif layer == nl - 1:
                    idx = idx_output
                    idx_output += 1
                else:
                    idx = idx_hidden
                    idx_hidden += 1

                nv[idx] = (px, py, z)
                nc[idx] = (r, g, b)

                # --- Connections (hidden + output layers only) ---
                if layer == 0:
                    continue

                if layer == nl - 1:
                    # Output layer: single vertical connection to same (x,y) one layer up
                    if cells[x, y, layer - 1] is not None:
                        prev_z = float((layer - 1) * 2 - nl)
                        conn_list_v.append((px, py, z))
                        conn_list_v.append((px, py, prev_z))
                        conn_list_c.append((0.2, 0.9, 0.2))
                        conn_list_c.append((0.2, 0.9, 0.2))
                else:
                    reach = cell.reach if config.autonomous_network_genes else config.length_of_dendrite
                    prev_z = float((layer - 1) * 2 - nl)
                    wm = config.weight_matrix
                    for dx in range(-reach, reach + 1):
                        nx = x + dx
                        if nx < 0 or nx >= WIDTH:
                            continue
                        for dy in range(-reach, reach + 1):
                            ny = y + dy
                            if ny < 0 or ny >= HEIGHT:
                                continue
                            if cells[nx, ny, layer - 1] is None:
                                continue
                            wi = (dx + reach) * wm + (dy + reach)
                            if wi >= len(cell.weights):
                                continue
                            w = cell.weights[wi]
                            aw = abs(w)
                            if aw < WEIGHT_DRAW_THRESHOLD:
                                continue
                            intensity = min(aw, 1.0)
                            if w > 0:
                                cr, cg, cb = 0.0, intensity, 0.0
                            else:
                                cr, cg, cb = intensity, 0.0, 0.0
                            npx = float(nx) - half_w
                            npy = float(ny) - half_h
                            conn_list_v.append((px, py, z))
                            conn_list_v.append((npx, npy, prev_z))
                            conn_list_c.append((cr, cg, cb))
                            conn_list_c.append((cr, cg, cb))

    _neuron_verts = nv
    _neuron_colors = nc
    _input_range = (0, n_input)
    _hidden_range = (n_input, n_hidden)
    _output_range = (n_input + n_hidden, n_output)

    if conn_list_v:
        _conn_verts = np.array(conn_list_v, dtype=np.float32)
        _conn_colors = np.array(conn_list_c, dtype=np.float32)
        _conn_count = len(conn_list_v) // 2
    else:
        _conn_verts = np.empty((0, 3), dtype=np.float32)
        _conn_colors = np.empty((0, 3), dtype=np.float32)
        _conn_count = 0

    state._3d_dirty = False


# ---------------------------------------------------------------------------
# HUD text rendering (pygame font → OpenGL texture)
# ---------------------------------------------------------------------------
def _get_hud_font():
    global _hud_font
    if _hud_font is None:
        pygame.font.init()
        _hud_font = pygame.font.SysFont("monospace", 14)
    return _hud_font


def _build_hud_texture(state, config):
    """Build a single OpenGL texture with all HUD text lines."""
    global _hud_texture_id, _hud_text_cache, _hud_w, _hud_h

    lines = [
        f"Layers: {config.num_layers} | Cells: {_cell_count} | Conns: {_conn_count}",
        f"Cycle: {state.training_cycles} | Correct: {state.bingo_count}/{config.how_much_training_data}",
        f"Loss: {state.running_avg_loss:.4f} | Max correct: {state.max_bingo_count}",
        f"LR: {config.learning_rate:.4f} | Dendrite: {config.length_of_dendrite} | BackProp: {state.back_prop}",
    ]
    text = "\n".join(lines)
    if text == _hud_text_cache and _hud_texture_id is not None:
        return  # no change

    _hud_text_cache = text
    font = _get_hud_font()

    # Render lines to a pygame surface
    line_surfaces = [font.render(line, True, (255, 255, 255)) for line in lines]
    max_w = max(s.get_width() for s in line_surfaces)
    total_h = sum(s.get_height() for s in line_surfaces)

    # Power-of-two dimensions for texture (some drivers need this)
    tw = 1
    while tw < max_w + 8:
        tw *= 2
    th = 1
    while th < total_h + 8:
        th *= 2

    surf = pygame.Surface((tw, th), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 160))  # semi-transparent black background
    y_off = 4
    for ls in line_surfaces:
        surf.blit(ls, (4, y_off))
        y_off += ls.get_height()

    # Convert to OpenGL texture
    tex_data = pygame.image.tostring(surf, "RGBA", True)
    if _hud_texture_id is None:
        _hud_texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, _hud_texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
    _hud_w = tw
    _hud_h = th


def _render_hud():
    """Draw HUD texture in screen-space (orthographic overlay)."""
    if _hud_texture_id is None:
        return

    # Save 3D state
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    # Draw textured quad at top-left
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBindTexture(GL_TEXTURE_2D, _hud_texture_id)
    glColor4f(1, 1, 1, 1)

    x0, y0 = 5, WINDOW_HEIGHT - _hud_h - 5
    x1, y1 = x0 + _hud_w, y0 + _hud_h

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x0, y0)
    glTexCoord2f(1, 0); glVertex2f(x1, y0)
    glTexCoord2f(1, 1); glVertex2f(x1, y1)
    glTexCoord2f(0, 1); glVertex2f(x0, y1)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)

    # Restore 3D state
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


# ---------------------------------------------------------------------------
# Layer labels (world-space text at each layer's z-position)
# ---------------------------------------------------------------------------
def _build_label_textures(config):
    """Create small textures for layer labels."""
    global _label_textures, _label_cache_layers

    if config.num_layers == _label_cache_layers and _label_textures:
        return
    _label_cache_layers = config.num_layers

    # Delete old textures
    for tex_id, _, _ , _, _, _ in _label_textures:
        try:
            glDeleteTextures([tex_id])
        except Exception:
            pass
    _label_textures = []

    font = _get_hud_font()
    half_w = WIDTH / 2.0
    half_h = HEIGHT / 2.0

    for layer in range(config.num_layers):
        if layer == 0:
            label = "L0 In"
        elif layer == config.num_layers - 1:
            label = f"L{layer} Out"
        else:
            label = f"L{layer}"

        surf = font.render(label, True, (220, 220, 220), (0, 0, 0))
        w, h = surf.get_size()
        # Pad to power-of-two
        tw = 1
        while tw < w:
            tw *= 2
        th = 1
        while th < h:
            th *= 2
        padded = pygame.Surface((tw, th), pygame.SRCALPHA)
        padded.fill((0, 0, 0, 0))
        padded.blit(surf, (0, 0))

        tex_data = pygame.image.tostring(padded, "RGBA", True)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)

        z = float(layer * 2 - config.num_layers)
        px = -half_w - 3.0
        py = -half_h - 1.5
        _label_textures.append((tex_id, tw, th, px, py, z))


def _render_labels():
    """Draw layer labels as textured quads in world space."""
    if not _label_textures:
        return

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    scale = 0.06  # world-space scale for label quads

    for tex_id, tw, th, px, py, z in _label_textures:
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glColor4f(1, 1, 1, 0.9)
        w = tw * scale
        h = th * scale
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex3f(px, py, z)
        glTexCoord2f(1, 0); glVertex3f(px + w, py, z)
        glTexCoord2f(1, 1); glVertex3f(px + w, py + h, z)
        glTexCoord2f(0, 1); glVertex3f(px, py + h, z)
        glEnd()

    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
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
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)


def render_3d_network(state, config):
    """Render the full network using cached vertex arrays.

    Only rebuilds geometry when state._3d_dirty is True.
    """
    if not HAS_OPENGL:
        return

    # Rebuild cache if needed
    if state._3d_dirty or _neuron_verts is None:
        rebuild_3d_cache(state, config)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, state.zoom)
    glRotatef(state.rotation_x, 1, 0, 0)
    glRotatef(state.rotation_y, 0, 1, 0)

    if _neuron_verts is not None and len(_neuron_verts) > 0:
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, _neuron_verts.ctypes)
        glColorPointer(3, GL_FLOAT, 0, _neuron_colors.ctypes)

        # Draw input neurons (small points)
        start, count = _input_range
        if count > 0:
            glPointSize(4)
            glDrawArrays(GL_POINTS, start, count)

        # Draw hidden neurons (medium points)
        start, count = _hidden_range
        if count > 0:
            glPointSize(8)
            glDrawArrays(GL_POINTS, start, count)

        # Draw output neurons (large points)
        start, count = _output_range
        if count > 0:
            glPointSize(12)
            glDrawArrays(GL_POINTS, start, count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    # Draw connections
    if _conn_verts is not None and len(_conn_verts) > 0:
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glLineWidth(1)

        glVertexPointer(3, GL_FLOAT, 0, _conn_verts.ctypes)
        glColorPointer(3, GL_FLOAT, 0, _conn_colors.ctypes)
        glDrawArrays(GL_LINES, 0, len(_conn_verts))

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    # Layer labels (world space)
    _build_label_textures(config)
    _render_labels()

    # HUD overlay (screen space)
    _build_hud_texture(state, config)
    _render_hud()


def render_3d_backprop(state, config, current_layer, current_pos):
    """Render backprop visualization: cached network + highlight overlay.

    Instead of re-rendering the entire network per cell, we render the cached
    network once and overlay the current cell's highlight on top.
    """
    if not HAS_OPENGL:
        return

    # Rebuild cache if needed (just once, not per cell)
    if state._3d_dirty or _neuron_verts is None:
        rebuild_3d_cache(state, config)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, state.zoom)
    glRotatef(state.rotation_x, 1, 0, 0)
    glRotatef(state.rotation_y, 0, 1, 0)

    # --- Draw cached network (dimmed) ---
    if _neuron_verts is not None and len(_neuron_verts) > 0:
        # Dim the base network slightly so highlights stand out
        dimmed_colors = _neuron_colors * 0.4
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, _neuron_verts.ctypes)
        glColorPointer(3, GL_FLOAT, 0, dimmed_colors.ctypes)
        glPointSize(5)
        glDrawArrays(GL_POINTS, 0, len(_neuron_verts))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    if _conn_verts is not None and len(_conn_verts) > 0:
        dimmed_conn = _conn_colors * 0.2
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glLineWidth(1)
        glVertexPointer(3, GL_FLOAT, 0, _conn_verts.ctypes)
        glColorPointer(3, GL_FLOAT, 0, dimmed_conn.ctypes)
        glDrawArrays(GL_LINES, 0, len(_conn_verts))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    # --- Overlay: highlight current cell and its connections ---
    nl = config.num_layers
    half_w = WIDTH / 2.0
    half_h = HEIGHT / 2.0
    x, y = current_pos
    z = float(current_layer * 2 - nl)

    try:
        # Yellow highlight point for current cell
        glPointSize(16)
        glBegin(GL_POINTS)
        glColor3f(1.0, 1.0, 0.0)
        glVertex3f(float(x) - half_w, float(y) - half_h, z)
        glEnd()

        # Highlight cells in adjacent layers
        cell = state.cells[x, y, current_layer]
        if cell is None:
            return

        glPointSize(10)
        glBegin(GL_POINTS)

        # Layer above (input to this cell) — colored by charge
        if current_layer > 0:
            reach = cell.reach if config.autonomous_network_genes else config.length_of_dendrite
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                        upper = state.cells[nx, ny, current_layer - 1]
                        if upper is not None:
                            ci = min(abs(upper.charge), 1.0)
                            glColor3f(0.2, 0.5 + 0.5 * ci, 1.0)  # blue-cyan
                            glVertex3f(float(nx) - half_w, float(ny) - half_h, float((current_layer - 1) * 2 - nl))

        # Layer below (this cell's error propagates to) — colored by error
        if current_layer < nl - 1 and current_layer + 1 < state.cells.shape[2]:
            reach_below = cell.reach if config.autonomous_network_genes else config.length_of_dendrite
            for dx in range(-reach_below, reach_below + 1):
                for dy in range(-reach_below, reach_below + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                        lower = state.cells[nx, ny, current_layer + 1]
                        if lower is not None:
                            ei = min(abs(lower.error), 1.0)
                            glColor3f(1.0, 0.3, ei)  # red-magenta
                            glVertex3f(float(nx) - half_w, float(ny) - half_h, float((current_layer + 1) * 2 - nl))
        glEnd()

        # Draw connections from current cell
        glLineWidth(3)
        glBegin(GL_LINES)

        # Connections to upper layer (dendrite inputs)
        if current_layer > 0:
            reach = cell.reach if config.autonomous_network_genes else config.length_of_dendrite
            wm = config.weight_matrix
            prev_z = float((current_layer - 1) * 2 - nl)
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                        if state.cells[nx, ny, current_layer - 1] is not None:
                            wi = (dx + reach) * wm + (dy + reach)
                            if wi < len(cell.weights):
                                w = cell.weights[wi]
                                aw = abs(w)
                                intensity = min(aw, 1.0)
                                if w > 0:
                                    glColor3f(0.0, intensity, 0.4)
                                else:
                                    glColor3f(intensity, 0.0, 0.4)
                                glVertex3f(float(x) - half_w, float(y) - half_h, z)
                                glVertex3f(float(nx) - half_w, float(ny) - half_h, prev_z)

        # Gradient arrow (downward)
        gi = min(abs(cell.gradient), 1.0)
        glColor3f(1.0, 0.5 * gi, 0.0)
        glVertex3f(float(x) - half_w, float(y) - half_h, z)
        glVertex3f(float(x) - half_w, float(y) - half_h, z - 0.8)

        glEnd()

    except Exception as e:
        print(f"Error in render_3d_backprop overlay: {e}")

    # HUD
    _build_hud_texture(state, config)
    _render_hud()


def render_3d_network_no_connections(state, config):
    """Simplified 3D render — neurons only, no connections."""
    if not HAS_OPENGL:
        return

    if state._3d_dirty or _neuron_verts is None:
        rebuild_3d_cache(state, config)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, state.zoom)
    glRotatef(state.rotation_x, 1, 0, 0)
    glRotatef(state.rotation_y, 0, 1, 0)

    if _neuron_verts is not None and len(_neuron_verts) > 0:
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, _neuron_verts.ctypes)
        glColorPointer(3, GL_FLOAT, 0, _neuron_colors.ctypes)
        glPointSize(5)
        glDrawArrays(GL_POINTS, 0, len(_neuron_verts))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    _render_labels()
    _render_hud()

    pygame.display.flip()
