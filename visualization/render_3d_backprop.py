def draw_connection(start, end, weight, weight_delta, show_weight_overlay=False):
    """
    Draws a connection between two nodes (cells) with visual feedback for weight changes.

    Parameters:
    - start: (x, y, z) tuple representing the starting node.
    - end: (x, y, z) tuple representing the ending node.
    - weight: float, the current weight value of the connection.
    - weight_delta: float, the change in the weight from the last update.
    - show_weight_overlay: bool, when True the function overlays the numeric weight 
      value at the midpoint of the connection.

    The connection's color will be interpolated between a default color and a highlight
    color (red) depending on the magnitude of the weight update. A small change results in
    only a slight color shift, while a larger change makes the connection line appear more red.
    """
    # Default color for static connections (light grey)
    default_color = (200, 200, 200)
    # Highlight color used to indicate a weight change (red)
    highlight_color = (255, 0, 0)
    
    # Calculate the magnitude of the weight change
    update_magnitude = abs(weight_delta)
    
    # Define a threshold for what we consider a "minor" update
    minor_threshold = 0.01

    # Determine the color based on the weight change magnitude.
    # If the change is significant, interpolate the color towards red.
    if update_magnitude > minor_threshold:
        # Clamp the factor so that even drastic changes don't exceed our interpolation scale
        factor = min(update_magnitude, 1.0)
        color = (
            int(default_color[0] * (1 - factor) + highlight_color[0] * factor),
            int(default_color[1] * (1 - factor) + highlight_color[1] * factor),
            int(default_color[2] * (1 - factor) + highlight_color[2] * factor)
        )
    else:
        color = default_color

    # Draw the connection line.
    # (Assume draw_line is an existing function in your rendering system)
    draw_line(start, end, color=color)

    # Optionally, overlay the current weight value on the connection.
    if show_weight_overlay:
        midpoint = (
            (start[0] + end[0]) / 2, 
            (start[1] + end[1]) / 2, 
            (start[2] + end[2]) / 2
        )
        # Draw the weight value (formatted to 2 decimal places) at the connection's midpoint.
        # (Assume draw_text is an existing function in your rendering system)
        draw_text(midpoint, f"{weight:.2f}", color=(255, 255, 255)) 