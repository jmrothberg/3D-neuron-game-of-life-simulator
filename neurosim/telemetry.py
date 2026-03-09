"""Per-layer telemetry for debugging and monitoring."""
from dataclasses import dataclass
import numpy as np


@dataclass
class LayerTelemetry:
    layer_id: int
    num_cells: int = 0
    num_active: int = 0
    avg_charge: float = 0.0
    avg_error: float = 0.0
    avg_gradient: float = 0.0
    max_gradient: float = 0.0
    min_gradient: float = 0.0
    avg_weight_magnitude: float = 0.0
    avg_weights_per_cell: float = 0.0
    nan_count: int = 0


def compute_telemetry(state, config):
    """Compute per-layer telemetry. Call once per training cycle."""
    telemetry = []
    epsilon = config.epsilon

    for layer in range(1, config.num_layers - 1):
        t = LayerTelemetry(layer_id=layer)
        cells_in_layer = [
            state.cells[x, y, layer]
            for x, y in np.ndindex(state.cells.shape[:2])
            if state.cells[x, y, layer] is not None
        ]
        t.num_cells = len(cells_in_layer)

        active_cells = [c for c in cells_in_layer if len(c.weights) > 0 and np.any(np.abs(np.array(c.weights)) > epsilon)]
        t.num_active = len(active_cells)

        # Validate and count NaN issues
        nan_issues = []
        for cell in cells_in_layer:
            nan_issues.extend(cell.validate())
        t.nan_count = len(nan_issues)

        if active_cells:
            gradients = [c.gradient for c in active_cells]
            t.avg_gradient = float(np.mean(gradients))
            t.max_gradient = float(np.max(gradients))
            t.min_gradient = float(np.min(gradients))
            t.avg_error = float(np.mean([abs(c.error) for c in active_cells]))
            t.avg_charge = float(np.mean([abs(c.charge) for c in active_cells]))
            t.avg_weight_magnitude = float(np.mean([np.mean(np.abs(np.array(c.weights))) for c in active_cells]))
            t.avg_weights_per_cell = float(np.mean([len(c.weights) for c in active_cells]))

        telemetry.append(t)

    return telemetry


def format_telemetry(telemetry):
    """Format telemetry as a readable string."""
    lines = ["=== Layer Telemetry ==="]
    total_nan = 0
    for t in telemetry:
        lines.append(f"Layer {t.layer_id}: {t.num_active}/{t.num_cells} active")
        lines.append(f"  Charge: {t.avg_charge:.4f} | Error: {t.avg_error:.4e}")
        lines.append(f"  Gradient: avg={t.avg_gradient:.4e} max={t.max_gradient:.4e} min={t.min_gradient:.4e}")
        lines.append(f"  Weights: mag={t.avg_weight_magnitude:.4f} per_cell={t.avg_weights_per_cell:.1f}")
        if t.nan_count > 0:
            lines.append(f"  *** NaN issues: {t.nan_count} ***")
        total_nan += t.nan_count
    if total_nan > 0:
        lines.append(f"\n*** TOTAL NaN ISSUES: {total_nan} ***")
    else:
        lines.append("\nNo NaN issues detected.")
    return "\n".join(lines)
