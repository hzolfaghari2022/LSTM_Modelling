"""Generate the architecture figure for the implemented V15 one-step model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from plot_style import apply_publication_style


INK = "#27333d"
MUTED = "#687985"
LINE = "#7d8a93"
BLUE = "#3d7cc4"
BLUE_EDGE = "#24558f"
ORANGE = "#ed9858"
ORANGE_EDGE = "#b86129"
GOLD = "#f0c767"
GOLD_EDGE = "#b17a00"
PURPLE = "#7651a8"


def _arrow(axis, start, end, width=0.9, alpha=0.75, color=LINE, zorder=1):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=color,
            alpha=alpha,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def _node(axis, x, y, radius, face, edge):
    axis.add_patch(
        Circle(
            (x, y),
            radius,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.25,
            zorder=4,
        )
    )


def _loop(axis, x, y, radius):
    axis.add_patch(
        FancyArrowPatch(
            (x - 0.43 * radius, y + 0.72 * radius),
            (x + 0.43 * radius, y + 0.72 * radius),
            connectionstyle="arc3,rad=-1.6",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.85,
            color=INK,
            zorder=5,
        )
    )


def _box(axis, x, y, width, height, edge, fill="white", dashed=False):
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.5,
            linestyle=(0, (4, 2)) if dashed else "-",
            zorder=0,
        )
    )


def _connect_layers(axis, left_x, right_x, y_values, left_r, right_r):
    for left_y in y_values:
        for right_y in y_values:
            _arrow(
                axis,
                (left_x + left_r, left_y),
                (right_x - right_r, right_y),
                width=0.55,
                alpha=0.38,
            )


def generate_architecture_figure(figures_folder, feature_count=13):
    """Save a truthful node-style figure of the primary hybrid predictor."""
    apply_publication_style()
    figures_folder = Path(figures_folder)
    figures_folder.mkdir(parents=True, exist_ok=True)
    output = figures_folder / "19_one_input_hybrid_lstm_architecture.png"

    figure, axis = plt.subplots(figsize=(15.0, 8.1))
    figure.subplots_adjust(left=0.018, right=0.982, bottom=0.055, top=0.88)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.suptitle(
        "Implemented One-Input Hybrid LSTM Architecture",
        y=0.965,
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.5,
        0.915,
        "Causal one-step measured-feedback prediction for the voice-coil actuator",
        ha="center",
        fontsize=12,
        color=MUTED,
    )

    # Single externally applied input.
    _box(axis, 0.018, 0.315, 0.105, 0.39, BLUE_EDGE, "#eef5fb", dashed=True)
    axis.text(0.0705, 0.75, "INPUT", ha="center", fontsize=13, fontweight="bold", color=INK)
    _node(axis, 0.0705, 0.545, 0.030, BLUE, BLUE_EDGE)
    axis.text(0.0705, 0.475, r"Coil current $I_k$", ha="center", fontsize=11, fontweight="bold", color=INK)
    axis.text(0.0705, 0.365, "1 physical input", ha="center", fontsize=10.5, fontweight="bold", color=BLUE_EDGE)

    # Causal feature sequence supplied to the recurrent path.
    _box(axis, 0.152, 0.285, 0.135, 0.45, "#8796a1", "#f7f9fa")
    axis.text(0.2195, 0.78, "CAUSAL SEQUENCE", ha="center", fontsize=12, fontweight="bold", color=INK)
    axis.text(
        0.2195,
        0.615,
        rf"$64\times{int(feature_count)}$",
        ha="center",
        fontsize=17,
        fontweight="bold",
        color=BLUE_EDGE,
    )
    axis.text(
        0.2195,
        0.485,
        "Current-derived, load/time,\nand past-response features\navailable only through $k-1$",
        ha="center",
        va="center",
        fontsize=9.7,
        linespacing=1.35,
        color=INK,
    )
    axis.text(
        0.2195,
        0.325,
        f"{int(feature_count)} causal features; 1 physical input",
        ha="center",
        fontsize=9.1,
        fontweight="bold",
        color=PURPLE,
    )
    _arrow(axis, (0.123, 0.51), (0.152, 0.51), width=1.2)

    # Two stacked LSTM layers and the dense head.
    layers = [
        (0.325, 0.13, "LSTM LAYER 1", "48 hidden units", ORANGE_EDGE, ORANGE, 0.390, 0.025),
        (0.495, 0.13, "LSTM LAYER 2", "48 hidden units", ORANGE_EDGE, ORANGE, 0.560, 0.025),
        (0.665, 0.105, "DENSE LAYER", "32 neurons + tanh", GOLD_EDGE, GOLD, 0.7175, 0.022),
    ]
    node_y = [0.69, 0.56, 0.43, 0.25]
    for x0, width, title, subtitle, edge, face, center, radius in layers:
        _box(axis, x0, 0.18, width, 0.60, edge)
        axis.text(x0 + width / 2, 0.825, title, ha="center", fontsize=12.5, fontweight="bold", color=INK)
        axis.text(x0 + width / 2, 0.145, subtitle, ha="center", fontsize=10.2, color=MUTED)
        for y in node_y:
            _node(axis, center, y, radius, face, edge)
            if "LSTM" in title:
                _loop(axis, center, y, radius)
        axis.text(center, 0.34, r"$\vdots$", ha="center", va="center", fontsize=21, color=INK)

    _connect_layers(axis, 0.390, 0.560, node_y, 0.025, 0.025)
    _connect_layers(axis, 0.560, 0.7175, node_y, 0.025, 0.022)
    for y in node_y:
        _arrow(axis, (0.287, 0.51), (0.365, y), width=0.65, alpha=0.65)

    # One learned output from the LSTM path.
    _box(axis, 0.795, 0.54, 0.12, 0.18, BLUE_EDGE, "#f0f6fc", dashed=True)
    axis.text(0.855, 0.755, "LSTM OUTPUT", ha="center", fontsize=11.5, fontweight="bold", color=INK)
    _node(axis, 0.835, 0.64, 0.022, BLUE, BLUE_EDGE)
    axis.text(0.872, 0.64, r"$\Delta x_k$", va="center", fontsize=12.5, color=INK)
    axis.text(0.855, 0.565, "1 learned residual", ha="center", fontsize=9.5, color=MUTED)
    for y in node_y:
        _arrow(axis, (0.7395, y), (0.813, 0.64), width=0.65, alpha=0.48)

    # The implemented force path is a separate causal rule.
    _box(axis, 0.795, 0.255, 0.12, 0.17, PURPLE, "#f6f1fa")
    axis.text(0.855, 0.465, "CAUSAL FORCE RULE", ha="center", fontsize=10.8, fontweight="bold", color=INK)
    axis.text(
        0.855,
        0.34,
        "$I_k$ + latest reliable\npast $F/I$ gain",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.287, 0.31),
            (0.795, 0.34),
            connectionstyle="arc3,rad=0.24",
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color=PURPLE,
            alpha=0.60,
            zorder=1,
        )
    )

    # Two final physical responses from the complete hybrid system.
    _box(axis, 0.938, 0.285, 0.052, 0.43, BLUE_EDGE, "#eef5fb", dashed=True)
    axis.text(0.964, 0.755, "SYSTEM\nRESPONSES", ha="center", fontsize=10.5, fontweight="bold", color=INK)
    _node(axis, 0.964, 0.58, 0.017, BLUE, BLUE_EDGE)
    _node(axis, 0.964, 0.42, 0.017, BLUE, BLUE_EDGE)
    axis.text(0.964, 0.535, r"$\hat{x}_k$", ha="center", fontsize=12, color=INK)
    axis.text(0.964, 0.375, r"$\hat{F}_k$", ha="center", fontsize=12, color=INK)
    axis.text(0.964, 0.315, "2 final\nresponses", ha="center", fontsize=8.8, color=MUTED)
    _arrow(axis, (0.915, 0.64), (0.947, 0.58), width=1.0, color=BLUE_EDGE)
    _arrow(axis, (0.915, 0.34), (0.947, 0.42), width=1.0, color=PURPLE)

    axis.text(
        0.49,
        0.105,
        r"LSTM path: $64\times13 \rightarrow 48 \rightarrow 48 \rightarrow 32 \rightarrow 1$ displacement residual",
        ha="center",
        fontsize=10.8,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.49,
        0.065,
        "Complete hybrid predictor: 1 applied current input → displacement and Lorentz-force responses",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    print("  wrote", output.name, flush=True)
    return output


if __name__ == "__main__":
    generate_architecture_figure(Path(__file__).resolve().parent / "FiguresResults")
