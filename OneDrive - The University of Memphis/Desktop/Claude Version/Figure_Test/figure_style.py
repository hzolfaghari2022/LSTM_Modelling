"""Shared publication-style formatting for every generated figure.

This module changes presentation only.  It never slices the arrays used for
training, prediction, or metrics.  Full-record time series are drawn first and
then their displayed x-axis is limited to the first half of the record.
"""

from __future__ import annotations

import numpy as np
import matplotlib
from matplotlib import font_manager


REQUESTED_FONT_FAMILIES = ("Garamond", "Times New Roman")
_AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
FONT_FAMILY = next(
    (name for name in REQUESTED_FONT_FAMILIES if name in _AVAILABLE_FONTS),
    "Nimbus Roman" if "Nimbus Roman" in _AVAILABLE_FONTS else "DejaVu Serif",
)
TIME_VIEW_FRACTION = 0.50
LABEL_SIZE = 16
TICK_SIZE = 13
LEGEND_SIZE = 13
PANEL_TITLE_SIZE = 14
FIGURE_TITLE_SIZE = 18


def configure_plot_style():
    """Set large, consistent serif typography before figures are created."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY],
            "font.size": TICK_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.labelweight": "semibold",
            "figure.titlesize": FIGURE_TITLE_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.30,
        }
    )


def _is_full_record_time_axis(axis):
    """Return True for a time-history axis, excluding intentional zooms."""
    label = axis.get_xlabel().strip().lower()
    title = axis.get_title().strip().lower()
    contains_time = "time" in label or "time" in title
    intentional_zoom = "zoom" in label or "zoom" in title
    return axis.get_visible() and contains_time and not intentional_zoom


def _line_time_limits(axis):
    """Find the full finite time range from ordinary plotted signal lines."""
    candidates = []
    for line in axis.get_lines():
        try:
            values = np.asarray(line.get_xdata(orig=False), dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        values = values[np.isfinite(values)]
        # Ignore helper lines such as axhline/axvline when a real signal exists.
        if values.size >= 3 and np.ptp(values) > 0:
            candidates.append(values)
    if not candidates:
        return None
    values = np.concatenate(candidates)
    return float(np.min(values)), float(np.max(values))


def _show_first_half(axis):
    limits = _line_time_limits(axis)
    if limits is None:
        limits = tuple(float(value) for value in axis.get_xlim())
    start, end = limits
    if np.isfinite(start) and np.isfinite(end) and end > start:
        axis.set_xlim(start, start + TIME_VIEW_FRACTION * (end - start))


def _style_legend(axis, move_to_white_space):
    legend = axis.get_legend()
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontsize(LEGEND_SIZE)
    legend.set_frame_on(True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#b0b0b0")
    frame.set_alpha(0.94)
    if move_to_white_space:
        # Matplotlib's 'best' placement minimizes overlap with the plotted data.
        try:
            legend.set_loc("best")
        except AttributeError:  # Compatibility with older Matplotlib releases.
            legend._loc = 0
        legend.set_bbox_to_anchor(None)


def prepare_figure_for_export(figure):
    """Apply large fonts, half-duration views, and clean legend placement."""
    time_axes = [axis for axis in figure.axes if _is_full_record_time_axis(axis)]

    # Apply the view limit after all data have been plotted.  Shared x-axes
    # automatically inherit the same limit, while the underlying data stay full.
    for axis in time_axes:
        _show_first_half(axis)

    for axis in figure.axes:
        axis.xaxis.label.set_fontfamily(FONT_FAMILY)
        axis.yaxis.label.set_fontfamily(FONT_FAMILY)
        axis.xaxis.label.set_fontsize(LABEL_SIZE)
        axis.yaxis.label.set_fontsize(LABEL_SIZE)
        axis.title.set_fontfamily(FONT_FAMILY)
        axis.title.set_fontsize(max(axis.title.get_fontsize(), PANEL_TITLE_SIZE))
        axis.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
        axis.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
        axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontfamily(FONT_FAMILY)
        for annotation in axis.texts:
            annotation.set_fontfamily(FONT_FAMILY)
            if annotation is not axis.title:
                annotation.set_fontsize(max(annotation.get_fontsize(), 11))
        _style_legend(axis, axis in time_axes)

    for text in figure.texts:
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontsize(max(text.get_fontsize(), FIGURE_TITLE_SIZE))

    return figure
