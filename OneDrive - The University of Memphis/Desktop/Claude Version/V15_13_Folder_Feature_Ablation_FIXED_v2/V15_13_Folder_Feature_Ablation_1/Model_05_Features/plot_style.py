"""Shared publication style for every Matplotlib figure in this project."""

import matplotlib


PUBLICATION_SERIF_FONTS = [
    "Times New Roman",
    "Garamond",
    "EB Garamond",
    "Times",
    "Liberation Serif",
    "DejaVu Serif",
]


def apply_publication_style():
    """Prefer Times New Roman and use close serif fallbacks when unavailable."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": PUBLICATION_SERIF_FONTS,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

