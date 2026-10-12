"""Publication plotting defaults for journal-ready figures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicationPlotStyle:
    """House style for figures intended for journal submission."""

    axis_label_fontsize: int = 20
    axis_label_fontsize_min: int = 18
    axis_label_fontsize_max: int = 24
    labelpad: int = 12
    tick_fontsize: int = 12
    legend_fontsize: int = 12
    annotation_fontsize: int = 12
    grayscale: bool = True
    use_plot_titles: bool = False
    show_gridlines: bool = False
    despine_top: bool = True
    despine_right: bool = True

    @property
    def palette(self) -> tuple[str, ...]:
        """Return a grayscale-safe plotting palette."""

        return ("#111111", "#555555", "#888888", "#BBBBBB")
