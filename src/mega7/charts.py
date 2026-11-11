"""Charts for the results folder (matplotlib, PNG).

Colours follow the entity, never its rank: each stock keeps one colour in every chart (the first
seven slots of a palette checked for colour-vision deficiency), the equal-weight Mega 7 portfolio
takes the eighth, and the three index funds are drawn in neutral inks. Gridlines are hairlines, and
two measures with different scales are drawn as two panels, never on two y-axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

COLOURS = {
    "AAPL": "#2a78d6",
    "MSFT": "#eb6834",
    "GOOGL": "#1baf7a",
    "AMZN": "#eda100",
    "NVDA": "#e87ba4",
    "META": "#008300",
    "TSLA": "#6250d6",
    "Mega 7": "#e34948",
    "SPY": "#0b0b0b",
    "QQQ": "#52514e",
    "RSP": "#898781",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# Diverging blue <-> red with a neutral grey midpoint, for signed values such as factor betas.
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#184f95", "#6da7ec", "#f0efec", "#ec8b8a", "#b52f2f"])
SHADE = "#f0efec"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "font.family": "sans-serif",
        }
    )


def colour(name: str) -> str:
    return COLOURS.get(name, COLOURS["Mega 7"] if name.startswith("Mega 7") else MUTED)


def shade_windows(ax, windows: list[tuple[pd.Timestamp, pd.Timestamp, str]], label: bool = True) -> None:
    """Grey bands for episodes, labelled along the top edge (alternating between two rows so that
    neighbouring labels do not collide)."""
    for k, (start, end, name) in enumerate(windows):
        ax.axvspan(start, end, color=SHADE, zorder=0, lw=0)
        if label:
            ax.text(start, 1.01 + 0.07 * (k % 2), name, transform=ax.get_xaxis_transform(), fontsize=6.5,
                    color=MUTED, ha="left", va="bottom", rotation=0)


def wealth_chart(returns: pd.DataFrame, title: str, log: bool = True):
    """Growth of $1 for each column, on a log scale, with each line labelled at its end."""
    style()
    fig, ax = plt.subplots(figsize=(9, 5))
    wealth = (1 + returns).cumprod()
    for col in wealth.columns:
        ax.plot(wealth.index, wealth[col], color=colour(col), lw=1.8 if col.startswith("Mega 7") else 1.3, label=col)
    if log:
        ax.set_yscale("log")
    ends = wealth.iloc[-1].sort_values()
    placed: list[float] = []
    for col, value in ends.items():
        y = np.log10(value) if log else value
        while any(abs(y - p) < 0.07 for p in placed):
            y += 0.07
        placed.append(y)
        ax.annotate(f"{col} {value:,.1f}x", xy=(wealth.index[-1], 10**y if log else y), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK_2, va="center")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1 (log scale)" if log else "Growth of $1")
    ax.legend(ncol=4, loc="upper left")
    ax.margins(x=0.0)
    fig.subplots_adjust(right=0.82)
    return fig


def drawdown_chart(drawdowns: pd.DataFrame, title: str):
    style()
    fig, ax = plt.subplots(figsize=(9, 4))
    for col in drawdowns.columns:
        ax.plot(drawdowns.index, -drawdowns[col] * 100, color=colour(col), lw=1.6 if col.startswith("Mega 7") else 1.2, label=col)
    ax.set_title(title)
    ax.set_ylabel("Drawdown from peak, %")
    ax.legend(ncol=4, loc="lower left")
    ax.margins(x=0.0)
    return fig


def heatmap(matrix: pd.DataFrame, title: str, ax=None, vmin: float = 0.0, vmax: float = 1.0, fmt: str = "{:.2f}",
            diverging: bool = False):
    """A heatmap with the value printed in every cell; the CSV is the table view.

    One hue (blue, light to dark) for magnitudes; with `diverging=True`, blue for negative and red for
    positive values around a grey zero, with `vmin = -vmax`.
    """
    style()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.grid(False)
    square = matrix.shape[0] == matrix.shape[1]
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=DIVERGING if diverging else BLUES, vmin=vmin, vmax=vmax,
                   aspect="equal" if square else "auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=0)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.tick_params(length=0, colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    values = matrix.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                continue
            strong = abs(v) / vmax > 0.6 if diverging else (v - vmin) / (vmax - vmin) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if strong else INK)
    ax.set_title(title)
    if own:
        return fig, im
    return ax, im


def small_multiples(frame: pd.DataFrame, title: str, ylabel: str, ncols: int = 4, reference: pd.Series | None = None,
                    reference_label: str = "", ylim: tuple[float, float] | None = None, windows=None):
    """One small panel per column, sharing both axes; an optional grey reference line in every panel."""
    style()
    n = frame.shape[1]
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.4 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, frame.columns):
        if windows:
            shade_windows(ax, windows, label=False)
        if reference is not None:
            ax.plot(reference.index, reference, color=MUTED, lw=1.0, label=reference_label)
        ax.plot(frame.index, frame[col], color=colour(col), lw=1.4, label=col)
        ax.set_title(col, fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if ylim:
            ax.set_ylim(*ylim)
    for ax in axes[n:]:
        ax.set_visible(False)
    if reference is not None:
        axes[0].legend(loc="lower left")
    fig.supylabel(ylabel, fontsize=9, color=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def bar_chart(values: pd.Series, title: str, xlabel: str, fmt: str = "{:.0%}", colours: list[str] | None = None,
              reference: float | None = None, reference_label: str = ""):
    """Horizontal bars, one per entity in its own colour, each labelled with its value."""
    style()
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(values) + 1.2))
    ax.grid(axis="y", visible=False)
    cols = colours or [colour(i) for i in values.index]
    ax.barh(range(len(values)), values.to_numpy(), color=cols, height=0.62)
    ax.set_yticks(range(len(values)), values.index)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, colors=INK_2)
    span = np.nanmax(np.abs(values.to_numpy()))
    for i, v in enumerate(values.to_numpy()):
        ax.text(v + (0.01 * span if v >= 0 else -0.01 * span), i, fmt.format(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK_2)
    if reference is not None:
        ax.axvline(reference, color=MUTED, lw=1.0)
        ax.text(reference, -0.7, reference_label, fontsize=7.5, color=MUTED, ha="center")
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.margins(x=0.12)
    return fig
