#!/usr/bin/env python3
"""Figure 1: a list of numbers as a location, and cosine similarity as the basis for ranking.

Research question
-----------------
None. A teaching figure for the Method section ("Comparing Each Title With Its Neighbors").
Upper left: a list of two numbers, [3, 2], as a point in an x-y plane. Upper right: a list of
three numbers, [3, 2, 4], as a point in x-y-z space. Lower left: each point as an arrow from the
origin; the cosine of the angle between two arrows is their cosine similarity (one pair worked
through). Lower right: the other points ranked from most to least similar to the black point.

Data
----
Toy coordinates fixed below; no corpus data are used.

Reproducibility
---------------
    python scripts/figure1_distance_demo.py

Writes figures/figure1_distance_demo.png (300 dpi) and figures/figure1_distance_demo.pdf.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Arc

from presented_published.style import PublicationPlotStyle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("figures/figure1_distance_demo")
S = PublicationPlotStyle()
INK, MID, LIGHT = "#111111", "#888888", "#BBBBBB"

POINT_2D = (3, 2)
POINT_3D = (3, 2, 4)
FOCAL = np.array([5.0, 2.0])
# Point 2 lies on the black arrow at half its length: similarity 1, so length does not matter.
OTHERS = np.array([[2.0, 4.0], [2.5, 1.0], [5.5, 4.5], [0.8, 5.0], [4.0, 0.4], [1.2, 1.6]])
HIGHLIGHT = 0  # the pair whose angle is drawn in the lower-left panel


def apa(v: float) -> str:
    """Two decimals without a leading zero (APA style for values that cannot exceed 1)."""
    return f"{v:.2f}".replace("0.", ".", 1) if v < 1 else f"{v:.2f}"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.labelsize": S.axis_label_fontsize,
        "axes.labelpad": S.labelpad, "xtick.labelsize": S.tick_fontsize,
        "ytick.labelsize": S.tick_fontsize, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False})


def plane(ax: plt.Axes, lim: float) -> None:
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks(range(int(lim) + 1))
    ax.set_yticks(range(int(lim) + 1))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0, labelpad=S.labelpad + 4)


def main() -> None:
    style()
    fig = plt.figure(figsize=(12, 11))
    a = fig.add_subplot(2, 2, 1)
    b = fig.add_subplot(2, 2, 2, projection="3d")
    c = fig.add_subplot(2, 2, 3)
    d = fig.add_subplot(2, 2, 4)

    # Upper left: two numbers as a point in a plane.
    x, y = POINT_2D
    plane(a, 5)
    a.plot([x, x], [0, y], ls="--", lw=1, color=MID)
    a.plot([0, x], [y, y], ls="--", lw=1, color=MID)
    a.scatter([x], [y], s=90, color=INK, zorder=3)
    a.annotate(f"[{x}, {y}]", (x, y), xytext=(10, 8), textcoords="offset points",
               fontsize=S.annotation_fontsize + 4)

    # Upper right: three numbers as a point in space.
    x3, y3, z3 = POINT_3D
    lim = 5
    b.set_xlim(0, lim)
    b.set_ylim(0, lim)
    b.set_zlim(0, lim)
    b.plot([x3, x3], [y3, y3], [0, z3], ls="--", lw=1, color=MID)
    b.plot([x3, x3], [0, y3], [0, 0], ls="--", lw=1, color=MID)
    b.plot([0, x3], [y3, y3], [0, 0], ls="--", lw=1, color=MID)
    b.scatter([x3], [y3], [z3], s=90, color=INK, depthshade=False)
    b.scatter([x3], [y3], [0], s=25, facecolor="white", edgecolor=MID, depthshade=False)
    b.text(x3 + 0.25, y3, z3 + 0.3, f"[{x3}, {y3}, {z3}]", fontsize=S.annotation_fontsize + 4)
    b.set_xlabel("x", labelpad=6)
    b.set_ylabel("y", labelpad=6)
    b.set_zlabel("z", labelpad=4)
    b.tick_params(labelsize=S.tick_fontsize)
    for axis in (b.xaxis, b.yaxis, b.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(LIGHT)
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    b.set_proj_type("ortho")   # no perspective, so the point reads at its true height
    # Label each dashed leg with its number; the z ticks sit on a far corner of the box.
    lf = S.annotation_fontsize + 2
    b.text(x3 / 2, y3 - 0.2, 0, f"x = {x3}", fontsize=lf, ha="center", va="top")
    b.text(x3 + 0.6, y3 / 2, 0, f"y = {y3}", fontsize=lf, ha="left", va="center")
    b.text(x3 + 0.15, y3, z3 / 2, f"z = {z3}", fontsize=lf, ha="left", va="center")
    b.view_init(elev=20, azim=-60)

    # Lower left: each point as an arrow from the origin; similarity is the cosine of the angle.
    plane(c, 6)
    fs = S.annotation_fontsize + 3

    def arrow(p: np.ndarray, color: str, lw: float) -> None:
        c.annotate("", xy=p, xytext=(0, 0),
                   arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw,
                               "shrinkA": 0, "shrinkB": 5, "mutation_scale": 14})

    for k, q in enumerate(OTHERS):
        arrow(q, INK if k == HIGHLIGHT else LIGHT, 1.6 if k == HIGHLIGHT else 1.0)
    arrow(FOCAL, INK, 1.6)
    h = OTHERS[HIGHLIGHT]
    ang = np.degrees(np.arctan2([FOCAL[1], h[1]], [FOCAL[0], h[0]]))
    c.add_patch(Arc((0, 0), 2.4, 2.4, theta1=ang.min(), theta2=ang.max(), color=INK, lw=1.2))
    mid = np.radians(ang.mean())
    c.text(1.45 * np.cos(mid), 1.45 * np.sin(mid), r"$\theta$", fontsize=fs + 2,
           ha="center", va="center")
    cos = OTHERS @ FOCAL / (np.linalg.norm(OTHERS, axis=1) * np.linalg.norm(FOCAL))
    c.text(0.25, 5.85, rf"cosine similarity = cos $\theta$ = {apa(cos[HIGHLIGHT])}",
           ha="left", va="top", fontsize=fs)
    c.scatter(OTHERS[:, 0], OTHERS[:, 1], s=80, facecolor="white", edgecolor=INK, lw=1.3, zorder=3)
    for k, q in enumerate(OTHERS, start=1):
        off = (4, -20) if k == 2 else (7, 5)   # point 2 sits on the black arrow; label below it
        c.annotate(str(k), q, xytext=off, textcoords="offset points", fontsize=fs)
    c.scatter([FOCAL[0]], [FOCAL[1]], s=90, color=INK, zorder=4)

    # Lower right: the other points ranked from most to least similar.
    order = np.argsort(-cos)
    rows = np.arange(len(order))
    d.barh(rows, cos[order], color=MID, height=0.6)
    d.set_yticks(rows, [f"Point {k + 1}" for k in order])
    d.invert_yaxis()
    for r, k in zip(rows, order, strict=True):
        d.text(cos[k] + 0.015, r, apa(cos[k]), va="center", fontsize=S.tick_fontsize)
    d.set_xlim(0, 1.1)
    d.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".50", ".75", "1"])
    d.set_xlabel("Cosine similarity with the black point")
    d.set_ylabel("Most to least similar")

    fig.tight_layout(w_pad=3, h_pad=3)
    meta = {"Title": "figure1_distance_demo",
            "Description": f"Generated {date.today().isoformat()} by "
                           "scripts/figure1_distance_demo.py; toy coordinates, no corpus data."}
    fig.savefig(OUT.with_suffix(".png"), dpi=300, metadata=meta)
    fig.savefig(OUT.with_suffix(".pdf"), metadata={"Title": meta["Title"],
                                                   "Subject": meta["Description"]})
    print(f"wrote {OUT}.png and .pdf; ranked similarities:",
          ", ".join(f"{k + 1}: {cos[k]:.3f}" for k in order))


if __name__ == "__main__":
    main()
