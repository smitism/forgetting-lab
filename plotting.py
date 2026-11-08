"""
plotting.py - shared drawing code used by both the web app and the GIF maker.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE = "#2a6fdb"
ORANGE = "#e8833a"
CMAP = LinearSegmentedColormap.from_list("bo", [BLUE, "#f4f1ec", ORANGE])


def draw_task(ax, grid, points, labels, title):
    """Colour the plane by what the network answers, then draw the real dots on top."""
    ax.imshow(np.array(grid), origin="lower", extent=[-1, 1, -1, 1],
              cmap=CMAP, vmin=-1, vmax=1, alpha=0.75, interpolation="bilinear")
    pts = np.array(points)
    lab = np.array(labels)
    ax.scatter(pts[:, 0], pts[:, 1], c=np.where(lab > 0, ORANGE, BLUE),
               s=14, edgecolors="white", linewidths=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def draw_state(demo, frame, subtitle=""):
    """One picture: Task A view and Task B view of the network at one moment."""
    pts = demo["points"]
    log = demo["log"]
    acc_a = log["acc_a"][frame]
    acc_b = log["acc_b"][frame]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), dpi=100)
    g = demo["frames"][frame]
    draw_task(axes[0], g["a"], pts["x"], pts["task_a_labels"],
              f"Task A: left vs right   accuracy {acc_a:.0%}")
    draw_task(axes[1], g["b"], pts["x"], pts["task_b_labels"],
              f"Task B: bottom vs top   accuracy {acc_b:.0%}")
    fig.suptitle(subtitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
