"""
app.py - interactive web page for the Forgetting Lab.

Run locally:   streamlit run app.py
It only plays back results.json (made by experiment.py), so it loads instantly.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from plotting import BLUE, ORANGE, draw_state

st.set_page_config(page_title="Why do neural networks forget?", layout="centered")


@st.cache_data
def load():
    return json.loads((Path(__file__).parent / "results.json").read_text())


data = load()
cfg = data["settings"]
by_replay = data["by_replay"]
sizes = [int(k) for k in by_replay]
steps_per_task = cfg["steps_per_task"]

st.title("Why do neural networks forget?")
st.write(
    "A small neural network, built from scratch, first learns **Task A** "
    "(is the dot on the left or right?), then learns **Task B** (is it at the bottom or top?). "
    "Watch what happens to Task A. Then see how keeping a few old examples changes the story."
)

# ---------------- controls ----------------
replay = st.select_slider(
    "Old Task A examples kept while learning Task B",
    options=sizes,
    value=0,
    help="0 means the network gets no reminders of Task A.",
)
demo = by_replay[str(replay)]["demo"]
steps = demo["log"]["step"]
step = st.select_slider("Training step", options=steps, value=steps[-1])
frame = steps.index(step)

phase = "learning Task A" if step <= steps_per_task else "learning Task B"
fig = draw_state(demo, frame, f"Step {step}: {phase}")
st.pyplot(fig)
plt.close(fig)

c1, c2 = st.columns(2)
c1.metric("Task A accuracy", f"{demo['log']['acc_a'][frame]:.0%}")
c2.metric("Task B accuracy", f"{demo['log']['acc_b'][frame]:.0%}")
st.caption(
    "Background colour is the network's answer at each spot (blue = class -1, orange = class +1). "
    "Dots are the training examples, coloured by their correct class."
)

# ---------------- accuracy over time ----------------
st.subheader("Accuracy over training")
fig2, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
ax.plot(steps, np.array(demo["log"]["acc_a"]) * 100, color=BLUE, lw=2.2, label="Task A")
ax.plot(steps, np.array(demo["log"]["acc_b"]) * 100, color=ORANGE, lw=2.2, label="Task B")
ax.axvline(steps_per_task, color="grey", ls="--", lw=1)
ax.text(steps_per_task + 3, 8, "Task B starts", color="grey", fontsize=9)
ax.axvline(step, color="black", lw=1, alpha=0.4)
ax.set_xlabel("training step")
ax.set_ylabel("accuracy (%)")
ax.set_ylim(0, 105)
ax.legend(loc="lower left", frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ---------------- the comparison ----------------
st.subheader("How much of Task A survives?")
st.write(
    f"Accuracy on Task A *after* learning Task B, averaged over {len(cfg['seeds'])} different "
    "random starts. The thin line shows how much the runs varied."
)
means = [by_replay[str(s)]["acc_a_after_b_mean"] * 100 for s in sizes]
stds = [by_replay[str(s)]["acc_a_after_b_std"] * 100 for s in sizes]
fig3, ax3 = plt.subplots(figsize=(8, 3.2), dpi=100)
bars = ax3.bar([str(s) for s in sizes], means, yerr=stds, color=BLUE, capsize=4, width=0.55)
for b, m in zip(bars, means):
    ax3.text(b.get_x() + b.get_width() / 2, 4, f"{m:.0f}%", ha="center", color="white",
             fontweight="bold")
ax3.set_xlabel("old examples kept")
ax3.set_ylabel("Task A accuracy (%)")
ax3.set_ylim(0, 105)
for side in ("top", "right"):
    ax3.spines[side].set_visible(False)
fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

st.markdown(
    """
**What is going on?** Learning Task B means nudging the network's weights to get Task B right.
Nothing in that nudging knows about Task A, so the weights Task A relied on get pulled away.
Mixing in a few old Task A examples adds a reminder to every nudge, so the weights stay
good for both tasks.

Built with a from-scratch autograd engine (no PyTorch). Code: see the repository README.
"""
)
