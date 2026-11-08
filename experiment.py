"""
experiment.py - make a small neural network learn Task A, then Task B,
and measure how much of Task A it forgets. Then test a fix: mixing a few
stored Task A examples into the Task B training ("replay").

Run:  python experiment.py
Output: results.json  (used by the web app and the GIF maker)
"""
import json
import random
import time

import numpy as np

from engine import MLP

# ---------------- settings you can play with ----------------
SEEDS = [0, 1, 2, 3, 4]          # each seed = a different random start; we average over them
REPLAY_SIZES = [0, 5, 20, 50]    # how many old Task A examples we keep while learning Task B
STEPS_PER_TASK = 150
LEARNING_RATE = 0.1
BATCH = 16                       # new examples per step
N_POINTS = 200                   # points per task
GRID = 20                        # picture resolution for the animation
FRAME_EVERY = 10                 # save a picture every 10 steps
SIZES = [3, 8, 8, 1]             # 3 inputs: x, y, and a "task flag" (-1 = Task A, +1 = Task B)


# ---------------- the two tasks ----------------
# Both tasks use dots spread over the SAME square, so the network must reuse the
# same weights for both. Task A: left = class -1, right = class +1.
# Task B: bottom = class -1, top = class +1.
# The third input tells the network which task it is being asked about. Without it,
# the same dot would have two different correct answers and no network could get
# both tasks right.
def make_tasks(rng):
    pts = rng.uniform(-1, 1, size=(N_POINTS, 2))
    # leave a small gap near the dividing line so the tasks are cleanly learnable
    pts = pts[(np.abs(pts[:, 0]) > 0.1) & (np.abs(pts[:, 1]) > 0.1)]
    flag_a = np.full((len(pts), 1), -1.0)
    flag_b = np.full((len(pts), 1), 1.0)
    task_a = (np.hstack([pts, flag_a]), np.where(pts[:, 0] > 0, 1.0, -1.0))
    task_b = (np.hstack([pts, flag_b]), np.where(pts[:, 1] > 0, 1.0, -1.0))
    return task_a, task_b


# ---------------- numpy copy of the network, for fast accuracy / pictures ----------------
def np_forward(weights, X):
    h = X
    for W, b in weights:
        h = np.tanh(h @ np.array(W).T + np.array(b))
    return h[:, 0]


def accuracy(model, task):
    X, y = task
    pred = np_forward(model.export(), X)
    return float(np.mean(np.sign(pred) == y))


def boundary_grid(model):
    """The network's answer at every spot on the plane, once per task flag."""
    xs = np.linspace(-1, 1, GRID)
    gx, gy = np.meshgrid(xs, xs)
    base = np.stack([gx.ravel(), gy.ravel()], axis=1)
    weights = model.export()
    grids = {}
    for name, flag in (("a", -1.0), ("b", 1.0)):
        pts = np.hstack([base, np.full((len(base), 1), flag)])
        out = np_forward(weights, pts).reshape(GRID, GRID)
        grids[name] = np.round(out, 3).tolist()      # row = y, column = x
    return grids


# ---------------- training ----------------
def loss_on(model, X, y):
    """Mean squared error: how far the output is from the correct +1 / -1."""
    total = 0
    for xi, yi in zip(X, y):
        out = model([float(xi[0]), float(xi[1]), float(xi[2])])
        total = total + (out - float(yi)) ** 2
    return total * (1.0 / len(X))


def train_step(model, batch_idx, task, replay, lr):
    X, y = task
    bx, by = X[batch_idx], y[batch_idx]
    loss = loss_on(model, bx, by)
    if replay is not None:                       # also practise a few old examples
        rx, ry = replay
        loss = loss + loss_on(model, rx, ry)
    model.zero_grad()                            # reset gradients (don't forget this!)
    loss.backward()                              # work out the gradients
    for p in model.parameters():                 # nudge every weight against its gradient
        p.data -= lr * p.grad


def run(seed, replay_size, record_frames=False):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    task_a, task_b = make_tasks(rng)
    model = MLP(SIZES)
    log = {"step": [], "acc_a": [], "acc_b": []}
    frames = []

    def snapshot(step):
        log["step"].append(step)
        log["acc_a"].append(accuracy(model, task_a))
        log["acc_b"].append(accuracy(model, task_b))
        if record_frames:
            frames.append(boundary_grid(model))

    # ---- phase 1: learn Task A ----
    snapshot(0)
    for step in range(1, STEPS_PER_TASK + 1):
        idx = rng.choice(len(task_a[0]), BATCH, replace=False)
        train_step(model, idx, task_a, None, LEARNING_RATE)
        if step % FRAME_EVERY == 0:
            snapshot(step)
    acc_a_after_a = accuracy(model, task_a)

    # ---- keep a few Task A examples to practise later ----
    replay = None
    if replay_size > 0:
        keep = rng.choice(len(task_a[0]), replay_size, replace=False)
        replay = (task_a[0][keep], task_a[1][keep])

    # ---- phase 2: learn Task B ----
    for step in range(1, STEPS_PER_TASK + 1):
        idx = rng.choice(len(task_b[0]), BATCH, replace=False)
        train_step(model, idx, task_b, replay, LEARNING_RATE)
        if step % FRAME_EVERY == 0:
            snapshot(STEPS_PER_TASK + step)

    result = {
        "acc_a_after_a": acc_a_after_a,
        "acc_a_after_b": accuracy(model, task_a),
        "acc_b_after_b": accuracy(model, task_b),
        "log": log,
    }
    if record_frames:
        result["frames"] = frames
        result["points"] = {
            "x": task_a[0][:, :2].tolist(),
            "task_a_labels": task_a[1].tolist(),
            "task_b_labels": task_b[1].tolist(),
        }
    return result


def main():
    t0 = time.time()
    output = {
        "settings": {
            "steps_per_task": STEPS_PER_TASK,
            "learning_rate": LEARNING_RATE,
            "seeds": SEEDS,
            "replay_sizes": REPLAY_SIZES,
            "grid": GRID,
            "frame_every": FRAME_EVERY,
        },
        "by_replay": {},
    }
    for rs in REPLAY_SIZES:
        runs = []
        for seed in SEEDS:
            r = run(seed, rs, record_frames=(seed == SEEDS[0]))
            runs.append(r)
            print(f"replay={rs:>2} seed={seed}  "
                  f"A after A={r['acc_a_after_a']:.2f}  "
                  f"A after B={r['acc_a_after_b']:.2f}  "
                  f"B after B={r['acc_b_after_b']:.2f}", flush=True)
        a_after_b = [r["acc_a_after_b"] for r in runs]
        b_after_b = [r["acc_b_after_b"] for r in runs]
        a_after_a = [r["acc_a_after_a"] for r in runs]
        output["by_replay"][str(rs)] = {
            "acc_a_after_a_mean": float(np.mean(a_after_a)),
            "acc_a_after_b_mean": float(np.mean(a_after_b)),
            "acc_a_after_b_std": float(np.std(a_after_b)),
            "acc_b_after_b_mean": float(np.mean(b_after_b)),
            "acc_b_after_b_std": float(np.std(b_after_b)),
            "demo": runs[0],           # seed 0 run, with pictures, used by the app
        }
    with open("results.json", "w") as f:
        json.dump(output, f)
    print(f"\nDone in {time.time() - t0:.0f}s -> results.json")
    print("\nSummary (average over seeds):")
    print("replay | A after A | A after B | B after B")
    for rs in REPLAY_SIZES:
        s = output["by_replay"][str(rs)]
        print(f"{rs:>6} | {s['acc_a_after_a_mean']:.2f}      | "
              f"{s['acc_a_after_b_mean']:.2f}±{s['acc_a_after_b_std']:.2f} | "
              f"{s['acc_b_after_b_mean']:.2f}±{s['acc_b_after_b_std']:.2f}")


if __name__ == "__main__":
    main()
