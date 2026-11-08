"""
make_gif.py - turn results.json into an animated GIF for LinkedIn / the README.

Run:  python make_gif.py            (uses replay=0, the 'forgetting' story)
      python make_gif.py 20         (uses 20 kept examples, the 'fixed' story)
"""
import json
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from plotting import draw_state

replay = sys.argv[1] if len(sys.argv) > 1 else "0"
data = json.load(open("results.json"))
steps_per_task = data["settings"]["steps_per_task"]
demo = data["by_replay"][replay]["demo"]
steps = demo["log"]["step"]

images = []
for i, step in enumerate(steps):
    phase = "learning Task A" if step <= steps_per_task else "learning Task B"
    fig = draw_state(demo, i, f"Step {step}: {phase}  (old examples kept: {replay})")
    fig.canvas.draw()
    images.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)

# pause on the first and last pictures so viewers can read them
durations = [0.25] * len(images)
durations[0] = 1.0
durations[-1] = 2.5
out = f"forgetting_replay{replay}.gif"
imageio.mimsave(out, images, duration=durations, loop=0)
print("saved", out, f"({len(images)} frames)")
