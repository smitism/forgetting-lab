# Forgetting Lab: why do neural networks forget?

A small neural network, built from scratch (no PyTorch), learns **Task A** and then **Task B**.
After learning B it has mostly forgotten A. This project measures that forgetting and tests one fix:
keeping a few old examples and mixing them into the new training ("replay").

![forgetting](forgetting_replay0.gif)

## The two tasks
Both tasks use dots on the same square. The network gets three inputs: `x`, `y`, and a task flag.

- **Task A:** is the dot on the left (class -1) or the right (class +1)?
- **Task B:** is the dot at the bottom (class -1) or the top (class +1)?

The task flag lets one network solve both tasks in principle. Forgetting is then a training problem,
not an impossible-task problem.

## Results
Accuracy on Task A **after** also learning Task B. Mean ± std over 5 random starts (seeds 0 to 4).

| Old examples kept | Task A accuracy after Task B | Task B accuracy |
|---|---|---|
| 0 (no fix) | 52% ± 3% | 100% |
| 5 | 75% ± 19% | 100% |
| 20 | 98% ± 2% | 100% |
| 50 | 100% ± 0% | 100% |

Task A accuracy was 100% for every run before Task B began, so the drop to about 52% (coin-flip level)
is forgetting. A small replay set recovers most of it, and 20 examples was nearly enough here.

**Caveats.** This is a tiny network on a toy problem, so the numbers say little about large models.
Five seeds is a small sample, and the 5-example result varies a lot between runs.

## How it works
- `engine.py`: a micrograd-style autograd engine (`Value`, `Neuron`, `Layer`, `MLP`).
- `experiment.py`: builds the tasks, trains A then B, tries each replay size across seeds, writes `results.json`.
- `plotting.py`, `app.py`, `make_gif.py`: picture, web app and GIF, all reading `results.json`.

The training step is the loop from micrograd: forward pass, loss, `zero_grad`, `backward`, nudge each weight.
Replay just adds the loss on the stored old examples to the loss on the new batch.

## Run it
```
pip install -r requirements.txt
python experiment.py        # about 2 minutes, writes results.json
streamlit run app.py        # interactive page
python make_gif.py 0        # forgetting_replay0.gif
python make_gif.py 20       # forgetting_replay20.gif
```

## Ideas to extend
- Protect important weights instead of storing examples (elastic weight consolidation).
- Add a third and fourth task and see whether forgetting accumulates.
- Redo it in PyTorch on handwritten digits.
