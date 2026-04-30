---
name: seqax-train
description: Launch a sweep of seqax training runs to measure each Hydra-style hyperparameter override in isolation and in combination, then report a comparison table. Use when the user asks to "train", "kick off training", "sweep", "ablate", or run train.py with a specific config (e.g. c4_a100x8_1b) and one or more overrides.
---

# seqax-train

Launches a **sweep** of `train.py` (Hydra + wandb) runs from the seqax repo root: one baseline, one per individual override, and one with all overrides combined. Monitors them all via wandb, then emits a table that shows how each override moves the needle on its own and when stacked with the others.

## Inputs

The user's `args` to this skill should be:

```
<config_name> [override1=value1 override2=value2 ...]
```

- `config_name` — required. The base name of a file in `configs/` **without** the `.yaml` extension (e.g. `c4_a100x8_1b`). Validate it exists by listing `configs/`. If the user gives just a fragment, match it; if ambiguous, ask.
- Overrides — **one or more** Hydra dotted-path overrides (e.g. `training.learning_rate=1e-4 model.d_ff=2048`). Zero overrides is a degenerate case — ask the user what they want to vary before proceeding. Do **not** invent overrides the user did not ask for.

If the user did not specify `wandb_project=...` in the overrides, ask which wandb project to log to before launching — `train.py` only initializes `wandb.init(...)` when `config.wandb_project` is set, so without it metrics won't be tracked. The wandb project is a *sweep-wide* override and applies to every run in the sweep.

## Sweep design

Given N user-supplied overrides `O_1 ... O_N` (excluding `wandb_project`), launch **N + 2** runs:

| Leg              | Overrides applied                          | Purpose                                  |
|------------------|--------------------------------------------|------------------------------------------|
| `baseline`       | (none)                                     | Reference point — pure config defaults.  |
| `only_<key_i>`   | `O_i` only, for each i in 1..N             | Isolated impact of each override.        |
| `all`            | `O_1 ... O_N`                              | Combined impact (interactions).          |

Special case: if N = 1 you still launch 3 runs (`baseline`, `only_<key_1>`, `all`) — the `only_` and `all` legs are duplicates in that case, so collapse to 2 runs (`baseline`, `with_<key_1>`) and note it in the report.

Derive a unique `paths.model_name` per leg so wandb run names and checkpoint dirs don't collide — e.g. `<config_name>__sweep_<timestamp>__<leg>`. The `paths.model_name` value is passed as the wandb run name automatically by `train.py`.

## Command (per leg)

Run from the repo root (`/Users/ankur/dev/ai_space/seqax`):

```
python train.py --config-name=<config_name> <leg_overrides...> wandb_project=<project> paths.model_name=<leg_model_name>
```

Hydra resolves `--config-name` against `configs/` (already wired via `@hydra.main(config_path="configs", ...)` in `train.py`).

## Procedure

1. Parse `args` into `config_name` and the override list. Split out `wandb_project=...` if present; treat it as sweep-wide.
2. Confirm `configs/<config_name>.yaml` exists (use Glob/Read).
3. If no `wandb_project` is set, ask the user which wandb project to use for the whole sweep.
4. Build the leg list per the "Sweep design" table above. Generate a shared sweep timestamp and the per-leg `paths.model_name` values.
5. Echo the full launch plan: one line per leg showing its name and the exact command you're about to run. Ask the user to confirm before launching more than ~3 runs.
6. Launch the legs via Bash from the repo root. Capture stdout for each run to confirm wandb run URLs.
7. **Start a monitoring loop.** Immediately after launching the sweep, invoke the `loop` skill with a 2-minute interval to babysit all runs together:

   ```
   /loop 2m check seqax wandb sweep <sweep_id> (runs: <run1>, <run2>, ...): pull latest scalars for each run (loss + key metrics), confirm each run's loss is trending down and nothing looks pathological (NaN, exploding grads, flat loss, throughput collapse). Report a one-line status per run each tick. When all runs have finished (or any has hard-failed), stop the loop and emit the final comparison table.
   ```

   Each tick should:
   - For every still-running run, fetch recent scalars via the `wandb` Python API (`wandb.Api().run("project/run_id").history()`).
   - Compare the latest loss window against the previous window per run — flag if loss is flat or rising over multiple ticks.
   - Sanity-check other metrics if present (grad norm finite, throughput non-zero, lr following schedule).
   - Emit one short line per run: `<leg> step=N loss=X (Δ=...) status=ok|warn|fail`.
   - Stop the loop (omit the next ScheduleWakeup) once every leg is in a terminal state, or when a confirmed regression needs user attention.

## Final output: comparison table

When the sweep finishes (or a fail-fast triggers), emit a markdown table where **columns are sweep legs** (the hyperparameter settings) and **rows are metrics**. One column per leg, in launch order: `baseline`, each `only_<key_i>`, then `all`.

```
| metric             | baseline | only_<key_1> | only_<key_2> | ... | all |
|--------------------|---------:|-------------:|-------------:|-----|----:|
| overrides          |    —     | O_1          | O_2          | ... | O_1,O_2,... |
| final loss         |          |              |              |     |     |
| best loss          |          |              |              |     |     |
| Δ vs baseline      |    0     | ±x           | ±y           | ... | ±z  |
| grad norm (final)  |          |              |              |     |     |
| throughput (tok/s) |          |              |              |     |     |
| MFU %              |          |              |              |     |     |
| steps completed    |          |              |              |     |     |
| wandb run          |          |              |              |     |     |
```

- `Δ vs baseline` row is `final_loss(leg) − final_loss(baseline)`; negative means the override helped. Render as `-0.1234` / `+0.0087`.
- If any metric isn't available for a leg (e.g. run failed), put `—` in that cell and add a footnote explaining why.
- After the table, add 2–4 sentences interpreting the result:
  1. Which individual override moved loss the most?
  2. Did the `all` leg beat or underperform the sum of individual effects? (i.e. are there interactions?)
  3. Any leg that diverged or looked pathological?
  4. A concrete recommendation: which overrides to keep, which to drop, what to try next.

Keep the table and the interpretation to the point — it's the artifact the user came for, not a running log.

## Notes

- Do not edit YAML files to apply overrides — always pass them on the command line so the config file stays clean.
- Do not invent overrides the user did not ask for. The sweep legs are **only** `baseline`, one-at-a-time, and all-together over the user's list.
- `train.py` initializes wandb when `config.wandb_project` is set (`train.py:531-555`), logging model and training hparams. Metrics are logged via `training_io.log()` which calls `wandb_run.log(metrics_dict, step=step)`.
- The sweep shares one timestamp suffix across legs so the user can filter wandb for all runs in one sweep.
