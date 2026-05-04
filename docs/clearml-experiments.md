# clearml_experiments file structure

Exported ClearML experiment data lives in `clearml_experiments/`. Three-level hierarchy: **group → variant → run**.

```
clearml_experiments/
├── <group>/                          # dataset_hardware_paramcount (e.g. slim_v4_32_31m)
│   ├── <variant>/                    # branch or feature name (e.g. lcm, main, alibi)
│   │   ├── <experiment_name>_<id>/   # one training run
│   │   │   ├── metadata.json
│   │   │   ├── metrics/
│   │   │   │   ├── loss_loss.csv
│   │   │   │   ├── learning_rate_learning_rate.csv
│   │   │   │   ├── grad_norm_grad_norm.csv
│   │   │   │   ├── raw_grad_norm_raw_grad_norm.csv
│   │   │   │   ├── final_loss_eval.csv
│   │   │   │   ├── final_perplexity_eval.csv
│   │   │   │   └── :monitor:machine_*.csv
│   │   │   └── logs/
│   │   │       └── task.log
│   │   └── ...
│   └── ...
└── ...
```

## Groups (36 total)

| Group | Variants | Total runs | Top variants |
|---|---|---|---|
| slim_v4_32_31m | 11 | 295 | lcm(127), cluster_qk_alibi_lora(66), cluster_qk_alibi(31), cluster_qk(29) |
| c4_a100x8x4_37m | 7 | 150 | power_scheduler(95), main(23), exponent_scaling(14) |
| c4_v4_32_31m | 5 | 57 | parition_qk_alibi_embed_post_training_retrieval_penalty(19), parition_qk_alibi_embed_post_training(18), alibi(11) |
| c4_a100x8x4_10m | 2 | 53 | mup(29), hpo(24) |
| longcrawl_v4_32_31m | 7 | 42 | attn_temp(14), parition_qk_alibi_embed_post_training(14), mla(5), main(4) |
| synthetic_v4_32_15m | 5 | 29 | cope(15), main(9) |
| wikitext103_v4_32_31m | 2 | 28 | cope(27) |
| ClearML_examples | 3 | 23 | HyperParameter_Optimization(12), ML___DL_Frameworks(9) |
| starcoder_v4_32_84m | 4 | 17 | lcm(8), main(6) |
| c4_a100x8x4_10m_narrow | 1 | 16 | hpo(16) |
| c4_a100x8x4_270m | 6 | 16 | exponent_scaling(4), mup(4), shared_kv(4) |
| synthetic_rtx4090_15m | 4 | 16 | cope(7), synthetic_benchmark(7) |
| c4_a100x8x4_13m | 4 | 14 | exponent_scaling(7), main(3), power_scheduler(3) |
| wikitext103_rtx4090_31m | 1 | 12 | cope(12) |
| slim_v4_32_84m | 3 | 10 | lcm(4), main(4) |
| starcoder_v4_32_31m | 4 | 8 | lcm(4), main(2) |
| c4_a100x8x4_1b | 4 | 8 | main(3), mup(2), shared_kv(2) |
| slim_v4_32_270m | 2 | 6 | main(3), ntk(3) |
| synthetic_rtx4090_12m | 1 | 4 | synthetic_benchmark(4) |

## metadata.json schema

```json
{
  "experiment_name": "slim_v4-32_31m_block_size=1_...",
  "experiment_id": "d3a41adb55954745b2a318ebc9032c1b",
  "project": "a96b194906f147afb87f685fd48442d2",
  "status": "completed",           // completed | running | failed
  "created": "2025-01-19 22:06:55.557000+00:00",
  "last_updated": "2025-01-19 22:43:37.381000+00:00",
  "user": "883f32d0559342e2839ec37cbfba2034",
  "url": "https://app.clear.ml/projects/.../tasks/...",
  "hyperparameters": {
    "Hydra/model.<key>": "<value>",       // model arch: layers, block_size, d_model, d_ff, n_kv, ...
    "Hydra/training.<key>": "<value>",    // training: learning_rate, queue, n_log_iterations, ...
    "Hydra/paths.<key>": "<value>",       // output paths
    "Hydra/flat_tokens.<key>": "<value>", // data source
    "Args/config_name": "slim_v4-32_31m",
    "launch_multi_node/total_num_nodes": 4,
    "launch_multi_node/queue": "v4-32-workers"
  }
}
```

Hyperparameter values are strings or numbers. The `Hydra/` prefix is the useful namespace; strip `Hydra/model.` or `Hydra/training.` to get the config key name.

## Metric CSV schema

All CSVs: two columns, `step` and `value`.

```csv
step,value
0,10.397207
1,10.397207
2,10.387990
```

Training metrics: `loss_loss.csv`, `learning_rate_learning_rate.csv`, `grad_norm_grad_norm.csv`, `raw_grad_norm_raw_grad_norm.csv`.
Eval metrics (typically 1 row): `final_loss_eval.csv`, `final_perplexity_eval.csv`.
System metrics: `:monitor:machine_cpu_usage.csv`, `*_memory_used_gb.csv`, `*_disk_free_percent.csv`, `*_io_*.csv`, `*_network_*.csv`.

## Conventions used by existing reports

The `docs/lcm.qmd` report (the reference example):

- Points `EXPERIMENTS_DIR` at `clearml_experiments/<group>/<variant>` (one variant at a time)
- Points `BASELINE_DIR` at a specific run directory for the control
- Parses `metadata.json` hyperparameters to build a config dict per run
- Reads `metrics/loss_loss.csv` for loss curves
- Computes EMA-smoothed final loss for ranking
- Filters runs by creation timestamp (to exclude a known-buggy window)
- Plots loss curves (raw + EMA) with plotly
- Emits a styled pandas table of top-k runs with hyperparameter columns
