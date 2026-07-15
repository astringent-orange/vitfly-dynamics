# External experiment artifacts

The `code-release` branch intentionally contains source code, small
configuration files, and documentation only. Large generated or downloaded
artifacts are not stored in Git.

## Training

Place a versioned dataset archive so that extraction produces:

```text
training/datasets/<dataset_name>/
├── accepted_manifest.csv
└── <trajectory_id>/
    ├── data.csv
    └── <timestamp>.png
```

For STEP2, use `dataset_name = dynamic_astar_medium` and start training from
the repository root with either:

```bash
python3 training/train.py --config training/config/train_vitlstm_1f.txt
python3 training/train.py --config training/config/train_two_frame_vit_lstm.txt
```

The two configurations use the same accepted manifest and random seed.

## Simulation and evaluation

Fetch Unity binaries, Flightmare environment materials, generated
`dynamic_astar_medium` environments, trained checkpoints, and optional media
from the project artifact store. Record each archive's version, SHA-256 hash,
and source-code commit alongside the experiment results.

Do not commit datasets, model checkpoints, rollout output, Unity binaries, or
generated environment assets to this repository.
