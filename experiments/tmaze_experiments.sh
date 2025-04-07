#!/usr/bin/env bash

echo "Running Decision Transformer..."
python -m train.trainer --config experiments/config_dt_tmaze.yaml

echo "Running Decision Transformer + HCAM..."
python -m train.trainer --config experiments/config_dt_hcam_tmaze.yaml
