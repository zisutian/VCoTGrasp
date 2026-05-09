#!/usr/bin/env bash

TASK=run1
TB_DIR=../checkpoints/tb/${TASK}

python "export_tb_metrics.py" \
    --tb-dir "$TB_DIR" \
    --output-dir "$TB_DIR/vis"
