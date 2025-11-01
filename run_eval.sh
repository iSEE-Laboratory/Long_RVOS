#!/bin/bash

PRED_PATH=$1
SPLIT=$2

python /data1/tianming/long_rvos/eval/eval_long_rvos/eval_static.py --pred_path "$PRED_PATH" --split "$SPLIT"

python /data1/tianming/long_rvos/eval/eval_long_rvos/eval_dynamic.py --pred_path "$PRED_PATH" --split "$SPLIT"

python /data1/tianming/long_rvos/eval/eval_long_rvos/eval_hybrid.py --pred_path "$PRED_PATH" --split "$SPLIT"

python /data1/tianming/long_rvos/eval/eval_long_rvos/eval_overall.py --pred_path "$PRED_PATH" --split "$SPLIT"