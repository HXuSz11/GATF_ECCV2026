#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
RESULTS=${RESULTS:-../results_gatf/}

for SEED in 0
do
  CUDA_VISIBLE_DEVICES=${GPU} python src/main_incremental.py \
    --approach gatf --use-224 \
    --datasets imagenet_subset_kaggle --num-tasks 10 --nc-first-task 10 \
    --nepochs 200 --batch-size 256 --seed ${SEED} \
    --criterion ce --distiller mlp --adapter mlp --classifier bayes \
    --S 64 --lamb 5 --lr 0.1 --weight-decay 5e-4 --normalize --rotation \
    --distillation gatf --lambda-geometry 5 --lambda-transport-cycle 2 --lambda-isometry 0 \
    --preserve-pairwise-geometry --lambda-pairwise-geometry 0.05 \
    --prior-kind gls --prior-mode train_adapt --prior-ridge 1e-2 --prior-batches 50 --prior-residual-zero-init 1 \
    --gls-target-cov diag --no-adapt-train \
    --prior-update-interval 10 --prior-update-ema --prior-update-ema-m 0.90 \
    --exp-name imagenet100_10x10_GATF_GLSdiag_trainAdapt_updEMA_int10_m0p90_seed${SEED} \
    --results-path ${RESULTS}
done
