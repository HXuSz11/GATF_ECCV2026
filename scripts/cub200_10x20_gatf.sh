#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-0}
RESULTS=${RESULTS:-../results_gatf/}

for SEED in 0
do
  CUDA_VISIBLE_DEVICES=${GPU} python src/main_incremental.py \
    --approach gatf_cub200 \
    --datasets cub200 --num-tasks 10 --nc-first-task 20 \
    --nepochs 200 --batch-size 256 --seed ${SEED} \
    --criterion ce --distiller mlp --adapter mlp --classifier bayes \
    --S 32 --lamb 5 --lr 0.1 --weight-decay 5e-4 --normalize --rotation \
    --pretrained-net --use-224 \
    --distillation gatf --lambda-geometry 0.5 --lambda-transport-cycle 0.5 --lambda-isometry 0 \
    --lambda-pairwise-geometry 0 \
    --rho-ce 0.5 --lambda-fp-ce 1 --cutmix-alpha 0 --mixup-alpha 0 \
    --fp-use-grad-norm --lambda-fp-gn 1 \
    --DA-lr 0.05 --DA-wd 1e-4 \
    --prior-kind gls --prior-mode train_adapt --prior-ridge 1e-2 --prior-batches 50 --prior-residual-zero-init 1 \
    --gls-target-cov diag --no-adapt-train \
    --prior-update-interval 10 --prior-update-ema --prior-update-ema-m 0.90 \
    --exp-name cub200_10x20_GATF_GLSdiag_trainAdapt_fpce1_rhoce0p5_DAlr0p05_seed${SEED} \
    --results-path ${RESULTS}
done
