#!/usr/bin/env bash
set -euo pipefail

GPU=${GPU:-2}
RESULTS=${RESULTS:-../results_gatf/}
EMAS=(0.90)

for SEED in 0
do
  for M in "${EMAS[@]}"
  do
    M_TAG=$(echo "${M}" | sed 's/\./p/g')

    CUDA_VISIBLE_DEVICES=${GPU} python src/main_incremental.py \
      --approach gatf \
      --datasets cifar100_icarl --num-tasks 5 --nc-first-task 20 \
      --nepochs 200 --batch-size 256 --seed ${SEED} \
      --criterion ce --distiller mlp --adapter mlp --classifier bayes \
      --S 64 --lamb 5 --lr 0.1 --weight-decay 5e-4 --normalize --rotation \
      --distillation gatf --lambda-geometry 5 --lambda-transport-cycle 0 --lambda-isometry 0 \
      --preserve-pairwise-geometry --lambda-pairwise-geometry 0.05 \
      --prior-kind gls --prior-mode train_adapt --prior-ridge 1e-2 --prior-batches 50 --prior-residual-zero-init 1 \
      --gls-target-cov diag \
      --prior-update-interval 10 --prior-update-ema --prior-update-ema-m ${M} \
      --exp-name cifar100_5x20_GATF_GLSdiag_trainAdapt_updEMA_int10_m${M_TAG}_seed${SEED} \
      --results-path ${RESULTS}
  done
done
