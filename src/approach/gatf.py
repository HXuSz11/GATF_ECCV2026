# -*- coding: utf-8 -*-
import copy
import random
import os
import csv

import torch
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet18 import resnet18
from .models.vit import vit_small
from .models.resnet32 import resnet32
from .incremental_learning import Inc_Learning_Appr
from .criterions.proxy_nca import ProxyNCA
from .criterions.proxy_yolo import ProxyYolo
from .criterions.ce import CE

from torch.distributions.multivariate_normal import MultivariateNormal


# ------------------------------ Pseudo dataset for linear head ------------------------------

class SampledDataset(torch.utils.data.Dataset):
    """Sample pseudo-prototypes from class Gaussians to train a linear head."""
    def __init__(self, distributions, samples, task_offset):
        self.distributions = distributions
        self.samples = samples
        self.total_classes = task_offset[-1]
    def __len__(self): return self.samples
    def __getitem__(self, index):
        target = random.randint(0, self.total_classes - 1)
        val = self.distributions[target].sample()
        return val, target


# ------------------------------ Prior adapters (fixed prior + learnable residual) ------------------------------

class RidgePriorAdapter(nn.Module):
    """
    Fixed linear prior: y = x @ P + b  (row-vector convention),
    implemented via nn.Linear (y = x @ W^T + bias) so set W = P^T.
    """
    def __init__(self, P: torch.Tensor, residual: nn.Module, bias: torch.Tensor = None, zero_init_residual: bool = True):
        super().__init__()
        assert P.dim() == 2 and P.shape[0] == P.shape[1], "P must be [S,S]"
        S = P.shape[0]
        self.prior = nn.Linear(S, S, bias=(bias is not None))
        with torch.no_grad():
            self.prior.weight.copy_(P.t().contiguous())
            if bias is not None:
                self.prior.bias.copy_(bias.contiguous())
        for p in self.prior.parameters():
            p.requires_grad = False

        self.residual = residual
        if zero_init_residual:
            self._zero_init(self.residual)

    @staticmethod
    def _zero_init(m: nn.Module):
        for mm in m.modules():
            if isinstance(mm, nn.Linear):
                nn.init.zeros_(mm.weight)
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prior(x) + self.residual(x)


class WhitenedProcrustesPriorAdapter(nn.Module):
    """
    Whitened Procrustes prior (affine):
        x_c = x - mu_old
        y_c ≈ x_c @ A
        y = y_c + mu_new
    where A = Sigma_old^{-1/2} R Sigma_new^{1/2}, and R is orthogonal from Procrustes
    in whitened space.

    Then final:
        A_prior(x) = prior(x) + residual(x)
    """
    def __init__(self,
                 mu_old: torch.Tensor,
                 mu_new: torch.Tensor,
                 A: torch.Tensor,
                 residual: nn.Module,
                 zero_init_residual: bool = True):
        super().__init__()
        assert mu_old.dim() == 1 and mu_new.dim() == 1
        assert A.dim() == 2 and A.shape[0] == A.shape[1] == mu_old.numel()
        self.register_buffer("mu_old", mu_old.detach().clone())
        self.register_buffer("mu_new", mu_new.detach().clone())
        self.register_buffer("A", A.detach().clone())

        self.residual = residual
        if zero_init_residual:
            self._zero_init(self.residual)

    @staticmethod
    def _zero_init(m: nn.Module):
        for mm in m.modules():
            if isinstance(mm, nn.Linear):
                nn.init.zeros_(mm.weight)
                if mm.bias is not None:
                    nn.init.zeros_(mm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A.to(dtype=x.dtype)
        mu_old = self.mu_old.to(dtype=x.dtype)
        mu_new = self.mu_new.to(dtype=x.dtype)
        y_prior = (x - mu_old) @ A + mu_new
        return y_prior + self.residual(x)


# ----------------------------------------- Approach -----------------------------------------

class Appr(Inc_Learning_Appr):
    """
    Geometry-Anchored Transport Framework (GATF) with GLS prior transport and optional geometry-preserving regularization.

    Prior adapter upgrades (NEW unified interface):
      --prior-kind {none, dpcr, wpro, gls}
      --prior-mode {off, train, adapt, train_adapt}
      --prior-ridge
      --prior-batches
      --prior-residual-zero-init
      --gls-target-cov {full, diag}  (only for gls)

    NEW: optional periodic prior re-estimation in TRAIN stage + (hard/EMA) in-place update:
      --prior-update-interval K    (0 disables; otherwise every K epochs recompute prior in train_backbone)
      --prior-update-ema           (if set, use EMA update instead of hard overwrite)
      --prior-update-ema-m m       (EMA momentum; prior <- m*prior + (1-m)*prior_new)

    Optional drift logging:
      --log-drift-every E  (0 disables; otherwise every E epochs output drift_task{t}_epoch{e}.csv)

    CUB/fine-grained training options:
      - Feature Perturbation on classification loss in feature space:
        --rho-ce, --lambda-fp-ce, --fp-use-grad-norm, --lambda-fp-gn
      - Separate LR/WD for distiller+adapter:
        --DA-lr, --DA-wd
    """
    def __init__(
        self,
        model,
        device,
        nepochs=200,
        lr=0.05,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=1,
        momentum=0,
        wd=0,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1,
        fix_bn=False,
        eval_on_train=False,
        num_tasks=5,
        nc_first_task=20,
        datasets="tiny",
        logger=None,
        N=10000,
        alpha=1.0,
        lr_backbone=0.01,
        lr_adapter=0.01,
        beta=1.0,
        distillation="projected",
        use_224=False,
        S=64,
        dump=False,
        rotation=False,
        distiller="linear",   # "linear" | "mlp"
        adapter="linear",     # "linear" | "mlp"
        criterion="proxy-nca",
        lamb=10,
        tau=2,
        smoothing=0.0,
        sval_fraction=0.95,
        adaptation_strategy="full",
        pretrained_net=False,
        normalize=False,
        shrink=0.0,
        multiplier=32,
        classifier="bayes",
        nnet="resnet18",

        # Geometry transport loss weights
        lambda_geometry=None,
        lambda_transport_cycle=1.0,
        lambda_isometry=0.0,
        preserve_pairwise_geometry=False,
        lambda_pairwise_geometry=0.1,

        # Aux old-class logit KD (vanilla preserved)
        aux_logit_kd=False,
        aux_logit_kd_weight=None,
        aux_logit_kd_tau=None,
        kd_rampup_fraction=0.3,

        # Drift utilities
        drift_scale=1.0,
        drift_momentum=0.5,
        drift_proto_blend=False,

        # Optional backbone transfer
        use_transfer_backbone=False,
        transfer_backbone_weight=0.5,
        transfer_backbone_rampup_fraction=0.3,

        # ------------------------ KD gates + EMA + Proto-contrast ------------------------
        kd_conflict_aware=False,
        kd_conflict_gamma=10.0,
        kd_conflict_wmin=0.2,
        kd_conflict_wmax=1.0,

        kd_gate_mode='off',           # 'off' | 'conflict' | 'proto' | 'hybrid' | 'conflict_only'
        kd_proto_thresh=0.6,
        kd_proto_gamma=10.0,
        kd_floor_min=0.6,
        kd_floor_max=0.95,

        new_head_ema_kd=False,
        new_head_ema_m=0.99,
        new_head_ema_tau=2.0,
        new_head_ema_weight=0.5,

        proto_contrast=False,
        proto_contrast_weight=0.5,
        proto_contrast_margin=0.2,
        proto_contrast_topk=10,
        proto_contrast_metric="cos",       # 'maha' | 'cos' | 'l2'
        proto_contrast_rampup_fraction=0.3,

        print_gate_diag=False,

        # -------------------- PRIOR adapter upgrade (NEW) --------------------
        prior_kind="none",              # 'none' | 'dpcr' | 'wpro' | 'gls'
        prior_mode="off",               # 'off' | 'train' | 'adapt' | 'train_adapt'
        prior_ridge=1e-3,               # for example, 1e-2 in the camera-ready scripts
        prior_batches=50,
        prior_residual_zero_init=1,
        gls_target_cov="full",          # 'full' | 'diag' (only for gls)

        # -------------------- NEW: train-stage periodic prior update --------------------
        prior_update_interval=0,        # 0 disables; otherwise every K epochs
        prior_update_ema=False,         # use EMA update instead of hard overwrite
        prior_update_ema_m=0.90,        # EMA momentum

        # -------------------- NEW: drift logging --------------------
        log_drift_every=0,              # 0 disables

        # -------------------- Adapter refinement gate --------------------
        adapt_train=True,              # False with --no-adapt-train: reuse the co-evolved adapter only

        # --------------------  Feature Perturbation knobs --------------------
        rho_ce=0.0,                     # FP step size for classification loss (A)
        lambda_fp_ce=0.5,               # weight for adversarial classification loss
        fp_use_grad_norm=False,         # enable grad-norm penalty (second-order)
        lambda_fp_gn=0.1,               # weight for grad-norm penalty
        mixup_alpha=0.0,                 # parsed for CUB script compatibility; camera-ready runs keep this at 0
        cutmix_alpha=0.0,                # parsed for CUB script compatibility; camera-ready runs keep this at 0

        # --------------------  Distiller/Adapter LR/WD overrides --------------------
        DA_lr=0.0,                      # distiller+adapter LR (0 -> fallback)
        DA_wd=0.0,                      # distiller+adapter WD (0 -> fallback)
    ):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience,
                                   clipgrad, momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor,
                                   fix_bn, eval_on_train, logger, exemplars_dataset=None)

        # ---------------- Basic hyper-parameters ----------------
        self.N, self.S, self.dump = N, S, dump
        self.lamb, self.alpha, self.beta, self.tau = lamb, alpha, beta, tau
        self.lr_backbone, self.lr_adapter = lr_backbone, lr_adapter
        self.multiplier, self.shrink, self.smoothing = multiplier, shrink, smoothing
        self.adaptation_strategy = adaptation_strategy
        self.old_model = None
        self.pretrained = pretrained_net

        # ---------------- NEW: drift logging ----------------
        self.log_drift_every = int(log_drift_every)
        self.all_val_loader = None  # will be set by train_loop

        # ---------------- Adapter refinement gate ----------------
        self.adapt_train = bool(adapt_train)

        # ---------------- PRIOR config ----------------
        self.prior_kind = str(prior_kind).lower()
        self.prior_mode = str(prior_mode).lower()
        assert self.prior_kind in ("none", "dpcr", "wpro", "gls")
        assert self.prior_mode in ("off", "train", "adapt", "train_adapt")
        self.prior_ridge = float(prior_ridge)
        self.prior_batches = int(prior_batches)
        self.prior_residual_zero_init = bool(int(prior_residual_zero_init))
        self.gls_target_cov = str(gls_target_cov).lower()
        assert self.gls_target_cov in ("full", "diag")
        self._prior_cache = None

        # NEW: periodic prior updates in TRAIN stage
        self.prior_update_interval = int(prior_update_interval)
        self.prior_update_ema = bool(prior_update_ema)
        self.prior_update_ema_m = float(prior_update_ema_m)
        if self.prior_update_ema_m < 0.0 or self.prior_update_ema_m >= 1.0:
            raise ValueError("--prior-update-ema-m must be in [0, 1).")

        # ----------------  Feature Perturbation knobs ----------------
        self.rho_ce = float(rho_ce)
        self.lambda_fp_ce = float(lambda_fp_ce)
        self.fp_use_grad_norm = bool(fp_use_grad_norm)
        self.lambda_fp_gn = float(lambda_fp_gn)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        if self.mixup_alpha != 0.0 or self.cutmix_alpha != 0.0:
            raise NotImplementedError("The cleaned GATF camera-ready implementation accepts --mixup-alpha/--cutmix-alpha for script compatibility, but the final runs keep both at 0.")

        # ----------------  Distiller/Adapter LR/WD overrides ----------------
        self.DA_lr_user = float(DA_lr)
        self.DA_wd_user = float(DA_wd)
        # Keep defaults close to legacy behavior:
        # - If user passes --DA-lr > 0: use it.
        # - Else: use lr_adapter when pretrained, otherwise fall back to main lr (so non-pretrained behavior stays similar).
        self.DA_lr = self.DA_lr_user if self.DA_lr_user > 0 else (self.lr_adapter if self.pretrained else self.lr)
        self.DA_wd = self.DA_wd_user if self.DA_wd_user > 0 else self.wd

        self.nnet = str(nnet).lower()
        self.distiller_type = str(distiller).lower()
        self.adapter_type   = str(adapter).lower()
        self.distillation   = str(distillation).lower()
        assert self.nnet in ("resnet18", "resnet32", "vit")
        assert self.distiller_type in ("linear", "mlp")
        assert self.adapter_type   in ("linear", "mlp")
        assert self.distillation   in ("gatf", "projected", "logit", "feature", "none")

        upd_msg = "none"
        if (self.prior_update_interval is not None) and (self.prior_update_interval > 0):
            upd_msg = f"{'ema' if self.prior_update_ema else 'hard'}@{self.prior_update_interval} (m={self.prior_update_ema_m:.3f})"

        print(
            f"[GATF] nnet={self.nnet}, distiller={self.distiller_type}, adapter={self.adapter_type} | "
            f"prior={self.prior_kind}/{self.prior_mode} ridge={self.prior_ridge:g} batches={self.prior_batches} "
            f"priorUpdate={upd_msg} | "
            f"FP(rho_ce={self.rho_ce:g}, lambda_fp_ce={self.lambda_fp_ce:g}, "
            f"gradNorm={'on' if self.fp_use_grad_norm else 'off'}, lambda_fp_gn={self.lambda_fp_gn:g}) | "
            f"DA(lr={self.DA_lr:g}, wd={self.DA_wd:g}) | "
            f"logDriftEvery={self.log_drift_every} | "
            f"adaptTrain={self.adapt_train}"
        )

        # ---------------- Build backbone ----------------
        if self.nnet == "vit":
            state_dict = torch.load("dino_deitsmall16_pretrain.pth")
            self.model = vit_small(num_features=S)
            self.model.load_state_dict(state_dict, strict=False)
            for name, p in self.model.named_parameters():
                if "blocks.11" not in name: p.requires_grad = False
        elif self.nnet == "resnet18":
            self.model = resnet18(num_features=S, is_224=use_224)
            if pretrained_net:
                state_dict = torch.load("./resnet18-f37072fd.pth")
                state_dict.pop("fc.weight", None); state_dict.pop("fc.bias", None)
                self.model.load_state_dict(state_dict, strict=False)
        elif self.nnet == "resnet32":
            self.model = resnet32(num_features=S)
            if pretrained_net or use_224: raise RuntimeError("No pretrained weights for resnet32")
        self.model.to(device, non_blocking=True)

        # ---------------- Caches & accounting ----------------
        self.train_data_loaders, self.val_data_loaders = [], []
        self.means = torch.empty((0, self.S), device=self.device)
        self.covs = torch.empty((0, self.S, self.S), device=self.device)
        self.covs_raw = torch.empty((0, self.S, self.S), device=self.device)
        self.covs_inverted = None
        self.classifier = classifier
        self.pseudo_head = None
        self.is_normalization = normalize
        self.is_rotation = rotation
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.criterion_type = criterion
        self.criterion = {"proxy-yolo": ProxyYolo, "proxy-nca": ProxyNCA, "ce": CE}[criterion]
        self.heads = torch.nn.ModuleList()
        self.sval_fraction = sval_fraction

        # ---------------- Geometry-anchored (existing) ----------------
        self.lambda_geometry = self.lamb if (lambda_geometry is None) else float(lambda_geometry)
        self.lambda_transport_cycle = float(lambda_transport_cycle)
        self.lambda_isometry = float(lambda_isometry)
        self.preserve_pairwise_geometry = bool(preserve_pairwise_geometry)
        self.lambda_pairwise_geometry = float(lambda_pairwise_geometry)

        self.svals_explained_by = []
        self.distiller_module = None
        self.adapter_module = None

        # ---------------- Optional knobs ----------------
        self.use_aux_logit_kd = bool(aux_logit_kd)
        self.lambda_aux_logit_kd = float(aux_logit_kd_weight) if (aux_logit_kd_weight is not None) else float(self.lamb)
        self.tau_aux = float(aux_logit_kd_tau) if (aux_logit_kd_tau is not None) else float(self.tau)
        self.kd_rampup_fraction = float(kd_rampup_fraction)

        self.drift_scale = float(drift_scale)
        self.drift_momentum = float(drift_momentum)
        self.use_drift_proto_blend = bool(drift_proto_blend)
        self._drift_eps = 1e-8
        self._drift_weights_ema = None

        self.use_transfer_backbone = bool(use_transfer_backbone)
        self.lambda_transfer_backbone = float(transfer_backbone_weight)
        self.transfer_backbone_rampup_fraction = float(transfer_backbone_rampup_fraction)

        # ---------------- KD gates / EMA / Proto-contrast ----------------
        self.kd_conflict_aware = bool(kd_conflict_aware)
        self.kd_conflict_gamma = float(kd_conflict_gamma)
        self.kd_conflict_wmin = float(kd_conflict_wmin)
        self.kd_conflict_wmax = float(kd_conflict_wmax)

        self.kd_gate_mode = str(kd_gate_mode).lower()
        self.kd_proto_thresh = float(kd_proto_thresh)
        self.kd_proto_gamma  = float(kd_proto_gamma)
        self.kd_floor_min    = float(kd_floor_min)
        self.kd_floor_max    = float(kd_floor_max)

        self.use_new_head_ema_kd = bool(new_head_ema_kd)
        self.new_head_ema_m = float(new_head_ema_m)
        self.new_head_ema_tau = float(new_head_ema_tau)
        self.new_head_ema_weight = float(new_head_ema_weight)
        self.head_ema = None

        self.proto_contrast = bool(proto_contrast)
        self.proto_contrast_weight = float(proto_contrast_weight)
        self.proto_contrast_margin = float(proto_contrast_margin)
        self.proto_contrast_topk = int(proto_contrast_topk)
        self.proto_contrast_metric = str(proto_contrast_metric).lower()
        self.proto_contrast_rampup_fraction = float(proto_contrast_rampup_fraction)

        # -------------------- Gate diag (existing) --------------------
        self.print_gate_diag = bool(print_gate_diag)
        self._gate_diag_delta = 0.05
        self._gate_applied_thresh = 0.90

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        # Basics
        parser.add_argument('--N', type=int, default=10000)
        parser.add_argument('--S', type=int, default=64)
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--beta', type=float, default=1.0)
        parser.add_argument('--lamb', type=float, default=10)
        parser.add_argument('--lr-backbone', type=float, default=0.01)
        parser.add_argument('--lr-adapter', type=float, default=0.01)
        parser.add_argument('--multiplier', type=int, default=32)
        parser.add_argument('--tau', type=float, default=2)
        parser.add_argument('--shrink', type=float, default=0.0)
        parser.add_argument('--sval-fraction', type=float, default=0.95)
        parser.add_argument('--adaptation-strategy', type=str, choices=["none", "mean", "diag", "full"], default="full")
        parser.add_argument('--distiller', type=str, choices=["linear", "mlp"], default="mlp")
        parser.add_argument('--adapter', type=str, choices=["linear", "mlp"], default="mlp")
        parser.add_argument('--criterion', type=str, choices=["ce", "proxy-nca", "proxy-yolo"], default="ce")
        parser.add_argument('--nnet', type=str, choices=["vit", "resnet18", "resnet32"], default="resnet18")
        parser.add_argument('--classifier', type=str, choices=["linear", "bayes", "nmc"], default="bayes")
        parser.add_argument('--distillation', type=str, choices=["gatf", "projected", "logit", "feature", "none"], default="gatf")
        parser.add_argument('--smoothing', type=float, default=0.0)
        parser.add_argument('--use-224', action='store_true', default=False)
        parser.add_argument('--pretrained-net', action='store_true', default=False)
        parser.add_argument('--normalize', action='store_true', default=False)
        parser.add_argument('--dump', action='store_true', default=False)
        parser.add_argument('--rotation', action='store_true', default=False)

        # Geometry-anchored transport
        parser.add_argument('--lambda-geometry', type=float, default=None)
        parser.add_argument('--lambda-transport-cycle', type=float, default=1.0)
        parser.add_argument('--lambda-isometry', type=float, default=0.0)
        parser.add_argument('--preserve-pairwise-geometry', action='store_true', default=False)
        parser.add_argument('--lambda-pairwise-geometry', type=float, default=0.1)

        # Aux KD
        parser.add_argument('--aux-logit-kd', action='store_true', default=False)
        parser.add_argument('--aux-logit-kd-weight', type=float, default=None)
        parser.add_argument('--aux-logit-kd-tau', type=float, default=None)
        parser.add_argument('--kd-rampup-fraction', type=float, default=0.3)

        # Drift weights
        parser.add_argument('--drift-scale', type=float, default=1.0)
        parser.add_argument('--drift-momentum', type=float, default=0.5)
        parser.add_argument('--drift-proto-blend', action='store_true', default=False)

        # Backbone transfer
        parser.add_argument('--use-transfer-backbone', action='store_true', default=False)
        parser.add_argument('--transfer-backbone-weight', type=float, default=0.5)
        parser.add_argument('--transfer-backbone-rampup-fraction', type=float, default=0.3)

        # Gating + EMA + Proto-contrast
        parser.add_argument('--kd-conflict-aware', action='store_true', default=False)
        parser.add_argument('--kd-conflict-gamma', type=float, default=10.0)
        parser.add_argument('--kd-conflict-wmin', type=float, default=0.2)
        parser.add_argument('--kd-conflict-wmax', type=float, default=1.0)

        parser.add_argument('--kd-gate-mode', type=str, choices=['off','conflict','proto','hybrid','conflict_only'], default='off')
        parser.add_argument('--kd-proto-thresh', type=float, default=0.6)
        parser.add_argument('--kd-proto-gamma', type=float, default=10.0)
        parser.add_argument('--kd-floor-min', type=float, default=0.6)
        parser.add_argument('--kd-floor-max', type=float, default=0.95)

        parser.add_argument('--new-head-ema-kd', action='store_true', default=False)
        parser.add_argument('--new-head-ema-m', type=float, default=0.99)
        parser.add_argument('--new-head-ema-tau', type=float, default=2.0)
        parser.add_argument('--new-head-ema-weight', type=float, default=0.5)

        parser.add_argument('--proto-contrast', action='store_true', default=False)
        parser.add_argument('--proto-contrast-weight', type=float, default=0.5)
        parser.add_argument('--proto-contrast-margin', type=float, default=0.2)
        parser.add_argument('--proto-contrast-topk', type=int, default=10)
        parser.add_argument('--proto-contrast-metric', type=str, choices=['maha','cos','l2'], default='cos')
        parser.add_argument('--proto-contrast-rampup-fraction', type=float, default=0.3)

        parser.add_argument('--print-gate-diag', action='store_true', default=False)

        # PRIOR (NEW unified)
        parser.add_argument('--prior-kind', type=str, choices=['none','dpcr','wpro','gls'], default='none')
        parser.add_argument('--prior-mode', type=str, choices=['off','train','adapt','train_adapt'], default='off')
        parser.add_argument('--prior-ridge', type=float, default=1e-3)
        parser.add_argument('--prior-batches', type=int, default=50)
        parser.add_argument('--prior-residual-zero-init', type=int, choices=[0,1], default=1)
        parser.add_argument('--gls-target-cov', type=str, choices=['full','diag'], default='full')

        # NEW: train-stage periodic prior update
        parser.add_argument('--prior-update-interval', type=int, default=0,
                            help='Re-estimate prior every K epochs in train_backbone (0 disables).')
        parser.add_argument('--prior-update-ema', action='store_true', default=False,
                            help='Use EMA when updating prior in train_backbone (otherwise hard overwrite).')
        parser.add_argument('--prior-update-ema-m', type=float, default=0.90,
                            help='EMA momentum for prior update: prior <- m*prior + (1-m)*prior_new.')

        # NEW: drift log
        parser.add_argument('--log-drift-every', type=int, default=0,
                            help='Save drift_task{t}_epoch{e}.csv every E epochs (0 disables).')

        # Adapter refinement gate. The camera-ready best CIFAR/Tiny scripts use --no-adapt-train,
        # which skips the secondary post-hoc adapter training stage and reuses the co-evolved adapter.
        parser.add_argument('--adapt-train', dest='adapt_train', action='store_true',
                            help='Train/refine the adapter inside adapt_distributions (default).')
        parser.add_argument('--no-adapt-train', dest='adapt_train', action='store_false',
                            help='Skip post-hoc adapter refinement and reuse the co-evolved adapter for Gaussian transport.')
        parser.set_defaults(adapt_train=True)

        # --------------------  Feature Perturbation knobs --------------------
        parser.add_argument('--rho-ce', type=float, default=0.0, help='FP step size for classification loss (task 0 only).')
        parser.add_argument('--lambda-fp-ce', type=float, default=0.5, help='Weight for adversarial classification loss.')
        parser.add_argument('--fp-use-grad-norm', action='store_true', default=False,
                            help='Enable grad-norm penalty on perturbed features (second-order).')
        parser.add_argument('--lambda-fp-gn', type=float, default=0.1, help='Weight for grad-norm penalty.')
        parser.add_argument('--mixup-alpha', type=float, default=0.0, help='Accepted for CUB script compatibility; final GATF scripts keep it at 0.')
        parser.add_argument('--cutmix-alpha', type=float, default=0.0, help='Accepted for CUB script compatibility; final GATF scripts keep it at 0.')

        # --------------------  Distiller/Adapter LR/WD overrides --------------------
        parser.add_argument('--DA-lr', type=float, default=0.0, help='distiller and adapter learning rate')
        parser.add_argument('--DA-wd', type=float, default=0.0, help='distiller and adapter weight decay')

        return parser.parse_known_args(args)

    # ------------------------------ Helper losses/regs ------------------------------

    def _orthonorm_penalty(self, module: nn.Module) -> torch.Tensor:
        """Sum ||W^T W - I||_F^2 over Linear layers in `module`."""
        reg = torch.zeros((), device=self.device)
        for m in module.modules():
            if isinstance(m, nn.Linear):
                W = m.weight
                I = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
                reg = reg + torch.norm(W.t() @ W - I, p='fro') ** 2
        return reg

    def _pairwise_geom_loss(self, Z_ref: torch.Tensor, Z_hat: torch.Tensor, metric: str = "cos") -> torch.Tensor:
        """Preserve within-batch geometry between Z_ref and Z_hat."""
        if Z_ref.shape[0] > 128:
            idx = torch.randperm(Z_ref.shape[0], device=Z_ref.device)[:128]
            Z_ref, Z_hat = Z_ref[idx], Z_hat[idx]
        if metric == "cos":
            ref = F.normalize(Z_ref, dim=1) @ F.normalize(Z_ref, dim=1).t()
            hat = F.normalize(Z_hat, dim=1) @ F.normalize(Z_hat, dim=1).t()
            return F.mse_loss(hat, ref)
        else:
            ref, hat = torch.cdist(Z_ref, Z_ref, p=2), torch.cdist(Z_hat, Z_hat, p=2)
            return F.mse_loss(hat, ref)

    # ------------------------------  Feature Perturbation (FP) helpers ------------------------------

    def _fp_make_adv(self, z: torch.Tensor, loss_fn, rho: float, allow_second_order: bool = False) -> torch.Tensor:
        """
        Build adversarial features: z_adv = z + rho * ĝ, where ĝ = ∂loss/∂z / ||.||.
        - loss_fn: callable(z) -> scalar loss used to define direction
        - allow_second_order=False: if True, keep graph through the direction (expensive)
        """
        if rho <= 0:
            return None
        z_det = z.detach().requires_grad_(True)
        loss_dir = loss_fn(z_det)
        g = torch.autograd.grad(loss_dir, z_det, only_inputs=True, retain_graph=False)[0]
        g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
        z_adv = z_det + rho * g
        if not allow_second_order:
            z_adv = z_adv.detach()
        return z_adv

    def _fp_gradnorm(self, z: torch.Tensor, loss_fn) -> torch.Tensor:
        """
        Compute ||∂loss/∂z||_2 (mean over batch) on the given z (second-order w.r.t. model/criterion weights).
        We detach z from prior graph but keep model/criterion weights in graph.
        """
        z_req = z.detach().requires_grad_(True)
        loss_pert = loss_fn(z_req)
        grad = torch.autograd.grad(loss_pert, z_req, create_graph=True, only_inputs=True)[0]
        return grad.norm(dim=1).mean()

    # ------------------------------ Prior estimation helpers ------------------------------

    @staticmethod
    def _sym_sqrtm(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        cov = 0.5 * (cov + cov.t())
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=eps)
        s = torch.sqrt(evals)
        return (evecs * s.unsqueeze(0)) @ evecs.t()

    @staticmethod
    def _sym_invsqrtm(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        cov = 0.5 * (cov + cov.t())
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=eps)
        invs = 1.0 / torch.sqrt(evals)
        return (evecs * invs.unsqueeze(0)) @ evecs.t()

    @torch.no_grad()
    def _collect_pair_stats(self, loader, max_batches: int):
        """Collect mean/cov/cross stats for (z_old, z_new) on a few batches."""
        self.model.eval()
        self.old_model.eval()
        S = self.S

        n = 0
        sum_old = torch.zeros((S,), device=self.device, dtype=torch.float32)
        sum_new = torch.zeros((S,), device=self.device, dtype=torch.float32)
        sum_oo = torch.zeros((S, S), device=self.device, dtype=torch.float32)
        sum_nn = torch.zeros((S, S), device=self.device, dtype=torch.float32)
        sum_on = torch.zeros((S, S), device=self.device, dtype=torch.float32)

        bcnt = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                images = batch[0]
            else:
                images = batch
            images = images.to(self.device, non_blocking=True)

            z_new = self.model(images).float()
            z_old = self.old_model(images).float()

            bsz = z_old.shape[0]
            n += bsz
            sum_old += z_old.sum(dim=0)
            sum_new += z_new.sum(dim=0)
            sum_oo += z_old.t() @ z_old
            sum_nn += z_new.t() @ z_new
            sum_on += z_old.t() @ z_new

            bcnt += 1
            if bcnt >= max_batches:
                break

        n = max(int(n), 1)
        mu_old = sum_old / n
        mu_new = sum_new / n

        cov_old = (sum_oo / n) - (mu_old.unsqueeze(1) @ mu_old.unsqueeze(0))
        cov_new = (sum_nn / n) - (mu_new.unsqueeze(1) @ mu_new.unsqueeze(0))
        cross   = (sum_on / n) - (mu_old.unsqueeze(1) @ mu_new.unsqueeze(0))
        return mu_old, mu_new, cov_old, cov_new, cross

    @torch.no_grad()
    def _estimate_dpcr_prior(self, loader) -> dict:
        """DPCR ridge: P = (Z_old^T Z_old + λI)^-1 (Z_old^T Z_new) (uncentered)."""
        self.model.eval()
        self.old_model.eval()
        S = self.S

        cov = torch.zeros((S, S), device=self.device, dtype=torch.float32)
        cross = torch.zeros((S, S), device=self.device, dtype=torch.float32)
        max_batches = max(1, int(self.prior_batches))
        bcnt = 0
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device, non_blocking=True)
            z_new = self.model(images).float()
            z_old = self.old_model(images).float()
            cov += z_old.t() @ z_old
            cross += z_old.t() @ z_new
            bcnt += 1
            if bcnt >= max_batches:
                break

        cov = cov + float(self.prior_ridge) * torch.eye(S, device=self.device, dtype=cov.dtype)
        P = torch.linalg.solve(cov, cross)
        return {"kind": "dpcr", "P": P}

    @torch.no_grad()
    def _estimate_wpro_prior(self, loader) -> dict:
        """Whitened Procrustes: A = Σ_old^{-1/2} R Σ_new^{1/2}, plus (mu_old, mu_new)."""
        max_batches = max(1, int(self.prior_batches))
        mu_old, mu_new, cov_old, cov_new, cross = self._collect_pair_stats(loader, max_batches)

        ridge = float(self.prior_ridge)
        S = self.S
        cov_old = cov_old + ridge * torch.eye(S, device=self.device, dtype=cov_old.dtype)
        cov_new = cov_new + ridge * torch.eye(S, device=self.device, dtype=cov_new.dtype)

        W_old = self._sym_invsqrtm(cov_old, eps=1e-6)
        W_new = self._sym_invsqrtm(cov_new, eps=1e-6)
        S_new = self._sym_sqrtm(cov_new, eps=1e-6)

        M = W_old @ cross @ W_new
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh

        if torch.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vh

        A = W_old @ R @ S_new
        return {"kind": "wpro", "mu_old": mu_old, "mu_new": mu_new, "A": A}

    @torch.no_grad()
    def _estimate_gls_prior(self, loader) -> dict:
        """
        GLS ridge (Mahalanobis-aligned):
          minimize ||(X P + b - Y)||_{Σ_y^{-1}}^2 + λ||P||_F^2
        Using centered stats:
          X = z_old - mu_old, Y = z_new - mu_new
          Σ_y from cov_new, whiten target: Yw = Y Σ_y^{-1/2}
          solve P_w = (cov_x + λI)^-1 (cross Σ_y^{-1/2})
          P = P_w Σ_y^{1/2}, b = mu_new - mu_old @ P
        """
        max_batches = max(1, int(self.prior_batches))
        mu_old, mu_new, cov_old, cov_new, cross = self._collect_pair_stats(loader, max_batches)

        ridge = float(self.prior_ridge)
        S = self.S
        I = torch.eye(S, device=self.device, dtype=cov_old.dtype)

        if self.gls_target_cov == "diag":
            cov_new = torch.diag(torch.diag(cov_new))
        cov_new = cov_new + ridge * I

        cov_old = cov_old + ridge * I

        Sy_inv_sqrt = self._sym_invsqrtm(cov_new, eps=1e-6)
        Sy_sqrt = self._sym_sqrtm(cov_new, eps=1e-6)

        RHS = cross @ Sy_inv_sqrt
        P_w = torch.linalg.solve(cov_old, RHS)
        P = P_w @ Sy_sqrt

        b = mu_new - (mu_old @ P)
        return {"kind": "gls", "P": P, "b": b, "target_cov": self.gls_target_cov}

    def _wrap_adapter_with_prior(self, prior_info: dict, residual: nn.Module) -> nn.Module:
        kind = prior_info["kind"]
        if kind == "dpcr":
            return RidgePriorAdapter(P=prior_info["P"],
                                     bias=None,
                                     residual=residual,
                                     zero_init_residual=self.prior_residual_zero_init)
        if kind == "wpro":
            return WhitenedProcrustesPriorAdapter(mu_old=prior_info["mu_old"],
                                                  mu_new=prior_info["mu_new"],
                                                  A=prior_info["A"],
                                                  residual=residual,
                                                  zero_init_residual=self.prior_residual_zero_init)
        if kind == "gls":
            return RidgePriorAdapter(P=prior_info["P"],
                                     bias=prior_info["b"],
                                     residual=residual,
                                     zero_init_residual=self.prior_residual_zero_init)
        raise ValueError(f"Unknown prior kind: {kind}")

    def _estimate_prior(self, loader) -> dict:
        if self.prior_kind == "dpcr":
            return self._estimate_dpcr_prior(loader)
        if self.prior_kind == "wpro":
            return self._estimate_wpro_prior(loader)
        if self.prior_kind == "gls":
            return self._estimate_gls_prior(loader)
        return {"kind": "none"}

    # ------------------------------ NEW: in-place prior update (hard / EMA) ------------------------------

    @torch.no_grad()
    def _update_prior_inplace(self, adapter: nn.Module, prior_info: dict, use_ema: bool = False, m: float = 0.90):
        """
        Update the *prior part* of a wrapped adapter IN PLACE.

        - If adapter is RidgePriorAdapter: update prior.weight (and bias if exists)
        - If adapter is WhitenedProcrustesPriorAdapter: update buffers mu_old/mu_new/A

        If use_ema=True:
            param <- m * param + (1-m) * new_param

        Also refresh self._prior_cache to reflect the *current* prior used by adapter.
        """
        if adapter is None:
            return
        kind = prior_info.get("kind", "none")

        if isinstance(adapter, RidgePriorAdapter):
            if "P" not in prior_info:
                return
            P_new = prior_info["P"].detach()
            W_new = P_new.t().contiguous()

            if not use_ema:
                adapter.prior.weight.copy_(W_new)
            else:
                adapter.prior.weight.mul_(m).add_(W_new, alpha=(1.0 - m))

            if adapter.prior.bias is not None:
                b_new = prior_info.get("b", None)
                if b_new is None:
                    b_new = torch.zeros_like(adapter.prior.bias)
                else:
                    b_new = b_new.detach().contiguous()
                if not use_ema:
                    adapter.prior.bias.copy_(b_new)
                else:
                    adapter.prior.bias.mul_(m).add_(b_new, alpha=(1.0 - m))

            P_cur = adapter.prior.weight.detach().t().contiguous().clone()
            cache = {"kind": kind, "P": P_cur}
            if adapter.prior.bias is not None:
                cache["b"] = adapter.prior.bias.detach().clone()
            if kind == "gls":
                cache["target_cov"] = prior_info.get("target_cov", self.gls_target_cov)
            self._prior_cache = cache
            return

        if isinstance(adapter, WhitenedProcrustesPriorAdapter):
            if ("mu_old" not in prior_info) or ("mu_new" not in prior_info) or ("A" not in prior_info):
                return
            mu_old_new = prior_info["mu_old"].detach().to(adapter.mu_old.dtype)
            mu_new_new = prior_info["mu_new"].detach().to(adapter.mu_new.dtype)
            A_new = prior_info["A"].detach().to(adapter.A.dtype)

            if not use_ema:
                adapter.mu_old.copy_(mu_old_new)
                adapter.mu_new.copy_(mu_new_new)
                adapter.A.copy_(A_new)
            else:
                adapter.mu_old.mul_(m).add_(mu_old_new, alpha=(1.0 - m))
                adapter.mu_new.mul_(m).add_(mu_new_new, alpha=(1.0 - m))
                adapter.A.mul_(m).add_(A_new, alpha=(1.0 - m))

            self._prior_cache = {
                "kind": kind,
                "mu_old": adapter.mu_old.detach().clone(),
                "mu_new": adapter.mu_new.detach().clone(),
                "A": adapter.A.detach().clone(),
            }
            return

        return

    # ------------------------------ Drift weights ------------------------------

    @torch.no_grad()
    def _compute_class_drift_weights(self, adapter: nn.Module, t: int) -> torch.Tensor:
        if t == 0: return torch.ones((0,), device=self.device)
        C_old = self.task_offset[t]
        if C_old == 0: return torch.ones((0,), device=self.device)
        mu_old = self.means[:C_old].detach()
        mu_new_hat = adapter(mu_old).detach() if adapter is not None else mu_old
        d = torch.norm(mu_new_hat - mu_old, p=2, dim=1)
        d_norm = d / (d.max() + self._drift_eps)
        w = self.drift_scale * torch.sigmoid(d_norm)
        if self._drift_weights_ema is None or self._drift_weights_ema.numel() != C_old:
            self._drift_weights_ema = w.clone()
        else:
            m = float(self.drift_momentum)
            self._drift_weights_ema = m * self._drift_weights_ema + (1 - m) * w
        return self._drift_weights_ema

    # ------------------------------ NEW: Drift snapshot (CSV) ------------------------------

    def _get_exp_dir(self) -> str:
        exp_dir = getattr(self.logger, "exp_path", ".")
        try:
            os.makedirs(exp_dir, exist_ok=True)
        except Exception:
            pass
        return exp_dir

    def _get_old_val_loaders_for_drift(self, t: int):
        loaders = []
        if self.all_val_loader is not None:
            try:
                loaders = [dl for i, dl in enumerate(self.all_val_loader) if (dl is not None) and (i < t)]
            except Exception:
                loaders = []
        if len(loaders) == 0:
            loaders = self.val_data_loaders[:t]
        return loaders

    @torch.no_grad()
    def _snapshot_drift_metrics(self, t: int, epoch: int):
        """
        Save drift CSV:
          <exp_path>/drift_task{t}_epoch{epoch}.csv

        Metric:
          mu_l2 = || mean_current_model(old_val_data,class=c) - stored_mean(c) ||_2

        Notes:
          - Uses old validation loaders only (tasks < t).
          - Uses flip augmentation (same as prototype creation).
        """
        if t <= 0:
            return
        C_old = self.task_offset[t]
        if C_old <= 0:
            return

        loaders = self._get_old_val_loaders_for_drift(t)
        if loaders is None or len(loaders) == 0:
            return

        S = self.S
        sum_feats = torch.zeros((C_old, S), device=self.device, dtype=torch.float32)
        counts = torch.zeros((C_old,), device=self.device, dtype=torch.float32)

        was_training = self.model.training
        self.model.eval()

        for dl in loaders:
            # rebuild loader to ensure drop_last=False
            try:
                bs = int(getattr(dl, "batch_size", 128) or 128)
            except Exception:
                bs = 128
            try:
                nw = int(getattr(dl, "num_workers", 0) or 0)
            except Exception:
                nw = 0

            dl2 = torch.utils.data.DataLoader(
                dl.dataset, batch_size=bs, shuffle=False, num_workers=nw, drop_last=False
            )

            for batch in dl2:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, targets = batch[0], batch[1]
                else:
                    # cannot compute drift without labels
                    continue

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).long()

                feats = self.model(images).float()
                # flip augmentation (consistent with create_distributions)
                if images.dim() == 4 and images.size(-1) > 1:
                    feats2 = self.model(torch.flip(images, dims=(3,))).float()
                else:
                    feats2 = feats

                feats_all = torch.cat([feats, feats2], dim=0)
                tgt_all = torch.cat([targets, targets], dim=0)

                mask = (tgt_all >= 0) & (tgt_all < C_old)
                if not torch.any(mask):
                    continue
                idx = tgt_all[mask]
                sum_feats.index_add_(0, idx, feats_all[mask])
                ones = torch.ones((idx.shape[0],), device=self.device, dtype=counts.dtype)
                counts.index_add_(0, idx, ones)

        if was_training:
            self.model.train()

        valid = counts > 0
        if not torch.any(valid):
            return

        means_obs = sum_feats[valid] / counts[valid].unsqueeze(1)
        means_ref = self.means[:C_old][valid].detach().float()

        mu_err = torch.norm(means_obs - means_ref, dim=1).detach().cpu().numpy()
        cls_idx = torch.nonzero(valid, as_tuple=False).squeeze(1).detach().cpu().numpy()
        cnts = counts[valid].detach().cpu().numpy().astype(int)

        exp_dir = self._get_exp_dir()
        out_path = os.path.join(exp_dir, f"drift_task{t}_epoch{epoch}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "mu_l2", "count"])
            for k, v, c in zip(cls_idx, mu_err, cnts):
                writer.writerow([int(k), float(v), int(c)])

    # ------------------------------ Geometry-anchored transport loss ------------------------------

    def geometry_anchored_transport_loss(self, t, base_loss, features, distiller, adapter, images):
        if t == 0: return base_loss, torch.zeros((), device=self.device)
        with torch.no_grad():
            z_old = self.old_model(images)
        z_new = features
        z_old_hat = distiller(z_new)
        z_new_hat = adapter(z_old)

        mse_old = F.mse_loss(z_old_hat, z_old, reduction='none').mean(dim=1)
        mse_new = F.mse_loss(z_new_hat, z_new.detach(), reduction='none').mean(dim=1)
        cyc_new = F.mse_loss(adapter(z_old_hat.detach()), z_new.detach(), reduction='none').mean(dim=1)
        cyc_old = F.mse_loss(distiller(z_new_hat.detach()), z_old.detach(), reduction='none').mean(dim=1)

        L = (self.lambda_geometry * (mse_old.mean() + mse_new.mean())
             + self.lambda_transport_cycle * (cyc_new.mean() + cyc_old.mean()))
        if self.lambda_isometry > 0:
            L += self.lambda_isometry * (self._orthonorm_penalty(distiller) + self._orthonorm_penalty(adapter))
        if self.preserve_pairwise_geometry and self.lambda_pairwise_geometry > 0:
            L += self.lambda_pairwise_geometry * (self._pairwise_geom_loss(z_new.detach(), z_new_hat)
                                     + self._pairwise_geom_loss(z_old.detach(), z_old_hat))
        return base_loss + L, L

    # ------------------------------ KD logits helper ------------------------------

    def _weighted_kd_logits(self, student_logits, teacher_logits, class_weights=None, T=2.0,
                            eps=1e-8, sample_weights=None, reduce=True):
        log_q = torch.log_softmax(student_logits / max(T, 1e-6), dim=1)
        p = torch.softmax(teacher_logits / max(T, 1e-6), dim=1)
        if (class_weights is not None) and (class_weights.numel() == p.shape[1]):
            w = class_weights.view(1, -1).expand_as(p)
            per_sample = - (w * p * log_q).sum(dim=1)
        else:
            per_sample = - (p * log_q).sum(dim=1)
        if sample_weights is not None: per_sample = per_sample * sample_weights
        return per_sample.mean() if reduce else per_sample

    # ------------------------------ EMA utils ------------------------------

    def _reset_or_init_head_ema(self, head: nn.Module):
        need_reset = False
        if self.head_ema is None:
            need_reset = True
        else:
            s_ema = [p.shape for p in self.head_ema.parameters()]
            s_cur = [p.shape for p in head.parameters()]
            if s_ema != s_cur:
                need_reset = True
        if need_reset:
            self.head_ema = copy.deepcopy(head).to(self.device).eval()
            for p in self.head_ema.parameters(): p.requires_grad = False
            print("[EMA] Re-initialized EMA head for the current task head.")

    @torch.no_grad()
    def _update_head_ema(self, head: nn.Module):
        if self.head_ema is None: return
        m = float(self.new_head_ema_m)
        for p_ema, p in zip(self.head_ema.parameters(), head.parameters()):
            p_ema.data = m * p_ema.data + (1 - m) * p.data

    def _conflict_gate(self, p_old_max: torch.Tensor, p_new_max: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.kd_conflict_gamma * (p_old_max - p_new_max))
        return self.kd_conflict_wmin + (self.kd_conflict_wmax - self.kd_conflict_wmin) * gate

    # ----------------------------------------- Main loop -----------------------------------------

    def train_loop(self, t, trn_loader, val_loader, all_val_loader=None):
        # NEW: keep for drift snapshots
        self.all_val_loader = all_val_loader

        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.append(trn_loader)
        self.val_data_loaders.append(val_loader)
        self.old_model = copy.deepcopy(self.model); self.old_model.eval()
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])

        print("### Training backbone ###")
        self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)

        # ------------------------------ DUMP (task-level) ------------------------------
        if self.dump:
            exp_dir = self._get_exp_dir()
            torch.save(self.model.state_dict(), f"{exp_dir}/model_{t}.pth")
            if self.distiller_module is not None:
                torch.save(self.distiller_module.state_dict(), f"{exp_dir}/distiller_{t}.pth")
            if self.adapter_module is not None:
                torch.save(self.adapter_module.state_dict(), f"{exp_dir}/adapter_before_{t}.pth")
            if self._prior_cache is not None:
                torch.save(self._prior_cache, f"{exp_dir}/prior_{t}.pth")

        if t > 0 and self.adaptation_strategy != "none":
            print("### Adapting prototypes ###")
            self.adapt_distributions(t, trn_loader, val_loader)

        print("### Creating new prototypes ###\n")
        self.create_distributions(t, trn_loader, val_loader, num_classes_in_t)

        # Precompute inverted covariances (for Bayes head)
        covs = self.covs.clone()
        print(f"Cov matrix det (blockwise): {[float(torch.linalg.det(c)) for c in covs]}")
        for i in range(covs.shape[0]):
            print(f"Rank for class {i}: raw={torch.linalg.matrix_rank(self.covs_raw[i], tol=0.01)}, "
                  f"shrunk={torch.linalg.matrix_rank(self.covs[i], tol=0.01)}")
            covs[i] = self.shrink_cov(covs[i], 3)
        if self.is_normalization: covs = self.norm_cov(covs)
        self.covs_inverted = torch.linalg.inv(covs)

        # ------------------------------ DUMP (prototypes) ------------------------------
        if self.dump:
            exp_dir = self._get_exp_dir()
            torch.save(self.covs_inverted, f"{exp_dir}/covs_inv_{t}.pth")
            torch.save(self.means, f"{exp_dir}/means_{t}.pth")
            torch.save(self.covs, f"{exp_dir}/covs_{t}.pth")
            torch.save(self.covs_raw, f"{exp_dir}/covs_raw_{t}.pth")

        if self.classifier == "linear": self.train_linear_head(t)
        self.check_singular_values(t, val_loader)
        self.print_singular_values()
        self.print_covs(trn_loader, val_loader)
        self.print_mahalanobis(t)

    def _build_projection(self, kind: str) -> nn.Module:
        if kind == "mlp":
            return nn.Sequential(
                nn.Linear(self.S, self.multiplier * self.S),
                nn.GELU(),
                nn.Linear(self.multiplier * self.S, self.S),
            )
        return nn.Linear(self.S, self.S)

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size,
                                                 num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size,
                                                 num_workers=val_loader.num_workers, shuffle=False, drop_last=True)

        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,} shared parameters\n')

        distiller = self._build_projection(self.distiller_type).to(self.device, non_blocking=True)  # new->old

        # ---- adapter build (possibly PRIOR in TRAIN stage) ----
        adapter_residual = self._build_projection(self.adapter_type).to(self.device, non_blocking=True)
        adapter = adapter_residual

        use_train_prior = (t > 0 and self.prior_kind != "none" and self.prior_mode in ("train", "train_adapt"))

        if use_train_prior:
            prior_info = self._estimate_prior(trn_loader)
            self._prior_cache = copy.deepcopy(prior_info)
            adapter = self._wrap_adapter_with_prior(prior_info, adapter_residual).to(self.device, non_blocking=True)
            print(f"[PRIOR/TRAIN] kind={self.prior_kind} ridge={self.prior_ridge:g} batches={self.prior_batches} "
                  f"zero_init_res={self.prior_residual_zero_init} "
                  f"{'(gls_target_cov='+self.gls_target_cov+')' if self.prior_kind=='gls' else ''}")

        criterion = self.criterion(num_classes_in_t, self.S, self.device, smoothing=self.smoothing)

        # Optional PASS-style 4-way rotation in task 0
        if t == 0 and self.is_rotation:
            criterion = self.criterion(4 * num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size // 4,
                                                     num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size // 4,
                                                     num_workers=val_loader.num_workers, shuffle=False, drop_last=True)

        self.heads.eval()
        old_heads = copy.deepcopy(self.heads)

        if self.use_new_head_ema_kd and self.criterion_type == "ce" and hasattr(criterion, "head"):
            self._reset_or_init_head_ema(criterion.head)

        # ----------------  Optimizer setup with DA-lr / DA-wd ----------------
        parameters = list(self.model.parameters()) + list(criterion.parameters()) \
                   + list(distiller.parameters()) + list(adapter.parameters()) \
                   + list(self.heads.parameters())

        if self.pretrained:
            parameters_dict = [
                {"params": list(self.model.parameters())[:-1], "lr": self.lr_backbone},
                {"params": list(criterion.parameters()) + list(self.model.parameters())[-1:]},
                {"params": list(distiller.parameters()), "lr": self.DA_lr, "weight_decay": self.DA_wd},
                {"params": list(adapter.parameters()),   "lr": self.DA_lr, "weight_decay": self.DA_wd},
                {"params": list(self.heads.parameters())},
            ]
        else:
            # Keep legacy behavior for backbone/head: use global lr (self.lr) unless user explicitly turns on DA groups.
            parameters_dict = [
                {"params": list(self.model.parameters())},
                {"params": list(criterion.parameters())},
                {"params": list(distiller.parameters()), "lr": self.DA_lr, "weight_decay": self.DA_wd},
                {"params": list(adapter.parameters()),   "lr": self.DA_lr, "weight_decay": self.DA_wd},
                {"params": list(self.heads.parameters())},
            ]

        # Use param groups when needed (pretrained or DA overrides explicitly set); otherwise keep original "all params same lr/wd" path.
        use_param_groups = self.pretrained or (self.DA_lr_user > 0) or (self.DA_wd_user > 0)
        optimizer, lr_scheduler = self.get_optimizer(parameters_dict if use_param_groups else parameters, t, self.wd)

        for epoch in range(self.nepochs):
            # ---------------- NEW: periodic prior re-estimation + in-place update (TRAIN stage) ----------------
            if use_train_prior and (self.prior_update_interval > 0) and (epoch > 0) and (epoch % self.prior_update_interval == 0):
                prior_new = self._estimate_prior(trn_loader)
                self._update_prior_inplace(adapter, prior_new, use_ema=self.prior_update_ema, m=self.prior_update_ema_m)
                tag = "EMA" if self.prior_update_ema else "HARD"
                extra = f" m={self.prior_update_ema_m:.3f}" if self.prior_update_ema else ""
                print(f"[PRIOR/TRAIN][{tag}] epoch={epoch} interval={self.prior_update_interval}{extra} kind={self.prior_kind}")

            train_loss, valid_loss = [], []
            train_kd_feat_loss, valid_kd_feat_loss = [], []
            train_kd_logits_loss, valid_kd_logits_loss = [], []
            train_proto_repel_loss, valid_proto_repel_loss = [], []

            train_ac, train_determinant = [], []
            train_hits, val_hits, train_total, val_total = 0, 0, 0, 0

            g_conf_sum = g_proto_sum = g_floor_sum = g_final_sum = 0.0
            g_count = 0

            conf_count = strong_conf_count = gate_on_conf_count = 0
            raw_mass = supp_mass = 0.0
            conf_gap_sum = conf_gate_sum = 0.0
            sample_total = 0

            kd_ramp = min(1.0, (epoch + 1) / max(1, int(self.kd_rampup_fraction * self.nepochs)))
            tr_ramp = min(1.0, (epoch + 1) / max(1, int(self.transfer_backbone_rampup_fraction * self.nepochs)))
            proto_ramp = min(1.0, (epoch + 1) / max(1, int(self.proto_contrast_rampup_fraction * self.nepochs)))

            self.model.train(); criterion.train(); distiller.train(); adapter.train()

            proj_old_protos = None
            if t > 0 and (self.proto_contrast or self.kd_gate_mode in ('proto','hybrid')):
                with torch.no_grad():
                    C_old = self.task_offset[t]
                    if C_old > 0:
                        _adapter = (self.adapter_module or adapter)
                        mu_old = self.means[:C_old]
                        proj_old_protos = _adapter(mu_old).detach()

            old_covs_inv = None
            if t > 0 and self.proto_contrast and self.proto_contrast_metric == 'maha':
                with torch.no_grad():
                    C_old = self.task_offset[t]
                    if C_old > 0:
                        covs_old = self.covs[:C_old].clone()
                        for i in range(C_old):
                            covs_old[i] = self.shrink_cov(covs_old[i], self.shrink)
                        old_covs_inv = torch.linalg.inv(covs_old).detach()

            for images, targets in trn_loader:
                if t == 0 and self.is_rotation:
                    images, targets = compute_rotations(images, targets, num_classes_in_t)
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                train_total += bsz
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                features = self.model(images)
                if epoch < int(self.nepochs * 0.01):
                    features = features.detach()

                loss, logits = criterion(features, targets)

                kd_feat_loss = torch.zeros((), device=self.device)
                if self.distillation == "logit":
                    total_loss, kd_feat_loss = self.distill_logits(t=0, loss=loss, features=features, images=images, old_heads=old_heads)
                elif self.distillation == "projected":
                    total_loss, kd_feat_loss = self.distill_projected(t, loss, features, distiller, images)
                elif self.distillation == "gatf":
                    total_loss, kd_feat_loss = self.geometry_anchored_transport_loss(t, loss, features, distiller, adapter, images)
                elif self.distillation == "feature":
                    total_loss, kd_feat_loss = self.distill_features(t, loss, features, images)
                else:
                    total_loss = loss

                # ----------------  (A) FP on classification loss (task 0 only) ----------------
                if (t == 0) and (self.rho_ce > 0):
                    def _cls_loss_fn(z_):
                        l_, _ = criterion(z_, targets)
                        return l_

                    z_adv_ce = self._fp_make_adv(features, _cls_loss_fn, rho=self.rho_ce, allow_second_order=False)
                    if z_adv_ce is not None:
                        loss_adv_ce, _ = criterion(z_adv_ce, targets)
                        total_loss = total_loss + self.lambda_fp_ce * loss_adv_ce

                        if self.fp_use_grad_norm and (self.lambda_fp_gn > 0):
                            gn_ce = self._fp_gradnorm(z_adv_ce, _cls_loss_fn)
                            total_loss = total_loss + self.lambda_fp_gn * gn_ce

                # Aux KD on OLD classes (unchanged)
                if t > 0 and self.use_aux_logit_kd and len(self.heads) > 0 and self.criterion_type == "ce":
                    with torch.no_grad():
                        z_old = self.old_model(images)
                        teacher_logits_old = torch.cat([head(z_old) for head in self.heads], dim=1)
                    student_logits_old = torch.cat([head(features) for head in self.heads], dim=1)
                    class_w_nom = self._compute_class_drift_weights(self.adapter_module or adapter, t)

                    C_old_logits = teacher_logits_old.shape[1]
                    class_w = class_w_nom
                    if class_w.numel() != C_old_logits:
                        if C_old_logits % class_w.numel() == 0:
                            rep = C_old_logits // class_w.numel()
                            class_w = class_w.repeat_interleave(rep)
                        else:
                            reps = (C_old_logits + class_w.numel() - 1) // class_w.numel()
                            class_w = class_w.repeat(reps)[:C_old_logits]

                    sample_gate = None
                    g_conf = None
                    g_proto = None
                    g_floor = None
                    p_old_max_debug, p_new_max_debug = None, None

                    if self.kd_gate_mode in ('conflict','proto','hybrid','conflict_only'):
                        with torch.no_grad():
                            p_old = torch.softmax(teacher_logits_old / self.tau_aux, dim=1)
                            p_old_max, old_idx = p_old.max(dim=1)
                            p_old_max_debug = p_old_max
                            drift_samp = class_w[old_idx] if class_w.numel() > 0 else torch.zeros_like(p_old_max)

                            if self.kd_gate_mode in ('conflict','hybrid','conflict_only') and logits is not None:
                                p_new = torch.softmax(logits / self.tau_aux, dim=1)
                                p_new_max, _ = p_new.max(dim=1)
                                p_new_max_debug = p_new_max
                                g_conf = torch.sigmoid(self.kd_conflict_gamma * (p_old_max - p_new_max))

                            if self.kd_gate_mode in ('proto','hybrid'):
                                Pp = proj_old_protos
                                if Pp is None:
                                    C_old = self.task_offset[t]
                                    if C_old > 0:
                                        Pp = (self.adapter_module or adapter)(self.means[:C_old]).detach()
                                if Pp is not None and Pp.numel() > 0:
                                    z_n = F.normalize(features, dim=1)
                                    p_n = F.normalize(Pp, dim=1)
                                    sims = z_n @ p_n.t()
                                    s_old_max, _ = sims.max(dim=1)
                                    g_proto = torch.sigmoid(self.kd_proto_gamma * (s_old_max - self.kd_proto_thresh))

                            if self.kd_gate_mode != 'conflict_only':
                                g_floor = self.kd_floor_min + (self.kd_floor_max - self.kd_floor_min) * drift_samp

                            if self.kd_gate_mode == 'conflict_only':
                                sample_gate = g_conf
                            else:
                                g_candidates = [g for g in (g_conf, g_proto) if g is not None]
                                if len(g_candidates) > 0:
                                    g_comb = g_candidates[0]
                                    for gg in g_candidates[1:]:
                                        g_comb = torch.maximum(g_comb, gg)
                                    sample_gate = g_comb
                                    if g_floor is not None:
                                        sample_gate = torch.maximum(sample_gate, g_floor)

                    elif self.kd_conflict_aware and logits is not None:
                        with torch.no_grad():
                            p_old = torch.softmax(teacher_logits_old / self.tau_aux, dim=1)
                            p_old_max, _ = p_old.max(dim=1)
                            p_old_max_debug = p_old_max
                            p_new = torch.softmax(logits / self.tau_aux, dim=1)
                            p_new_max, _ = p_new.max(dim=1)
                            p_new_max_debug = p_new_max
                            g_conf = self._conflict_gate(p_old_max, p_new_max)
                            sample_gate = g_conf.detach()

                    kd_logits_loss = self._weighted_kd_logits(
                        student_logits_old, teacher_logits_old, class_w, T=self.tau_aux,
                        sample_weights=sample_gate
                    )
                    total_loss = total_loss + (self.lambda_aux_logit_kd * kd_ramp) * kd_logits_loss
                    train_kd_logits_loss.append(float(bsz * float(kd_logits_loss)))

                    with torch.no_grad():
                        if g_conf is not None: g_conf_sum += float(g_conf.mean()) * bsz
                        if g_proto is not None: g_proto_sum += float(g_proto.mean()) * bsz
                        if g_floor is not None: g_floor_sum += float(g_floor.mean()) * bsz
                        if sample_gate is not None:
                            g_final_sum += float(sample_gate.mean()) * bsz
                            g_count += bsz

                    if self.print_gate_diag and (p_old_max_debug is not None) and (p_new_max_debug is not None):
                        kd_raw_vec = self._weighted_kd_logits(
                            student_logits_old, teacher_logits_old, class_w, T=self.tau_aux,
                            sample_weights=None, reduce=False
                        ).detach()
                        kd_gated_vec = self._weighted_kd_logits(
                            student_logits_old, teacher_logits_old, class_w, T=self.tau_aux,
                            sample_weights=sample_gate, reduce=False
                        ).detach()

                        conf_mask = (p_old_max_debug > p_new_max_debug)
                        strong_conf_mask = (p_old_max_debug - p_new_max_debug > self._gate_diag_delta)

                        gate_on_conf_mask = None
                        if sample_gate is not None:
                            gate_on_conf_mask = conf_mask & (sample_gate < self._gate_applied_thresh)

                        conf_count += int(conf_mask.sum().item())
                        strong_conf_count += int(strong_conf_mask.sum().item())
                        if gate_on_conf_mask is not None:
                            gate_on_conf_count += int(gate_on_conf_mask.sum().item())

                        raw_mass += float(kd_raw_vec[conf_mask].sum().item())
                        supp_mass += float((kd_raw_vec[conf_mask] - kd_gated_vec[conf_mask]).sum().item())
                        conf_gap_sum += float((p_old_max_debug - p_new_max_debug)[conf_mask].sum().item())
                        if sample_gate is not None:
                            conf_gate_sum += float(sample_gate[conf_mask].sum().item())
                        sample_total += bsz

                # Prototype Contrast / Repel
                if t > 0 and self.proto_contrast:
                    metric_pc = self.proto_contrast_metric
                    if metric_pc == 'maha':
                        C_old = self.task_offset[t]
                        if C_old > 0 and old_covs_inv is not None:
                            proto_loss = self._proto_repel_loss(
                                z=features,
                                proj_old_protos=None,
                                margin=self.proto_contrast_margin,
                                topk=self.proto_contrast_topk,
                                metric='maha',
                                distiller=distiller,
                                means_old=self.means[:C_old].detach(),
                                covs_old_inv=old_covs_inv
                            )
                            total_loss = total_loss + (self.proto_contrast_weight * proto_ramp) * proto_loss
                            train_proto_repel_loss.append(float(bsz * float(proto_loss)))
                    else:
                        if proj_old_protos is not None:
                            proto_loss = self._proto_repel_loss(
                                z=features,
                                proj_old_protos=proj_old_protos,
                                margin=self.proto_contrast_margin,
                                topk=self.proto_contrast_topk,
                                metric=metric_pc
                            )
                            total_loss = total_loss + (self.proto_contrast_weight * proto_ramp) * proto_loss
                            train_proto_repel_loss.append(float(bsz * float(proto_loss)))

                # Optional backbone-side transfer loss
                if t > 0 and self.use_transfer_backbone:
                    with torch.no_grad():
                        z_old = self.old_model(images)
                        target_transfer = (self.adapter_module or adapter)(z_old).detach()
                    transfer_loss = F.mse_loss(features, target_transfer)
                    total_loss = total_loss + (self.lambda_transfer_backbone * tr_ramp) * transfer_loss

                # Anti-collapse regularizer
                ac, det = 0.0, torch.tensor(0.0, device=self.device)
                if self.alpha > 0:
                    ac, det = loss_ac(features, self.beta)
                    total_loss += self.alpha * ac

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                if self.use_new_head_ema_kd and hasattr(criterion, "head"):
                    self._update_head_ema(criterion.head)

                if logits is not None:
                    train_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
                train_ac.append(float(ac))
                train_determinant.append(float(torch.clamp(torch.abs(det), max=1e8)))
                train_kd_feat_loss.append(float(bsz * float(kd_feat_loss)))

            lr_scheduler.step()

            if epoch % 10 == 9:
                self.model.eval(); criterion.eval(); distiller.eval(); adapter.eval()
                with torch.no_grad():
                    for images, targets in val_loader:
                        if t == 0 and self.is_rotation:
                            images, targets = compute_rotations(images, targets, num_classes_in_t)
                        targets -= self.task_offset[t]
                        bsz = images.shape[0]; val_total += bsz
                        images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                        features = self.model(images)
                        loss, logits = criterion(features, targets)

                        if self.distillation == "logit":
                            _, kd_feat_loss = self.distill_logits(t=0, loss=loss, features=features, images=images, old_heads=old_heads)
                        elif self.distillation == "projected":
                            _, kd_feat_loss = self.distill_projected(t, loss, features, distiller, images)
                        elif self.distillation == "gatf":
                            _, kd_feat_loss = self.geometry_anchored_transport_loss(t, loss, features, distiller, adapter, images)
                        elif self.distillation == "feature":
                            _, kd_feat_loss = self.distill_features(t, loss, features, images)
                        else:
                            kd_feat_loss = torch.zeros((), device=self.device)

                        kd_logits_loss_val = torch.zeros((), device=self.device)
                        if t > 0 and self.use_aux_logit_kd and len(self.heads) > 0 and self.criterion_type == "ce":
                            z_old = self.old_model(images)
                            teacher_logits_old = torch.cat([head(z_old) for head in self.heads], dim=1)
                            student_logits_old = torch.cat([head(features) for head in self.heads], dim=1)
                            class_w_nom = self._compute_class_drift_weights(self.adapter_module or adapter, t)
                            C_old_logits = teacher_logits_old.shape[1]
                            class_w = class_w_nom
                            if class_w.numel() != C_old_logits:
                                if C_old_logits % class_w.numel() == 0:
                                    rep = C_old_logits // class_w.numel()
                                    class_w = class_w.repeat_interleave(rep)
                                else:
                                    reps = (C_old_logits + class_w.numel() - 1) // class_w.numel()
                                    class_w = class_w.repeat(reps)[:C_old_logits]
                            sample_gate = None
                            if self.kd_gate_mode in ('conflict','proto','hybrid','conflict_only'):
                                p_old = torch.softmax(teacher_logits_old / self.tau_aux, dim=1)
                                p_old_max, old_idx = p_old.max(dim=1)
                                drift_samp = class_w[old_idx] if class_w.numel() > 0 else torch.zeros_like(p_old_max)
                                g_conf = None
                                g_proto = None
                                if self.kd_gate_mode in ('conflict','hybrid','conflict_only') and logits is not None:
                                    p_new = torch.softmax(logits / self.tau_aux, dim=1)
                                    p_new_max, _ = p_new.max(dim=1)
                                    g_conf = torch.sigmoid(self.kd_conflict_gamma * (p_old_max - p_new_max))
                                if self.kd_gate_mode in ('proto','hybrid'):
                                    Pp = proj_old_protos
                                    if Pp is None:
                                        C_old = self.task_offset[t]
                                        if C_old > 0:
                                            Pp = (self.adapter_module or adapter)(self.means[:C_old]).detach()
                                    if Pp is not None and Pp.numel() > 0:
                                        z_n = F.normalize(features, dim=1)
                                        p_n = F.normalize(Pp, dim=1)
                                        sims = z_n @ p_n.t()
                                        s_old_max, _ = sims.max(dim=1)
                                        g_proto = torch.sigmoid(self.kd_proto_gamma * (s_old_max - self.kd_proto_thresh))
                                if self.kd_gate_mode == 'conflict_only':
                                    sample_gate = g_conf
                                else:
                                    g_candidates = [g for g in (g_conf, g_proto) if g is not None]
                                    if len(g_candidates) > 0:
                                        g_comb = g_candidates[0]
                                        for gg in g_candidates[1:]:
                                            g_comb = torch.maximum(g_comb, gg)
                                        g_floor = self.kd_floor_min + (self.kd_floor_max - self.kd_floor_min) * drift_samp
                                        sample_gate = torch.maximum(g_comb, g_floor)
                            elif self.kd_conflict_aware and logits is not None:
                                p_old = torch.softmax(teacher_logits_old / self.tau_aux, dim=1)
                                p_old_max, _ = p_old.max(dim=1)
                                p_new = torch.softmax(logits / self.tau_aux, dim=1)
                                p_new_max, _ = p_new.max(dim=1)
                                sample_gate = self._conflict_gate(p_old_max, p_new_max)
                            kd_logits_loss_val = self._weighted_kd_logits(
                                student_logits_old, teacher_logits_old, class_w, T=self.tau_aux,
                                sample_weights=sample_gate
                            )

                        proto_loss_val = torch.zeros((), device=self.device)
                        if t > 0 and self.proto_contrast:
                            if self.proto_contrast_metric == 'maha':
                                C_old = self.task_offset[t]
                                if C_old > 0 and old_covs_inv is not None:
                                    proto_loss_val = self._proto_repel_loss(
                                        z=features,
                                        proj_old_protos=None,
                                        margin=self.proto_contrast_margin,
                                        topk=self.proto_contrast_topk,
                                        metric='maha',
                                        distiller=distiller,
                                        means_old=self.means[:C_old].detach(),
                                        covs_old_inv=old_covs_inv
                                    )
                            else:
                                if proj_old_protos is not None:
                                    proto_loss_val = self._proto_repel_loss(
                                        z=features,
                                        proj_old_protos=proj_old_protos,
                                        margin=self.proto_contrast_margin,
                                        topk=self.proto_contrast_topk,
                                        metric=self.proto_contrast_metric
                                    )

                        if logits is not None:
                            val_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                        valid_loss.append(float(bsz * loss))
                        valid_kd_feat_loss.append(float(bsz * float(kd_feat_loss)))
                        valid_kd_logits_loss.append(float(bsz * float(kd_logits_loss_val)))
                        valid_proto_repel_loss.append(float(bsz * float(proto_loss_val)))

            train_loss = sum(train_loss) / max(train_total, 1e-8)
            train_kd_feat_loss = sum(train_kd_feat_loss) / max(train_total, 1e-8)
            train_kd_logits_loss = sum(train_kd_logits_loss) / max(train_total, 1e-8)
            train_proto_repel_loss = sum(train_proto_repel_loss) / max(train_total, 1e-8)
            train_determinant = sum(train_determinant) / max(len(train_determinant), 1)
            valid_loss = sum(valid_loss) / max(val_total, 1e-8)
            valid_kd_feat_loss = sum(valid_kd_feat_loss) / max(val_total, 1e-8)
            valid_kd_logits_loss = sum(valid_kd_logits_loss) / max(val_total, 1e-8)
            valid_proto_repel_loss = sum(valid_proto_repel_loss) / max(val_total, 1e-8)
            train_ac = sum(train_ac) / max(len(train_ac), 1)
            train_acc = train_hits / max(train_total, 1e-8)
            val_acc = val_hits / max(val_total, 1e-8)

            g_conf_mean  = (g_conf_sum  / g_count) if g_count > 0 else 0.0
            g_proto_mean = (g_proto_sum / g_count) if g_count > 0 else 0.0
            g_floor_mean = (g_floor_sum / g_count) if g_count > 0 else 0.0
            g_final_mean = (g_final_sum / g_count) if g_count > 0 else 0.0

            print(
                f"Epoch: {epoch} "
                f"Train: {train_loss:.2f} "
                f"KDf: {train_kd_feat_loss:.3f} KDlog: {train_kd_logits_loss:.3f} Proto: {train_proto_repel_loss:.3f} "
                f"Acc: {100 * train_acc:.2f} "
                f"Singularity: {train_ac:.3f} Det: {train_determinant:.5f} "
                f"| Gates(mean) conf/proto/floor/final: {g_conf_mean:.3f}/{g_proto_mean:.3f}/{g_floor_mean:.3f}/{g_final_mean:.3f} "
                f"| Val: {valid_loss:.2f} "
                f"KDf: {valid_kd_feat_loss:.3f} KDlog: {valid_kd_logits_loss:.3f} Proto: {valid_proto_repel_loss:.3f} "
                f"Acc: {100 * val_acc:.2f}"
            )

            if self.print_gate_diag and sample_total > 0:
                eps = 1e-8
                conf_rate = conf_count / max(sample_total, 1)
                strong_conf_rate = strong_conf_count / max(sample_total, 1)
                gate_on_conf_rate = gate_on_conf_count / max(conf_count, 1) if conf_count > 0 else 0.0
                supp_ratio_on_conf = supp_mass / max(raw_mass, eps) if raw_mass > 0 else 0.0
                conf_gap_mean = conf_gap_sum / max(conf_count, 1)
                conf_gate_mean = conf_gate_sum / max(conf_count, 1) if conf_count > 0 else 0.0

                print(
                    f"[GATE DIAG] ConfRate: {100*conf_rate:.1f}% | "
                    f"Strong(>{self._gate_diag_delta:.2f}): {100*strong_conf_rate:.1f}% | "
                    f"GateOnConf(<{self._gate_applied_thresh:.2f}): {100*gate_on_conf_rate:.1f}% | "
                    f"Supp@Conf: {100*supp_ratio_on_conf:.1f}% | "
                    f"Δp(mean|conf): {conf_gap_mean:.3f} | gate(mean|conf): {conf_gate_mean:.3f}"
                )

            # ------------------------------ NEW: drift snapshot ------------------------------
            if (t > 0) and (self.log_drift_every > 0) and ((epoch % self.log_drift_every) == (self.log_drift_every - 1)):
                self._snapshot_drift_metrics(t=t, epoch=epoch)

        if hasattr(criterion, "head"):
            frozen_head = copy.deepcopy(criterion.head).eval()
            for p in frozen_head.parameters(): p.requires_grad = False
            self.heads.append(frozen_head)

        self.distiller_module = copy.deepcopy(distiller).eval()
        self.adapter_module = copy.deepcopy(adapter).eval() if t > 0 else None

    # ------------------------------------ Prototypes ------------------------------------

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader, num_classes_in_t):
        self.model.eval()
        transforms = val_loader.dataset.transform
        new_means = torch.zeros((num_classes_in_t, self.S), device=self.device)
        new_covs = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)
        new_covs_not_shrinked = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)

        for c in range(num_classes_in_t):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[t]
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False, drop_last=False)

            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                feats = self.model(images)
                class_features[from_: from_ + bsz] = feats
                feats = self.model(torch.flip(images, dims=(3,)))
                class_features[from_ + bsz: from_ + 2 * bsz] = feats
                from_ += 2 * bsz

            new_means[c] = class_features.mean(dim=0)
            cov_raw = torch.cov(class_features.T)
            new_covs_not_shrinked[c] = cov_raw
            new_cov = self.shrink_cov(cov_raw, self.shrink)
            if self.adaptation_strategy == "diag":
                new_cov = torch.diag(torch.diag(new_cov))
            new_covs[c] = new_cov
            if torch.isnan(new_cov).any():
                raise RuntimeError(f"NaN in covariance matrix of class {c}")

        self.means = torch.cat((self.means, new_means), dim=0)
        self.covs = torch.cat((self.covs, new_covs), dim=0)
        self.covs_raw = torch.cat((self.covs_raw, new_covs_not_shrinked), dim=0)

    def adapt_distributions(self, t, trn_loader, val_loader):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size,
                                                 num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size,
                                                 num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        self.model.eval()

        # Save the incoming old statistics before any transport is applied.
        old_means = copy.deepcopy(self.means)
        old_covs = copy.deepcopy(self.covs)

        if not self.adapt_train:
            # Camera-ready default for the strongest CIFAR/Tiny runs:
            # skip secondary post-hoc adapter fitting and directly reuse the adapter
            # co-evolved during backbone training. This matches the one-stage GATF setting.
            if hasattr(self, "adapter_module") and self.adapter_module is not None:
                print("[adapt_distributions] Training DISABLED (--no-adapt-train). Reusing co-evolved adapter_module.")
                adapter = self.adapter_module.to(self.device).eval()
            else:
                print("[adapt_distributions] --no-adapt-train but adapter_module is None; building an untrained adapter for transport.")
                adapter_residual = self._build_projection(self.adapter_type).to(self.device, non_blocking=True)
                adapter = adapter_residual
                if t > 0 and self.prior_kind != "none" and self.prior_mode in ("adapt", "train_adapt"):
                    prior_info = self._estimate_prior(trn_loader)
                    self._prior_cache = copy.deepcopy(prior_info)
                    adapter = self._wrap_adapter_with_prior(prior_info, adapter_residual).to(self.device, non_blocking=True)
                    print(f"[PRIOR/ADAPT][NO-TRAIN] kind={self.prior_kind} ridge={self.prior_ridge:g} batches={self.prior_batches} "
                          f"zero_init_res={self.prior_residual_zero_init} "
                          f"{'(gls_target_cov='+self.gls_target_cov+')' if self.prior_kind=='gls' else ''}")
                self.adapter_module = copy.deepcopy(adapter).eval()
        else:
            adapter_residual = self._build_projection(self.adapter_type).to(self.device, non_blocking=True)
            adapter = adapter_residual

            if t > 0 and self.prior_kind != "none" and self.prior_mode in ("adapt", "train_adapt"):
                prior_info = self._estimate_prior(trn_loader)
                self._prior_cache = copy.deepcopy(prior_info)
                adapter = self._wrap_adapter_with_prior(prior_info, adapter_residual).to(self.device, non_blocking=True)
                print(f"[PRIOR/ADAPT] kind={self.prior_kind} ridge={self.prior_ridge:g} batches={self.prior_batches} "
                      f"zero_init_res={self.prior_residual_zero_init} "
                      f"{'(gls_target_cov='+self.gls_target_cov+')' if self.prior_kind=='gls' else ''}")

            # Warm-start residual from adapter trained during backbone training (if available)
            if hasattr(self, "adapter_module") and self.adapter_module is not None:
                try:
                    if hasattr(self.adapter_module, "residual") and hasattr(adapter, "residual"):
                        adapter.residual.load_state_dict(self.adapter_module.residual.state_dict(), strict=False)
                    elif hasattr(adapter, "residual"):
                        adapter.residual.load_state_dict(self.adapter_module.state_dict(), strict=False)
                    else:
                        adapter.load_state_dict(self.adapter_module.state_dict(), strict=False)
                    print("[adapt_distributions] Warm-start adapter (residual) from backbone training.")
                except Exception as e:
                    print(f"[adapt_distributions] Warm-start failed: {e}")

            optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())

            for epoch in range(self.nepochs // 2):
                adapter.train()
                train_loss, valid_loss = [], []
                train_ac, train_determinant = [], []

                for images, _ in trn_loader:
                    bsz = images.shape[0]
                    images = images.to(self.device, non_blocking=True)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        z_new = self.model(images)
                        z_old = self.old_model(images)
                    z_old_to_new = adapter(z_old)
                    loss = F.mse_loss(z_old_to_new, z_new)
                    ac, det = 0.0, torch.tensor(0.0, device=self.device)
                    if self.alpha > 0:
                        ac, det = loss_ac(z_old_to_new, self.beta)
                    total = loss + self.alpha * ac
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                    optimizer.step()
                    train_loss.append(float(bsz * loss))
                    train_ac.append(float(ac))
                    train_determinant.append(float(torch.clamp(torch.abs(det), max=1e8)))

                lr_scheduler.step()

                if epoch % 10 == 9:
                    adapter.eval()
                    with torch.no_grad():
                        for images, _ in val_loader:
                            bsz = images.shape[0]
                            images = images.to(self.device, non_blocking=True)
                            z_new = self.model(images)
                            z_old = self.old_model(images)
                            z_old_to_new = adapter(z_old)
                            total = F.mse_loss(z_old_to_new, z_new)
                            valid_loss.append(float(bsz * total))

                train_loss = sum(train_loss) / max(len(trn_loader.dataset), 1)
                train_determinant = sum(train_determinant) / max(len(train_determinant), 1)
                train_ac = sum(train_ac) / max(len(train_ac), 1)
                valid_loss = sum(valid_loss) / max(len(val_loader.dataset), 1)
                print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                      f"Singularity: {train_ac:.3f} Det: {train_determinant:.5f}")

            # ------------------------------ DUMP (adapter_after) ------------------------------
            if self.dump:
                exp_dir = self._get_exp_dir()
                torch.save(adapter.state_dict(), f"{exp_dir}/adapter_after_{t}.pth")
                if self._prior_cache is not None:
                    torch.save(self._prior_cache, f"{exp_dir}/prior_after_{t}.pth")

            self.adapter_module = copy.deepcopy(adapter).eval()

        # Distribution-level adaptation (full A = prior + residual)
        with torch.no_grad():
            adapter.eval()
            C_all = self.means.shape[0]
            C_old = self.task_offset[t]
            class_w = self._compute_class_drift_weights(adapter, t)

            if self.adaptation_strategy == "mean":
                if self.use_drift_proto_blend and C_old > 0:
                    mu_old = self.means[:C_old].clone()
                    mu_hat = adapter(mu_old)
                    self.means[:C_old] = mu_old + class_w.unsqueeze(1) * (mu_hat - mu_old)
                else:
                    self.means[:C_old] = adapter(self.means[:C_old])

            if self.adaptation_strategy in ("full", "diag"):
                for c in range(C_all):
                    cov = self.covs[c].clone()
                    if c < C_old:
                        distribution = self._safe_mvn(old_means[c], cov)
                        samples = distribution.sample((self.N,))
                        if torch.isnan(samples).any():
                            raise RuntimeError(f"NaN in sampled features for class {c}")
                        adapted_samples = adapter(samples)
                        mu_hat = adapted_samples.mean(0)
                        cov_new = torch.cov(adapted_samples.T)
                        cov_new = self.shrink_cov(cov_new, self.shrink)
                        if self.adaptation_strategy == "diag":
                            cov_new = torch.diag(torch.diag(cov_new))
                        if self.use_drift_proto_blend:
                            w_c = class_w[c] if class_w.numel() > 0 else 1.0
                            self.means[c] = self.means[c] + w_c * (mu_hat - self.means[c])
                            self.covs[c]  = (1 - w_c) * self.covs[c] + w_c * cov_new
                        else:
                            self.means[c] = mu_hat; self.covs[c] = cov_new

            # Diagnostics (unchanged)
            print("### Adaptation evaluation ###")
            for (subset, loaders) in [("train", self.train_data_loaders), ("val", self.val_data_loaders)]:
                old_mean_diff, new_mean_diff = [], []
                old_kld, new_kld = [], []
                old_cov_diff, old_cov_norm_diff, new_cov_diff, new_cov_norm_diff = [], [], [], []
                class_images = np.concatenate([dl.dataset.images for dl in loaders[-2:-1]])
                labels = np.concatenate([dl.dataset.labels for dl in loaders[-2:-1]])

                for c in list(np.unique(labels)):
                    train_indices = torch.tensor(labels) == c
                    if isinstance(trn_loader.dataset.images, list):
                        train_images = list(compress(class_images, train_indices))
                        ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
                    else:
                        ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
                    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False, drop_last=False)

                    from_ = 0
                    class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
                    for images in loader:
                        bsz = images.shape[0]
                        images = images.to(self.device, non_blocking=True)
                        feats = self.model(images)
                        class_features[from_: from_ + bsz] = feats
                        feats = self.model(torch.flip(images, dims=(3,)))
                        class_features[from_ + bsz: from_ + 2 * bsz] = feats
                        from_ += 2 * bsz

                    gt_mean = class_features.mean(0)
                    gt_cov = torch.cov(class_features.T)
                    gt_cov = self.shrink_cov(gt_cov, self.shrink)
                    if self.adaptation_strategy == "diag": gt_cov = torch.diag(torch.diag(gt_cov))
                    gt_gauss = self._safe_mvn(gt_mean, gt_cov)

                    old_mean_diff.append((gt_mean - old_means[c]).norm())
                    old_cov_diff.append(torch.norm(gt_cov - old_covs[c]))
                    old_cov_norm_diff.append(torch.norm(self.norm_cov(gt_cov.unsqueeze(0)) - self.norm_cov(old_covs[c].unsqueeze(0))))
                    old_gauss = self._safe_mvn(old_means[c], old_covs[c])
                    old_kld.append(torch.distributions.kl_divergence(old_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, old_gauss))

                    new_mean_diff.append((gt_mean - self.means[c]).norm())
                    new_cov_diff.append(torch.norm(gt_cov - self.covs[c]))
                    new_cov_norm_diff.append(torch.norm(self.norm_cov(gt_cov.unsqueeze(0)) - self.norm_cov(self.covs[c].unsqueeze(0))))
                    new_gauss = self._safe_mvn(self.means[c], self.covs[c])
                    new_kld.append(torch.distributions.kl_divergence(new_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, new_gauss))

                old_mean_diff = torch.stack(old_mean_diff); new_mean_diff = torch.stack(new_mean_diff)
                old_cov_diff = torch.stack(old_cov_diff); new_cov_diff = torch.stack(new_cov_diff)
                old_cov_norm_diff = torch.stack(old_cov_norm_diff); new_cov_norm_diff = torch.stack(new_cov_norm_diff)
                old_kld = torch.stack(old_kld); new_kld = torch.stack(new_kld)
                print(f"Old {subset} mean diff: {old_mean_diff.mean():.2f} ± {old_mean_diff.std():.2f}")
                print(f"New {subset} mean diff: {new_mean_diff.mean():.2f} ± {new_mean_diff.std():.2f}")
                print(f"Old {subset} cov diff: {old_cov_diff.mean():.2f} ± {old_cov_diff.std():.2f}")
                print(f"New {subset} cov diff: {new_cov_diff.mean():.2f} ± {new_cov_diff.std():.2f}")
                print(f"Old {subset} norm-cov diff: {old_cov_norm_diff.mean():.2f} ± {old_cov_norm_diff.std():.2f}")
                print(f"New {subset} norm-cov diff: {new_cov_norm_diff.mean():.2f} ± {new_cov_norm_diff.std():.2f}")
                print(f"Old {subset} KLD: {old_kld.mean():.2f} ± {old_kld.std():.2f}")
                print(f"New {subset} KLD: {new_kld.mean():.2f} ± {new_kld.std():.2f}\n")

    # ------------------------------ Legacy KD variants ------------------------------

    @torch.no_grad()
    def _safe_mvn(self, mean: torch.Tensor, cov: torch.Tensor, eps: float = 1e-4):
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        try:
            return torch.distributions.MultivariateNormal(mean, cov)
        except ValueError:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals_clamped = torch.clamp(eigvals, min=eps)
            cov_pd = (eigvecs * eigvals_clamped.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
            return torch.distributions.MultivariateNormal(mean, cov_pd)

    def distill_projected(self, t, loss, features, distiller, images):
        if t == 0: return loss, torch.zeros((), device=self.device)
        with torch.no_grad(): old_features = self.old_model(images)
        kd_loss = F.mse_loss(distiller(features), old_features)
        return loss + self.lamb * kd_loss, kd_loss

    def distill_features(self, t, loss, features, images):
        if t == 0: return loss, torch.zeros((), device=self.device)
        with torch.no_grad(): old_features = self.old_model(images)
        kd_loss = F.mse_loss(features, old_features)
        return loss + self.lamb * kd_loss, kd_loss

    def distill_logits(self, t, loss, features, images, old_heads):
        return loss, torch.zeros((), device=self.device)

    # ------------------------------ Optimizers ------------------------------

    def get_optimizer(self, parameters, t, wd):
        milestones = (int(0.3 * self.nepochs), int(0.6 * self.nepochs), int(0.9 * self.nepochs))
        lr = self.lr
        if t > 0 and not self.pretrained: lr *= 0.1
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(30, 60, 90)):
        optimizer = torch.optim.SGD(parameters, lr=self.lr_adapter, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_pseudo_head_optimizer(self, parameters, milestones=(15,)):
        optimizer = torch.optim.SGD(parameters, lr=0.1, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    # ------------------------------ Evaluation ------------------------------

    @torch.no_grad()
    def eval(self, t, val_loader):
        self.model.eval()
        tag_acc = Accuracy("multiclass", num_classes=self.means.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            if self.classifier == "linear":
                logits = self.pseudo_head(features)
                tag_preds = torch.argmax(logits, dim=1)
                taw_preds = torch.argmax(logits[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset
            else:
                if self.classifier == "bayes":
                    if self.is_normalization:
                        diff = F.normalize(features.unsqueeze(1), p=2, dim=-1) - F.normalize(self.means.unsqueeze(0), p=2, dim=-1)
                    else:
                        diff = features.unsqueeze(1) - self.means.unsqueeze(0)
                    res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                    res = res @ diff.unsqueeze(3)
                    dist = res.squeeze(2).squeeze(2)
                else:
                    dist = torch.cdist(features, self.means)
                tag_preds = torch.argmin(dist, dim=1)
                taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset
            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)
        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    def train_linear_head(self, t):
        distributions = []
        for c in range(self.means.shape[0]):
            cov = self.covs[c].clone()
            distributions.append(MultivariateNormal(self.means[c], cov))
        dataset = SampledDataset(distributions, 10000, self.task_offset)
        trn_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0, shuffle=True, drop_last=False)

        head = nn.Linear(self.S, self.task_offset[-1]).to(self.device)
        optimizer, lr_scheduler = self.get_pseudo_head_optimizer(head.parameters())

        for epoch in range(30):
            head.train(); train_loss = []
            for features, targets in trn_loader:
                bsz = features.shape[0]
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                logits = head(features)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()
            train_loss = sum(train_loss) / len(trn_loader.dataset)
            print(f"Epoch: {epoch} Linear head Train loss: {train_loss:.2f}")

        with torch.no_grad():
            head.eval(); self.pseudo_head = head
            logits_per_class = torch.zeros((0, self.means.shape[0]), device=self.device)
            for val_loader in self.val_data_loaders:
                for images, targets in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    features = self.model(images)
                    logits = self.pseudo_head(features)
                    logits_per_class = torch.cat((logits_per_class, logits), dim=0)
            logits_per_task = []
            for i in range(t + 1):
                logits_per_task.append(float(logits_per_class[:, self.task_offset[i]:self.task_offset[i + 1]].mean()))
            print(f"Logits per task: {list(logits_per_task)}")

    # ------------------------------ Diagnostics ------------------------------

    @torch.no_grad()
    def check_singular_values(self, t, val_loader):
        self.model.eval()
        self.svals_explained_by.append([])
        for i, _ in enumerate(self.train_data_loaders):
            if isinstance(self.train_data_loaders[i].dataset.images, list):
                train_images = self.train_data_loaders[i].dataset.images
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(self.train_data_loaders[i].dataset.images, val_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=val_loader.num_workers, shuffle=False, drop_last=False)
            from_ = 0
            class_features = torch.full((len(ds), self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                feats = self.model(images)
                class_features[from_: from_ + bsz] = feats
                from_ += bsz
            cov = torch.cov(class_features.T)
            svals = torch.linalg.svdvals(cov)
            xd = torch.cumsum(svals, 0); xd = xd[xd < self.sval_fraction * torch.sum(svals)]
            explain = xd.shape[0]
            self.svals_explained_by[t].append(explain)

    @torch.no_grad()
    def print_singular_values(self):
        print(f"### {self.sval_fraction} of eigenvalues sum is explained by: ###")
        for t, explained_by in enumerate(self.svals_explained_by):
            print(f"Task {t}: {explained_by}")

    @torch.no_grad()
    def shrink_cov(self, cov, alpha1=1.0, alpha2=0.0):
        if alpha2 == -1.0:
            return cov + alpha1 * torch.eye(cov.shape[0], device=self.device)
        diag_mean = torch.mean(torch.diagonal(cov))
        I = torch.eye(cov.shape[0], device=self.device)
        mask = I == 0.0
        off_diag_mean = torch.mean(cov[mask]) if mask.any() else torch.tensor(0.0, device=self.device)
        return cov + (alpha1 * diag_mean * I) + (alpha2 * off_diag_mean * (1 - I))

    @torch.no_grad()
    def norm_cov(self, cov):
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        std = torch.sqrt(diag + 1e-12)
        cov = cov / (std.unsqueeze(2) @ std.unsqueeze(1) + 1e-12)
        return cov

    @torch.no_grad()
    def print_covs(self, trn_loader, val_loader):
        self.model.eval()
        print("### Norms per task: ###")
        gt_means, gt_covs, gt_inverted_covs = [], [], []
        class_images = np.concatenate([dl.dataset.images for dl in self.train_data_loaders])
        labels = np.concatenate([dl.dataset.labels for dl in self.train_data_loaders])

        for c in list(np.unique(labels)):
            train_indices = torch.tensor(labels) == c
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(class_images, train_indices))
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False, drop_last=False)

            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                feats = self.model(images)
                class_features[from_: from_ + bsz] = feats
                feats = self.model(torch.flip(images, dims=(3,)))
                class_features[from_ + bsz: from_ + 2 * bsz] = feats
                from_ += 2 * bsz

            gt_means.append(class_features.mean(0))
            cov = torch.cov(class_features.T)
            gt_covs.append(cov)
            gt_inverted_covs.append(torch.linalg.inv(self.shrink_cov(cov, self.shrink)))

        gt_means = torch.stack(gt_means); gt_covs = torch.stack(gt_covs); gt_inverted_covs = torch.stack(gt_inverted_covs)
        mean_norms, cov_norms, gt_mean_norms, gt_cov_norms = [], [], [], []
        inverted_cov_norms, gt_inverted_cov_norms = [], []
        for task in range(len(self.task_offset[1:])):
            f, t_ = self.task_offset[task], self.task_offset[task + 1]
            mean_norms.append(round(float(torch.norm(self.means[f:t_], dim=1).mean()), 2))
            cov_norms.append(round(float(torch.linalg.matrix_norm(self.covs[f:t_]).mean()), 2))
            inv_cov = torch.linalg.inv(self.covs[f:t_]); inverted_cov_norms.append(round(float(torch.linalg.matrix_norm(inv_cov).mean()), 2))
            gt_mean_norms.append(round(float(torch.norm(gt_means[f:t_], dim=1).mean()), 2))
            gt_cov_norms.append(round(float(torch.linalg.matrix_norm(gt_covs[f:t_]).mean()), 2))
            gt_inverted_cov_norms.append(round(float(torch.linalg.matrix_norm(gt_inverted_covs[f:t_]).mean()), 2))
        print(f"Means: {mean_norms}")
        print(f"GT Means: {gt_mean_norms}")
        print(f"Covs: {cov_norms}")
        print(f"GT Covs: {gt_cov_norms}")
        print(f"Inverted Covs: {inverted_cov_norms}")
        print(f"GT Inverted Covs: {gt_inverted_cov_norms}")

    @torch.no_grad()
    def print_mahalanobis(self, t):
        self.model.eval()
        mahalanobis_per_class = torch.zeros((0, self.means.shape[0]), device=self.device)
        for val_loader in self.val_data_loaders:
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                diff = features.unsqueeze(1) - self.means.unsqueeze(0)
                res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                res = res @ diff.unsqueeze(3)
                dist = res.squeeze(2).squeeze(2)
                mahalanobis_per_class = torch.cat((mahalanobis_per_class, dist), dim=0)
        mahalanobis_per_task = []
        for i in range(t + 1):
            mahalanobis_per_task.append(float(mahalanobis_per_class[:, self.task_offset[i]:self.task_offset[i + 1]].mean()))
        print(f"Mahalanobis per task: {list(mahalanobis_per_task)}")

    # --------------------------------- Prototype Contrast (Bayes-friendly) ---------------------------------

    def _proto_repel_loss(
        self,
        z: torch.Tensor,
        proj_old_protos: torch.Tensor = None,
        margin: float = 0.2,
        topk: int = 0,
        metric: str = "maha",
        distiller: nn.Module = None,
        means_old: torch.Tensor = None,
        covs_old_inv: torch.Tensor = None
    ) -> torch.Tensor:
        if metric == "maha":
            if (distiller is None) or (means_old is None) or (covs_old_inv is None):
                return torch.zeros((), device=z.device)

            z_old = distiller(z)
            diff = z_old.unsqueeze(1) - means_old.unsqueeze(0)
            d2 = torch.einsum('bcs,csd,bcd->bc', diff, covs_old_inv, diff).clamp_min_(0.0)
            if topk and d2.shape[1] > topk:
                d2_topk, _ = torch.topk(d2, k=topk, largest=False, dim=1)
            else:
                d2_topk = d2
            loss_mat = F.relu(margin - d2_topk)
            return loss_mat.mean()

        elif metric == "cos":
            if proj_old_protos is None or proj_old_protos.numel() == 0:
                return torch.zeros((), device=z.device)
            z_n = F.normalize(z, dim=1)
            p_n = F.normalize(proj_old_protos, dim=1)
            sims = z_n @ p_n.t()
            thresh = 1.0 - margin
            if topk and proj_old_protos.shape[0] > topk:
                sims_topk, _ = torch.topk(sims, k=topk, dim=1)
            else:
                sims_topk = sims
            loss_mat = F.relu(sims_topk - thresh)
            return loss_mat.mean()

        else:  # 'l2'
            if proj_old_protos is None or proj_old_protos.numel() == 0:
                return torch.zeros((), device=z.device)
            d = torch.cdist(z, proj_old_protos, p=2)
            if topk and proj_old_protos.shape[0] > topk:
                d_topk, _ = torch.topk(d, k=topk, dim=1, largest=False)
            else:
                d_topk = d
            loss_mat = F.relu(margin - d_topk)
            return loss_mat.mean()


# ----------------------------------------- Utils -----------------------------------------

def compute_rotations(images, targets, total_classes):
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets

def loss_ac(features, beta):
    cov = torch.cov(features.T)
    cholesky = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
    ch_diag = torch.diag(cholesky)
    loss = - torch.clamp(ch_diag, max=beta).mean()
    return loss, torch.det(cov)
