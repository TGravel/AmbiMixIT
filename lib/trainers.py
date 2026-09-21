import random
from math import prod, pi, cos
from functools import partial
import platform

import torch
from torch.optim.lr_scheduler import LambdaLR

from lib.models import Model
from lib.utils import EPS, MetricAccumulator
from lib.data.dataloader_utils import double_mixture_generator, rotational_double_mixture_generator
from lib.losses import negative_sisnr, negative_snr, negative_sisnri, invariant_loss, sisnr, sisnri



# Tried it with torchmetrics instead of mir_eval, for better performance
from torch_mir_eval.batch_separation import bss_eval_sources
#import torchmetrics
import numpy as np


class Trainer:
    def __init__(self, config=None, model=None):
        self.global_step = 0
        self.scheduler = None
        self.loss_accumulator = MetricAccumulator()
        self.sisnri_accumulator = MetricAccumulator()
        self.sisnri_W_accumulator = MetricAccumulator()
        self.sisnri_XYZ_accumulator = MetricAccumulator()
        self.sdri_accumulator = MetricAccumulator()
        self.sdr_accumulator = MetricAccumulator()
        self.sir_accumulator = MetricAccumulator()
        self.sar_accumulator = MetricAccumulator()
        self.sisnr_accumulator = MetricAccumulator()
        self.doa_error_accumulator = MetricAccumulator()
        if model:
            self.model = model
            if config:
                self.model.config = config
        else:
            self.model = Model(config=config)
        self.model.to(self.model.config.device)
        # self.sdr_metric = torchmetrics.audio.SignalDistortionRatio().to(config.device)
        if hasattr(self.model.config, "weight_decay") and self.model.config.weight_decay > 0.0:
            self.optimizer = torch.optim.AdamW([
                {'params': self.model.parameters(), 'lr': self.model.config.lr},
            ], weight_decay=self.model.config.weight_decay)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters(), 'lr': self.model.config.lr},
        ])

    def train(self, dataloader):
        self.model.train()
        self.loss_accumulator.reset()

    def step(self, batch_loss):
        self.optimizer.zero_grad(set_to_none=True)
        batch_loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.model.config.grad_clip)
        self.optimizer.step()
        self.global_step += 1

        if self.scheduler is not None:
            self.scheduler.step()

        self.loss_accumulator.store(batch_loss)

    @torch.inference_mode()
    def validate(self, dataloader):
        self.model.eval()
        self.sisnri_accumulator.reset()
        self.sisnri_W_accumulator.reset()
        self.sisnri_XYZ_accumulator.reset()
        self.sisnr_accumulator.reset()
        self.sdri_accumulator.reset()
        self.sdr_accumulator.reset()
        self.sir_accumulator.reset()
        self.sar_accumulator.reset()
        audio_samples_to_log = None # log audio samples
        self.doa_error_accumulator.reset()

        channel_mismatch_warning_logged = False
        for batch_idx, (x_true_wave, s_true_wave) in enumerate(dataloader):
            x_true_wave = x_true_wave.to(self.model.config.device)
            s_true_wave = s_true_wave.to(self.model.config.device)

            if getattr(self.model, "in_channels", x_true_wave.size(1)) == 1 and x_true_wave.size(1) > 1:
                x_true_wave = x_true_wave[:, 0:1, :]
                if not channel_mismatch_warning_logged:
                    print("Warning: Model in_channels is 1 but input has more channels. Using only the first channel.")
                    channel_mismatch_warning_logged = True
            s_pred_wave = self.model(x_true_wave)


            C_out = s_true_wave.size(2)
            x_true_wave_for_loss = x_true_wave
            if C_out == 1 and x_true_wave.size(1) > 1:
                x_true_wave_for_loss = x_true_wave[:, 0:1, :]
            if x_true_wave_for_loss.dim() == 3:  # [B, C, T]
                x_true_wave_for_loss = x_true_wave_for_loss.unsqueeze(1)  # [B, 1, C, T]
            
            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=self.model.config.num_sources,
                max_sources=self.model.num_sources,
                # TODO: Still testing, Could improve performance originally num_mix=1
                allow_empty=False
            )

            batch_loss, best_perm_idx = invariant_loss(
                true=s_true_wave,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_sisnri, x_true_wave=x_true_wave_for_loss, eps=EPS),
                return_best_perm_idx=True  # Get the permutation index
            )
            batch_sisnri = -batch_loss
            self.sisnri_accumulator.store(batch_sisnri)

            B, S_true, C, T = s_true_wave.shape
            S_pred = s_pred_wave.shape[1]
            best_perms = mixing_matrices[best_perm_idx]
            s_pred_wave_flat = s_pred_wave.reshape(B, S_pred, C * T)
            permuted_s_pred_flat = torch.bmm(best_perms, s_pred_wave_flat)
            permuted_s_pred_wave = permuted_s_pred_flat.reshape(B, S_true, C, T)

            sisnr_abs = sisnr(s_true_wave, permuted_s_pred_wave)
            self.sisnr_accumulator.store(sisnr_abs.mean(dim=1))

            s_true_flat = s_true_wave.reshape(B, S_true, -1)
            permuted_s_pred_flat_for_eval = permuted_s_pred_wave.reshape(B, S_true, -1)
            x_true_baseline = x_true_wave_for_loss # [B, 1, C_out, T]
            x_true_flat_for_sdr = x_true_baseline.contiguous().reshape(B, 1, -1).expand(-1, S_true, -1)

            s_true_flat = s_true_flat.detach().contiguous().float()
            permuted_s_pred_flat_for_eval = permuted_s_pred_flat_for_eval.detach().contiguous().float()
            x_true_flat_for_sdr = x_true_flat_for_sdr.detach().contiguous().float()

            # Calculate SIR, SAR, SDR using torch_mir_eval
            sdr, sir, sar, _ = bss_eval_sources(s_true_flat, permuted_s_pred_flat_for_eval, compute_permutation=False)
            sdr_in, _, _, _ = bss_eval_sources(s_true_flat, x_true_flat_for_sdr, compute_permutation=False)
            self.sdr_accumulator.store(torch.mean(sdr, dim=1))
            self.sir_accumulator.store(torch.mean(sir, dim=1))
            self.sar_accumulator.store(torch.mean(sar, dim=1))
            self.sdri_accumulator.store(torch.mean(sdr - sdr_in, dim=1))

            # Added extra metrics for W and XYZ
            if C_out == 4:
                s_true_W = s_true_wave[:, :, 0:1, :]
                # TAU FOA uses ACN order: [W, Y, Z, X] -> XYZ = [3, 1, 2]
                s_true_XYZ = s_true_wave[:, :, [3, 1, 2], :]

                permuted_s_pred_W = permuted_s_pred_wave[:, :, 0:1, :]
                permuted_s_pred_XYZ = permuted_s_pred_wave[:, :, [3, 1, 2], :]

                x_true_W = x_true_wave[:, 0:1, :].unsqueeze(1)
                x_true_XYZ = x_true_wave[:, [3, 1, 2], :].unsqueeze(1)
                self.sisnri_W_accumulator.store(sisnri(s_true_W, permuted_s_pred_W, x_true_W))
                self.sisnri_XYZ_accumulator.store(sisnri(s_true_XYZ, permuted_s_pred_XYZ, x_true_XYZ))

                doa_errors = self._compute_doa_errors(permuted_s_pred_wave, s_true_wave)
                if doa_errors is not None:
                    self.doa_error_accumulator.store(doa_errors)

            # REPLACED BY MIR_EVAL
            #sdr_output = self.sdr_metric(permuted_s_pred_wave, s_true_wave)
            #input_pred = x_true_wave.expand_as(s_true_wave)
            #sdr_input = self.sdr_metric(input_pred, s_true_wave)
            #self.sdri_accumulator.store(sdr_output - sdr_input)


            if batch_idx == 0:
                audio_samples_to_log = (
                    x_true_wave.cpu(),
                    s_true_wave.cpu(),
                    permuted_s_pred_wave.cpu()
                )

        std, mean = self.sisnri_accumulator.std_mean()
        std_sisnr, mean_sisnr = self.sisnr_accumulator.std_mean()
        metrics = {
            'sisnri': mean.item(),
            'negative_sisnri': -mean.item(),
            'sisnri_std': std.item(),

            'sisnr': mean_sisnr.item(),
            'sisnr_std': std_sisnr.item(),

            'sdr': self.sdr_accumulator.std_mean()[1].item(),
            'sdr_std': self.sdr_accumulator.std_mean()[0].item(),

            'sdri': self.sdri_accumulator.std_mean()[1].item(),
            'sdri_std': self.sdri_accumulator.std_mean()[0].item(),

            'sir': self.sir_accumulator.std_mean()[1].item(),
            'sir_std': self.sir_accumulator.std_mean()[0].item(),

            'sar': self.sar_accumulator.std_mean()[1].item(),
            'sar_std': self.sar_accumulator.std_mean()[0].item(),

            'lr': self.optimizer.param_groups[0]['lr']
        }

        if not self.sisnri_W_accumulator.is_empty():
            std_W, mean_W = self.sisnri_W_accumulator.std_mean()
            metrics['sisnri_W'] = mean_W.item()
            metrics['sisnri_W_std'] = std_W.item()

        if not self.sisnri_XYZ_accumulator.is_empty():
            std_XYZ, mean_XYZ = self.sisnri_XYZ_accumulator.std_mean()
            metrics['sisnri_XYZ'] = mean_XYZ.item()
            metrics['sisnri_XYZ_std'] = std_XYZ.item()

        if not self.doa_error_accumulator.is_empty():
            std_doa, mean_doa = self.doa_error_accumulator.std_mean()
            metrics['doa_error_deg'] = mean_doa.item()
            metrics['doa_error_deg_std'] = std_doa.item()

        return metrics, audio_samples_to_log

    def _compute_doa_errors(self, pred_wave, true_wave):
        """Returns per-source angular errors (degrees) using active-intensity vectors."""
        pred_dirs, pred_valid = self._active_intensity_direction(pred_wave)
        true_dirs, true_valid = self._active_intensity_direction(true_wave)

        valid_mask = pred_valid & true_valid
        if not valid_mask.any():
            return None

        dot = (pred_dirs * true_dirs).sum(dim=2).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        errors = torch.acos(dot) * (180.0 / pi)

        return errors[valid_mask]

    def _active_intensity_direction(self, foa_wave):
        """Computes unit vectors from FOA waveforms using time-averaged active intensity."""
        W = foa_wave[:, :, 0:1, :]
        # TAU FOA uses ACN order: [W, Y, Z, X] -> XYZ = [3, 1, 2]
        XYZ = foa_wave[:, :, [3, 1, 2], :]
        intensity = W * XYZ
        intensity_mean = intensity.mean(dim=-1)
        norms = torch.linalg.norm(intensity_mean, dim=2, keepdim=True)
        unit = intensity_mean / (norms + EPS)
        valid = norms.squeeze(2) > 1e-6
        return unit, valid # watch out (intensity vector points in opposite of DOA)

    def get_loss(self):
        _, mean = self.loss_accumulator.std_mean()
        return mean.item()

    def get_model(self):
        return self.model


class MixtureInvariantTrainer(Trainer):
    def __init__(self, config=None, model=None, max_underlying_batches=None):
        if not model:
            if config.head_multiplier is None:
                config.head_multiplier = 2  # Experiment 1: Use double the sources for MixIT (*3 / *4) default: *2
            if config.head_multiplier < 2:
                raise ValueError("For Mixture Invariant Training, head_multiplier must be >= 2, otherwise there are not enough output heads for the mixtures.")
            model = Model(config=config, num_sources=config.num_sources * config.head_multiplier)

        super().__init__(config, model)

        self.sisnri_mixit_oracle_accumulator = MetricAccumulator()
        self.max_underlying_batches = max_underlying_batches

    def train(self, dataloader):
        super().train(dataloader)

        batch_count = 0
        channel_mismatch_warning_logged = False

        rotate = (
            getattr(self.model.config, 'mixit_rotate', False)
            and getattr(self.model, 'in_channels', 1) == 4
        )
        if rotate:
            pair_generator = rotational_double_mixture_generator(
                dataloader,
                p=getattr(self.model.config, 'mixit_rotate_prob', 1.0),
                rotate_both=True,
            )
        else:
            pair_generator = double_mixture_generator(dataloader)

        for x_true_wave_1, x_true_wave_2 in pair_generator:
            x_true_wave_1 = x_true_wave_1.to(self.model.config.device).unsqueeze(1)
            x_true_wave_2 = x_true_wave_2.to(self.model.config.device).unsqueeze(1)

            x_true_wave_double = torch.cat([x_true_wave_1, x_true_wave_2], dim=1)
            x_true_wave_mom = x_true_wave_double.sum(dim=1, keepdim=False)

            if getattr(self.model, "in_channels", x_true_wave_mom.size(1)) == 1 and x_true_wave_mom.size(1) > 1:
                x_true_wave_mom = x_true_wave_mom[:, 0:1, :]
                if not channel_mismatch_warning_logged:
                    print("Warning: Model in_channels is 1 but input has more channels. Using only the first channel.")
                    channel_mismatch_warning_logged = True

            s_pred_wave = self.model(x_true_wave_mom)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=2,
                max_sources=self.model.num_sources
            )

            if s_pred_wave.dim() == 3:  # [B, S, T]
                s_pred_wave = s_pred_wave.unsqueeze(2)  # [B, S, C, T]
            if x_true_wave_double.dim() == 3:  # [B, S, T]
                x_true_wave_double = x_true_wave_double.unsqueeze(2)  # [B, S, C, T]
            
            B, S_double, C_in, T = x_true_wave_double.shape

            C_out = self.model.out_channels
            if C_out == 1 and C_in > 1:
                x_true_wave_double_for_loss = x_true_wave_double[:, :, 0:1, :]
            else:
                x_true_wave_double_for_loss = x_true_wave_double

            batch_loss = invariant_loss(
                true=x_true_wave_double_for_loss,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=negative_sisnr,
            )
            self.step(batch_loss)
            
            batch_count += 2
            if self.max_underlying_batches is not None and batch_count >= self.max_underlying_batches:
                break

            yield 2


    @torch.inference_mode()
    def validate(self, dataloader):
        metrics, audio_samples = super().validate(dataloader)

        self.sisnri_mixit_oracle_accumulator.reset()

        channel_mismatch_warning_logged = False
        for x_true_wave, s_true_wave in dataloader:
            x_true_wave = x_true_wave.to(self.model.config.device)
            s_true_wave = s_true_wave.to(self.model.config.device)

            if getattr(self.model, "in_channels", x_true_wave.size(1)) == 1 and x_true_wave.size(1) > 1:
                x_true_wave = x_true_wave[:, 0:1, :]
                if not channel_mismatch_warning_logged:
                    print("Warning: Model in_channels is 1 but input has more channels. Using only the first channel.")
                    channel_mismatch_warning_logged = True

            s_pred_wave = self.model(x_true_wave)

            C_out = s_true_wave.size(2)
            x_true_wave_for_loss = x_true_wave
            if C_out == 1 and x_true_wave.size(1) > 1:
                x_true_wave_for_loss = x_true_wave[:, 0:1, :]
            if x_true_wave_for_loss.dim() == 3:
                x_true_wave_for_loss = x_true_wave_for_loss.unsqueeze(1)
            

            if s_pred_wave.dim() == 3:  # [B, S, T]
                s_pred_wave = s_pred_wave.unsqueeze(2)

            mixing_matrices = self.model.generate_mixing_matrices(
                num_targets=self.model.config.num_sources,
                max_sources=self.model.num_sources,
                allow_empty=True # Could improve performance by allowing empty sources here as well
            )

            
            batch_sisnri = -invariant_loss(
                true=s_true_wave,
                pred=s_pred_wave,
                mixing_matrices=mixing_matrices,
                loss_func=partial(negative_sisnri, x_true_wave=x_true_wave_for_loss, eps=EPS),
            )

            self.sisnri_mixit_oracle_accumulator.store(batch_sisnri)

        std, mean = self.sisnri_mixit_oracle_accumulator.std_mean()
        metrics['sisnri_mixit_oracle'] = mean.item()
        metrics['sisnri_mixit_oracle_std'] = std.item()
        return metrics, audio_samples



TRAINER_MAPPING = {
    'mixit': MixtureInvariantTrainer,
}


def get_trainer(model_name):
    try:
        return TRAINER_MAPPING[model_name]
    except KeyError:
        raise Exception(f'unknown model name: {model_name}')
    
def make_warmup_cosine_scheduler(optimizer, base_lr, total_steps, warmup_ratio=0.08, start_lr=3e-6, min_lr=1e-6): # Experiment: Added step-based Cosine Scheduler with Warmup 
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        min_lr_ratio = min_lr / base_lr
        start_lr_ratio = start_lr / base_lr

        def lr_lambda(step):
            # Warmup phase: start_lr -> base_lr
            if step < warmup_steps:
                alpha = (step + 1) / warmup_steps
                return start_lr_ratio + alpha * (1.0 - start_lr_ratio)
            
            # Maximum step reached, return min_lr_ratio
            if step >= total_steps:
                return min_lr_ratio
            
            # Cosine: base_lr -> min_lr
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + cos(pi * progress))  # 1 -> 0
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

        return LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps
