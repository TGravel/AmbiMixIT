import os
import itertools

import torch
from torch import nn
from lib.conv_tasnet import MaskGenerator # Note: ConvTasNet's MaskGenerator is used here! Changes made to accept output_dim

from lib.utils import EPS, get_logger, soft_mask
from lib.transforms import Transform



class Model(nn.Module):
    MIXING_MATRICES_CACHE = {}

    def __init__(self, config, num_sources=None, num_channels=None):
        super().__init__()

        self.args = locals()
        self.logger = get_logger('model')
        self.config = config
        self.in_channels  = getattr(self.config, 'in_channels', 1)
        self.out_channels = getattr(self.config, 'out_channels', 1)
        # Active-intensity TF features stack [|W|,|Y|,|Z|,|X|, weighted unit-I (3), reliability (1)]
        self.use_intensity_features = (
            self.in_channels == 4
            and bool(getattr(self.config, 'use_intensity_features', False))
        )
        # scalar-steering: predict a mono dry source per output head plus 4 scalars
        self.use_scalar_steering = bool(getattr(self.config, 'use_scalar_steering', False))
        if self.use_scalar_steering and (self.in_channels != 4 or self.out_channels != 4):
            raise ValueError("use_scalar_steering requires in_channels=4 and out_channels=4.")
        # When True (default), the YZX branch detaches mono_wave so only the scalar predictor sees YZX gradients
        self.scalar_steering_detach_mask = bool(getattr(self.config, 'scalar_steering_detach_mask', True))

        self.num_sources = self.config.num_sources if num_sources is None else num_sources

        self.transform = Transform(
            stft_frame_size=self.config.stft_frame_size,
            stft_hop_size=self.config.stft_hop_size,
            device=self.config.device,
        )
        self.num_frequency_bins = self.transform.num_frequency_bins

        # Input feature width per F-bin: in_channels magnitudes plus, for FOA, 3 directional + 1 reliability.
        feature_channels = self.in_channels + (4 if self.use_intensity_features else 0)
        mask_generator_input_dim = feature_channels * self.num_frequency_bins
        # Scalar steering uses a single (mono) mask channel
        mask_output_per_source = 1 if self.use_scalar_steering else self.out_channels
        mask_generator_output_dim = mask_output_per_source * self.num_frequency_bins

        self.mask_generator = MaskGenerator(
            input_dim=mask_generator_input_dim,
            num_sources=self.num_sources,
            kernel_size=3,
            num_feats=128,
            num_hidden=512,
            num_layers=4,
            num_stacks=3,
            msk_activate='sigmoid',
            output_dim=mask_generator_output_dim,
        )

        if self.use_scalar_steering:
            scalar_hidden = int(getattr(self.config, 'scalar_steering_hidden_dim', 128))
            # Time-distributed 1x1 conv on the mask-generator input features, time-pooled into a per-utterance vector, projected to [num_sources * 3] YZX gains.
            self.scalar_trunk = nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=mask_generator_input_dim, eps=1e-8),
                nn.Conv1d(mask_generator_input_dim, scalar_hidden, kernel_size=1),
                nn.PReLU(),
                nn.Conv1d(scalar_hidden, scalar_hidden, kernel_size=1),
                nn.PReLU(),
            )
            # Predict only YZX scalars; W gain is pinned to 1 so the W path is identical to the mono baseline and YZX losses cannot change the mono mask (mono_wave is detached on the YZX branch in forward()).
            self.scalar_proj = nn.Linear(scalar_hidden, self.num_sources * 3)
            nn.init.normal_(self.scalar_proj.weight, std=0.01)
            nn.init.zeros_(self.scalar_proj.bias)
            self.latest_scalars = None

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, weights_only=False)
        if 'device' in checkpoint.keys():
            del checkpoint['device'] ## FIXME: remove
        if device:
            checkpoint['config'].device = device
        state_dict = checkpoint.pop('state_dict')

        instance = cls(**checkpoint)
        instance.load_state_dict(state_dict)
        instance.to(instance.config.device)

        return instance

    def save(self, path):
        checkpoint = self.args.copy()
        del checkpoint['self']
        del checkpoint['__class__']
        checkpoint['config'] = self.config
        checkpoint['state_dict'] = self.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def _active_intensity_direction(self, x_mix_mag, x_mix_phase):
        """Per-TF-bin unit active-intensity vector and a [0, 1] reliability scalar.

        Returns:
            unit_direction: [B, 3, F, T] in the standard [X, Y, Z] basis.
            reliability:    [B, 1, F, T], |I| / (energy density), clamped.
        """
        if self.in_channels != 4:
            raise ValueError("Active-intensity features require FOA input with 4 channels.")

        w_mag = x_mix_mag[:, 0:1, :, :]
        w_phase = x_mix_phase[:, 0:1, :, :]
        # TAU FOA uses ACN order [W, Y, Z, X]. Reorder directional channels to [X, Y, Z].
        xyz_mag = x_mix_mag[:, [3, 1, 2], :, :]
        xyz_phase = x_mix_phase[:, [3, 1, 2], :, :]

        w_complex = torch.polar(w_mag, w_phase)
        xyz_complex = torch.polar(xyz_mag, xyz_phase)
        active_intensity = torch.real(torch.conj(w_complex) * xyz_complex)

        total_energy = w_mag.square() + xyz_mag.square().sum(dim=1, keepdim=True)
        intensity_norm = torch.linalg.vector_norm(active_intensity, dim=1, keepdim=True)
        reliability = (intensity_norm / (total_energy + EPS)).clamp(0.0, 1.0)
        unit_direction = active_intensity / (intensity_norm + EPS)
        return unit_direction, reliability

    def _build_mask_input(self, x_mix_mag, x_mix_phase):
        """Stack magnitudes and (for FOA) per-TF active-intensity features.

        Returns: [B, F_in, T] flat tensor for the mask generator, where
            F_in = (in_channels + 4) * num_frequency_bins  if FOA features on,
            F_in = in_channels * num_frequency_bins        otherwise.
        """
        B, C, F_bins, T = x_mix_mag.shape
        feats = [x_mix_mag]  # [B, C, F, T]
        if self.use_intensity_features:
            unit_dir, reliability = self._active_intensity_direction(x_mix_mag, x_mix_phase)
            # Weighted direction makes low-confidence bins toward zero, so the network gets a single feature that encodes both direction and confidence, plus reliability on its own as a separate scalar feature.
            feats.append(unit_dir * reliability)  # [B, 3, F, T]
            feats.append(reliability)             # [B, 1, F, T]
        stacked = torch.cat(feats, dim=1)  # [B, C', F, T]
        return stacked.reshape(B, stacked.size(1) * F_bins, T)

    def forward(self, x_true_wave):
        B, C_in, L_wave = x_true_wave.shape
        S = self.num_sources

        if C_in != self.in_channels:
            raise ValueError(f"Input channels mismatch: got {C_in}, expected {self.in_channels}.")

        x_mix_mag, x_mix_phase = self.transform.stft_multichannel(x_true_wave)
        B_check, C_check, F_actual, T_frames = x_mix_mag.shape
        if F_actual != self.num_frequency_bins:
            raise ValueError("Something wrong with the Bins of the STFT?")

        mask_input = self._build_mask_input(x_mix_mag, x_mix_phase)
        m_pred_mag_flat = self.mask_generator(mask_input)

        if self.use_scalar_steering:
            # Mono mask applied to W gives a dry per-source signal. 3 learned YZX scalars (W gain pinned to 1) re-encode it into FOA. Output is a rank-1 ambisonic  source per head.
            m_pred_mag_reshaped = m_pred_mag_flat.view(B, S, 1, self.num_frequency_bins, T_frames)
            w_mag = x_mix_mag[:, 0:1, :, :].unsqueeze(1)         # [B, 1, 1, F, T]
            w_phase = x_mix_phase[:, 0:1, :, :]                  # [B, 1, F, T]
            s_pred_mono_mag = soft_mask(m_pred_mag_reshaped, w_mag)
            mono_phase = w_phase.unsqueeze(1).expand(B, S, 1, self.num_frequency_bins, T_frames)
            s_pred_mono_wave = self.transform.istft_multichannel_multisource(
                s_pred_mono_mag, mono_phase, L_wave
            )                                                    # [B, S, 1, L_wave]

            scalar_feat = self.scalar_trunk(mask_input)          # [B, hidden, T_frames]
            scalar_feat = scalar_feat.mean(dim=-1)               # [B, hidden]
            yzx_scalars = self.scalar_proj(scalar_feat).view(B, S, 3)  # [B, S, 3]
            # Log [W=1, Y, Z, X] for DOA computation.
            self.latest_scalars = torch.cat(
                [torch.ones(B, S, 1, device=yzx_scalars.device, dtype=yzx_scalars.dtype),
                 yzx_scalars],
                dim=-1,
            ).detach()

            # W path: pinned gain of 1, gradient flows into the mono mask normally.
            pred_w = s_pred_mono_wave                            # [B, S, 1, L_wave]
            # YZX path: optionally detach mono_wave so YZX loss only updates the scalar predictor; without detach, YZX loss could also flows back into the mono mask.
            mono_wave_for_yzx = s_pred_mono_wave.detach() if self.scalar_steering_detach_mask else s_pred_mono_wave
            pred_yzx = yzx_scalars.unsqueeze(-1) * mono_wave_for_yzx  # [B, S, 3, L_wave] channel order is [W, Y, Z, X].
            return torch.cat([pred_w, pred_yzx], dim=2)

        m_pred_mag_reshaped = m_pred_mag_flat.view(B, S, self.out_channels, self.num_frequency_bins, T_frames)

        use_w_only = self.out_channels == 1 and self.in_channels > 1
        if use_w_only:
            if not getattr(self.config, "w_out_mode", False):
                raise ValueError("out_channels=1 with in_channels>1 requires w_out_mode until downmix is implemented.")
            x_mix_mag_target = x_mix_mag[:, 0:1, :, :]
            x_mix_phase_target = x_mix_phase[:, 0:1, :, :]
        else:
            x_mix_mag_target = x_mix_mag
            x_mix_phase_target = x_mix_phase

        s_pred_mag_sources = soft_mask(m_pred_mag_reshaped, x_mix_mag_target.unsqueeze(1))
        new_phase = x_mix_phase_target.unsqueeze(1).expand(B, S, self.out_channels, self.num_frequency_bins, T_frames)

        s_pred_wave = self.transform.istft_multichannel_multisource(s_pred_mag_sources, new_phase, L_wave)
        return s_pred_wave

    def generate_mixing_matrices(self, num_targets, max_sources, num_mix=None, allow_empty=False):
        parameters = locals()
        del parameters['self']

        def do_generate():
            output_perms = itertools.product([0, 1], repeat=max_sources)
            if num_mix is not None:
                output_perms = [perm for perm in output_perms if sum(perm) == num_mix]
            target_perms = list(itertools.product(output_perms, repeat=num_targets))
            perm_list = []
            for target_perm in target_perms:
                perm_sum = torch.tensor(target_perm).sum(dim=0)
                if (perm_sum <= 1).all() if allow_empty else (perm_sum == 1).all():
                    perm_list.append(target_perm)
            self.logger.info('mixing matrices are generated with %d permutations for parameters %s',
                             len(perm_list), parameters)
            return torch.tensor(perm_list).float().to(self.config.device)

        cache_key = '_'.join(str(v) for k, v in parameters.items())
        try:
            r = Model.MIXING_MATRICES_CACHE[cache_key]
        except KeyError:
            r = Model.MIXING_MATRICES_CACHE[cache_key] = do_generate()
        return r
