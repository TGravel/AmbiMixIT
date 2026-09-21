from math import prod
import torch


def snr(true_wave, pred_wave, snr_max=None):
    """
    true_wave, pred_wave: [B, S, C, T] or [B, 1, C, T]
    """
    eps = torch.finfo(true_wave.dtype).eps
    true_wave_square_sum = true_wave.square().sum(dim=(-2, -1))  # sum over [C, T]
    error = (true_wave - pred_wave).square().sum(dim=(-2, -1))

    true_wave_square_sum = torch.max(true_wave_square_sum, torch.tensor(eps, device=true_wave.device))
    error = torch.max(error, torch.tensor(eps, device=true_wave.device))

    if snr_max is None:
        ratio = true_wave_square_sum / (error + eps)
    else:
        threshold_factor = 10 ** (-snr_max / 10)
        soft_threshold = threshold_factor * true_wave_square_sum
        ratio = true_wave_square_sum / (error + soft_threshold + eps)

    ratio = torch.max(ratio, torch.tensor(eps, device=ratio.device))
    return 10 * torch.log10(ratio)


def sisnr(true_wave, pred_wave, eps=1e-8):
    """
    true_wave, pred_wave: [B, S, C, T]
    """
    B, S, C, T_samples = true_wave.shape
    
    true_flat = true_wave.reshape(B, S, C * T_samples)
    pred_flat = pred_wave.reshape(B, S, C * T_samples)
    
    dot = (true_flat * pred_flat).sum(dim=-1)  # [B, S]
    true_energy = true_flat.square().sum(dim=-1)  # [B, S]
    true_energy = torch.clamp(true_energy, min=eps)

    scale = dot / (true_energy + eps)
    proj = scale.unsqueeze(-1).unsqueeze(-1) * true_wave  # [B, S, 1, 1] * [B, S, C, T]

    noise = pred_wave - proj
    noise_energy = noise.reshape(B, S, C*T_samples).square().sum(dim=-1)
    ratio = proj.reshape(B, S, C*T_samples).square().sum(dim=-1) / (noise_energy + eps)

    return 10 * torch.log10(ratio + eps)  # [B, S]


def sisnri(true_wave, pred_wave, x_true_wave, eps=1e-8):
    """
    x_true_wave: [B, 1, C, T]
    """

    S_true = true_wave.shape[1]
    if x_true_wave.shape[1] == 1 and S_true > 1:
        x_true_wave_expanded = x_true_wave.expand(-1, S_true, -1, -1)
    else:
        x_true_wave_expanded = x_true_wave
    return sisnr(true_wave, pred_wave, eps=eps) - sisnr(true_wave, x_true_wave_expanded, eps=eps)


def negative_snr(true_wave, pred_wave, snr_max=None):
    if true_wave.dim() == 3:
        true_wave = true_wave.unsqueeze(1)
    if pred_wave.dim() == 3:
        pred_wave = pred_wave.unsqueeze(1)
    return -snr(true_wave, pred_wave, snr_max)


def negative_sisnr(true_wave, pred_wave, eps=1e-8):
    if true_wave.dim() == 3:
        true_wave = true_wave.unsqueeze(1)
    if pred_wave.dim() == 3:
        pred_wave = pred_wave.unsqueeze(1)
    return -sisnr(true_wave, pred_wave, eps=eps)


def negative_sisnri(true_wave, pred_wave, x_true_wave, eps=1e-8):
    if true_wave.dim() == 3:
        true_wave = true_wave.unsqueeze(1)
    if pred_wave.dim() == 3:
        pred_wave = pred_wave.unsqueeze(1)
    if x_true_wave.dim() == 3:
        x_true_wave = x_true_wave.unsqueeze(1)
    return -sisnri(true_wave, pred_wave, x_true_wave, eps=eps)


def invariant_loss(true, pred, mixing_matrices, loss_func, return_best_perm_idx=False):
    B, S_true, C, T = true.shape  # S_true is the number of sources in the ground truth
    B_pred, S_pred, C_pred, T_pred = pred.shape
    
    # Debug
    # print(f"invariant_loss shapes: true={true.shape}, pred={pred.shape}")
    # print(f"S_true={S_true}, S_pred={S_pred}, mixing_matrices={mixing_matrices.shape}")
    pred_flat = pred.reshape(B, S_pred, C * T)

    perm_losses = []
    for perm_idx in range(mixing_matrices.size(0)):
        pred_flat_mix = mixing_matrices[perm_idx].matmul(pred_flat)
        pred_mix = pred_flat_mix.reshape(B, S_true, C, T)
        perm_losses.append(loss_func(true, pred_mix).mean(dim=1))

    loss_perms = torch.stack(perm_losses, dim=1)
    batch_loss, best_perm_idx = loss_perms.min(dim=1)

    return (batch_loss, best_perm_idx) if return_best_perm_idx else batch_loss


