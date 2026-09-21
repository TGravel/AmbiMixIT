"""SO(3) sampling and ambisonic rotation helpers.

Used by both per-sample augmentation (lib/dcasedataset.py) and rotation-MixIT
(lib/data/dataloader_utils.py). Operates on FOA waveforms in ACN order
[W, Y, Z, X]. Rotation acts on the directional channels in the standard
[X, Y, Z] basis; W is invariant.
"""

from math import pi

import torch


def random_so3(batch_size, device=None, dtype=torch.float32):
    """Uniformly distributed 3D rotations via Shoemake's quaternion method.

    Returns a tensor of shape [batch_size, 3, 3].
    """
    u = torch.rand(batch_size, 3, device=device, dtype=dtype)
    s1 = torch.sqrt(1.0 - u[:, 0])
    s2 = torch.sqrt(u[:, 0])
    two_pi = 2.0 * pi
    qx = s1 * torch.sin(two_pi * u[:, 1])
    qy = s1 * torch.cos(two_pi * u[:, 1])
    qz = s2 * torch.sin(two_pi * u[:, 2])
    qw = s2 * torch.cos(two_pi * u[:, 2])

    R = torch.stack([
        torch.stack([1 - 2 * (qy * qy + qz * qz),
                     2 * (qx * qy - qz * qw),
                     2 * (qx * qz + qy * qw)], dim=-1),
        torch.stack([2 * (qx * qy + qz * qw),
                     1 - 2 * (qx * qx + qz * qz),
                     2 * (qy * qz - qx * qw)], dim=-1),
        torch.stack([2 * (qx * qz - qy * qw),
                     2 * (qy * qz + qx * qw),
                     1 - 2 * (qx * qx + qy * qy)], dim=-1),
    ], dim=-2)
    return R


def rotate_foa_acn(wave, R):
    """Rotate the directional channels of an FOA waveform in ACN order.

    wave: tensor with channel axis at position -2 of length 4 (order [W, Y, Z, X]).
          Supported leading shapes include [4, T], [B, 4, T], [B, S, 4, T].
    R:    rotation matrix broadcastable to wave's leading dims, with shape
          [3, 3] or [..., 3, 3]. Acts on the [X, Y, Z] basis.
    """
    if wave.size(-2) != 4:
        return wave
    W = wave[..., 0:1, :]
    xyz = wave[..., [3, 1, 2], :]
    xyz_rot = torch.einsum('...ij,...jt->...it', R, xyz)
    yzx_rot = xyz_rot[..., [1, 2, 0], :]
    return torch.cat([W, yzx_rot], dim=-2)
