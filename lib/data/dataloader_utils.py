from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
from torchaudio.datasets.librimix import LibriMix
from lib.data.dataset import Dataset

from lib.data.collate_utils import collate_fn_wsj0mix_train, collate_fn_wsj0mix_test
from lib.data.realm import RealM
from lib.data.spatial_aug import random_so3, rotate_foa_acn


SAMPLE_RATE = 8000


def get_dataset_specs(librimix_root=None, realm_root=None, dcase_root=None):
    if librimix_root and realm_root:
        raise Exception('only one dataset root should be given')
    elif librimix_root:
        dataset_name = 'librimix'
        dataset_root = librimix_root
    elif realm_root:
        dataset_name = 'realm'
        dataset_root = realm_root
    elif dcase_root:
        dataset_name = 'dcase19'
        dataset_root = dcase_root
    else:
        raise Exception('at least one dataset root should be given')

    return dataset_name, dataset_root


def create_dataloader(dataset_name, dataset_root, partition, batch_size, subsample_ratio=1.0, shuffle=None):
    assert partition in ('training', 'validation', 'testing')

    if partition in ('training', 'validation'):
        collate_fn = partial(collate_fn_wsj0mix_train, sample_rate=SAMPLE_RATE, duration=3)
    else:
        collate_fn = partial(collate_fn_wsj0mix_test)

    if dataset_name == 'librimix':
        subset_mapping = {
            'training': 'train-360',
            'validation': 'dev',
            'testing': 'test',
        }
        dataset = LibriMix(
            root=dataset_root,
            subset=subset_mapping[partition],
            num_speakers=2,
            sample_rate=SAMPLE_RATE,
            task='sep_clean',
        )
    elif dataset_name == 'realm':
        dataset = RealM(
            root=dataset_root,
            partition=partition,
        )
    elif dataset_name == 'dcase19':
        dataset = Dataset(
            dataset_root,
            sr=48000,
            ambiorder=1,
            dataset="dcase19"
        )
    else:
        raise Exception(f'unknown dataset name: {dataset_name}')

    if subsample_ratio < 1.0:
        subsampled_dataset_size = int(len(dataset) // (1/subsample_ratio))
        indices = torch.randperm(len(dataset)).int()[:subsampled_dataset_size]
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(partition == 'training' or shuffle),
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=(partition == 'training'),
        pin_memory=True,
    )


def double_mixture_generator(dataloader):
    """
    Yield two independently shuffled mixtures per step to increase pairing diversity.
    Creates two separate iterators over the same dataloader (which shuffles each epoch),
    so batch i from iterator A is paired with batch i from iterator B drawn from a
    different shuffle order.
    """
    iterator_a = iter(dataloader)
    iterator_b = iter(dataloader)
    while True:
        try:
            x_true_wave_1, _ = next(iterator_a)
            x_true_wave_2, _ = next(iterator_b)
            yield x_true_wave_1, x_true_wave_2
        except StopIteration:
            return


def rotational_double_mixture_generator(dataloader, p=1.0, rotate_both=True):
    """Like double_mixture_generator, but rotates the directional channels of each
    FOA mixture by an independent uniform-random SO(3) per sample before yielding.

    The rotated mixture is what the trainer uses both as the MoM input and as the
    sub-mixture target, so the MixIT objective is unchanged in form. Pure FOA
    algebra: no labels are consumed.

    Args:
        dataloader: yields (mixture[B, C, T], sources[...]). Mono inputs pass through.
        p: probability of applying a rotation to a given pair (per yield).
        rotate_both: if False, only rotate the second mixture.
    """
    for x1, x2 in double_mixture_generator(dataloader):
        if x1.size(1) == 4 and torch.rand(()).item() < p:
            if rotate_both:
                R1 = random_so3(x1.size(0), device=x1.device, dtype=x1.dtype)
                x1 = rotate_foa_acn(x1, R1)
            R2 = random_so3(x2.size(0), device=x2.device, dtype=x2.dtype)
            x2 = rotate_foa_acn(x2, R2)
        yield x1, x2
