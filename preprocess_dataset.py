import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dcasedataset import DCase2019Dataset, seed_worker


def preprocess_and_save(original_dataset_path, save_path, dataset_mode='train', num_val_samples=None, seed=42,
                        out_channels=4, num_workers=0, split=None):
    """
    Precompute mixtures/sources for a split. Uses num_workers=0 by default to avoid
    multiprocessing lock errors on some platforms (WSL/Windows).
    """
    print(f"Starting preprocessing for '{dataset_mode}' set...")
    os.makedirs(save_path, exist_ok=True)

    dataset = DCase2019Dataset(
        audio_dir=original_dataset_path,
        n_overlapping_sounds=2,
        chunk_size=2 * 16000,
        out_channels=out_channels,
        mode=dataset_mode,
        num_val_samples=num_val_samples,
        seed=seed,
        split=split if split is not None else list(range(1, 5)),
    )

    print(f"Dataset loaded with {len(dataset)} samples.")

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker)

    all_mixtures, all_sources = [], []
    for mixture_batch, sources_batch in tqdm(loader, desc=f"Preprocessing {dataset_mode}"):
        all_mixtures.append(mixture_batch.clone())
        all_sources.append(sources_batch.clone())

    mixtures = torch.cat(all_mixtures, dim=0)
    sources = torch.cat(all_sources, dim=0)
    torch.save({"mixtures": mixtures, "sources": sources}, os.path.join(save_path, f"{dataset_mode}.pt"))
    print(f"Finished! Saved samples to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', default='./datasets/dcase2019_16kHz/validation')
    parser.add_argument('--save-path', default=r'/mnt/f/Datasets/dcase2019/preprocessed/val')
    parser.add_argument('--mode', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num-val-samples', type=int, default=2048*5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-channels', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--split', type=int, nargs='+', default=None,
                        help='Original dataset split(s) to include, --split 0 for the held-out test split.')
    args = parser.parse_args()

    preprocess_and_save(
        original_dataset_path=args.dataset_path,
        save_path=args.save_path,
        dataset_mode=args.mode,
        num_val_samples=args.num_val_samples if (args.mode == 'val' or args.mode == 'test') else None,
        seed=args.seed,
        out_channels=args.out_channels,
        num_workers=args.num_workers,
        split=args.split,
    )
