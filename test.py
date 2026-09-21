import os
import argparse
from pathlib import Path
from types import SimpleNamespace

# Deterministic validation
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from lib.utils import DATASET_PATH, PREPROCESSED_ROOT, CONFIG_FILENAME, BEST_CHECKPOINT_FILENAME, BEST_METRICS_FILENAME, configure_console_logger, \
    get_logger, default, build_run_name, ensure_clean_results_dir, setup_determinism, metrics_to_str
from lib.data.dataloader_utils import get_dataset_specs, create_dataloader
from lib.models import Model
from lib.trainers import get_trainer
from dcasedataset import DCase2019Dataset, SoundAugmenter, PrecomputedValDataset, ChannelSliceDataset, seed_worker


class Test:
    def __init__(self, train_results_dir,
                 eval_method=None,
                 eval_blind_num_repeat=None,
                 seed=None,
                 device_name=None,
                 num_val_samples=None,
                 in_channels=None,
                 out_channels=None,
                 n_overlapping_sounds=None,
                 zero_yzx=None,
                 mode=None):
        args = locals()
        del args['self']

        self.dataset_name, self.dataset_root = 'dcase19', DATASET_PATH
        run_name = build_run_name(
            args=args,
            prepend_items={'test': self.dataset_name},
            exclude_keys=['train_results_dir']
        )
        self.results_dir = os.path.join(train_results_dir, 'test', run_name)

        self.config = SimpleNamespace()
        self.config.results_dir = self.results_dir
        self.config.train_results_dir = train_results_dir
        self.config.eval_method = default(eval_method, None)
        self.config.eval_blind_num_repeat = default(eval_blind_num_repeat, None)
        self.config.seed = default(seed, 42)
        self.config.device_name = default(device_name, 'cuda')
        self.config.num_val_samples = default(num_val_samples, None)
        self.config.in_channels = default(in_channels, 1)
        self.config.out_channels = default(out_channels, 1)
        self.config.n_overlapping_sounds = default(n_overlapping_sounds, 2)
        self.config.zero_yzx = default(zero_yzx, False)
        self.config.mode = default(mode, 'val')
        self.config.valid_batch_size = 16
        if self.config.n_overlapping_sounds > 2 and self.config.num_val_samples is None:
            self.config.num_val_samples = 10240

        self.logger = None
        self.config.device = torch.device(self.config.device_name)

    def start(self):
        ensure_clean_results_dir(self.config.results_dir)
        setup_determinism(self.config.seed)
        self.logger = get_logger('test', self.config.results_dir)
        self.logger.info('config: %s', self.config)
        torch.save(self.config, os.path.join(self.config.results_dir, CONFIG_FILENAME))

        model = Model.load(
            path=os.path.join(self.config.train_results_dir, BEST_CHECKPOINT_FILENAME),
            device=self.config.device
        ).eval()

        if self.config.eval_method:
            model.config.eval_method = self.config.eval_method

        if self.config.eval_blind_num_repeat:
            model.config.eval_blind_num_repeat = self.config.eval_blind_num_repeat

        if model.config.eval_method == 'blind':
            model_name = 'mixcycle'
            partition = 'validation'
            batch_size = 128
            shuffle = True
        elif model.config.eval_method == 'reference-valid':
            model_name = model.config.model_name
            partition = 'validation'
            batch_size = 128
            shuffle = False
        else:
            model_name = model.config.model_name
            partition = 'testing'
            batch_size = 1
            shuffle = False

        #dataloader = create_dataloader(
        #    dataset_name=self.dataset_name,
        #    dataset_root=self.dataset_root,
        #    partition=partition,
        #    batch_size=batch_size,
        #    shuffle=shuffle
        #)
        if self.config.n_overlapping_sounds <= 2:
            base_valid_dataset = PrecomputedValDataset(
                preprocessed_dir=os.path.join(PREPROCESSED_ROOT, f'{self.config.mode}_seed{self.config.seed}', f'{self.config.mode}.pt'),
            )
            if self.config.num_val_samples and self.config.num_val_samples < len(base_valid_dataset):
                subset_indices = torch.arange(self.config.num_val_samples)
                base_valid_dataset = Subset(base_valid_dataset, subset_indices)
        else:
            if self.config.mode == 'val':
                val_audio_dir = Path(self.dataset_root).parent / 'validation'
            elif self.config.mode == 'test':
                val_audio_dir = Path(self.dataset_root).parent / 'test'
            else:
                raise ValueError(f"Invalid mode: {self.config.mode}. Must be 'val' or 'test'.")
            base_valid_dataset = DCase2019Dataset(
                audio_dir=str(val_audio_dir),
                n_overlapping_sounds=self.config.n_overlapping_sounds,
                chunk_size=2 * 16000,  # 2 seconds
                out_channels=self.config.in_channels,
                mode=self.config.mode,
                num_val_samples=self.config.num_val_samples,
                seed=self.config.seed,
                augmentations=None
            )

        valid_dataset = ChannelSliceDataset(
            base_valid_dataset,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
        )
        if self.config.zero_yzx:
            if self.config.in_channels < 4:
                raise ValueError("--zero-yzx requires in_channels >= 4.")
            valid_dataset = ZeroYZXInputDataset(valid_dataset)
        print(f"Number of validation samples: {len(valid_dataset)}")
        dataloader = DataLoader(
            valid_dataset, batch_size=self.config.valid_batch_size, shuffle=False,
            num_workers=8, drop_last=False, pin_memory=True, worker_init_fn=seed_worker
        )

        self.logger.info('using %d samples for evaluation', len(dataloader.dataset))

        trainer = get_trainer(model_name)(model=model)
        with torch.inference_mode():
            metrics, audio_samples = trainer.validate(dataloader)

        self.logger.info('[TEST] %s', metrics_to_str(metrics))

        torch.save(metrics, os.path.join(self.config.results_dir, BEST_METRICS_FILENAME))

        self.logger.info('completed')


class ZeroYZXInputDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        mixture, sources = self.base_dataset[idx]
        if mixture.size(0) < 4:
            raise ValueError("zero_yzx requested but mixture has fewer than 4 channels.")
        mixture = mixture.clone()
        mixture[1:4, :] = 0.0
        return mixture, sources


if __name__ == '__main__':
    configure_console_logger()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['train', 'val', 'test'], required=False, default='val')
    arg_parser.add_argument('--train-results-dir', type=str, required=True)
    arg_parser.add_argument('--eval-method', choices=['reference', 'reference-valid', 'blind'])
    arg_parser.add_argument('--eval-blind-num-repeat', type=int)
    arg_parser.add_argument('--seed', type=int)
    arg_parser.add_argument('--device-name', type=str)
    arg_parser.add_argument('--num-val-samples', type=int)
    arg_parser.add_argument('--in-channels', type=int)
    arg_parser.add_argument('--out-channels', type=int)
    arg_parser.add_argument('--n-overlapping-sounds', type=int, help='Number of overlapping sources per mixture (val).')
    arg_parser.add_argument('--zero-yzx', action='store_true', help='Zero YZX input channels for FOA ablation (keeps W).')

    cmd_args = arg_parser.parse_args()
    Test(**vars(cmd_args)).start()
