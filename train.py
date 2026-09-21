import os
# Deterministic training
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import argparse
from time import time
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import wandb

#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.utils import DATASET_PATH, PREPROCESSED_ROOT, CONFIG_FILENAME, BEST_CHECKPOINT_FILENAME, METRICS_HISTORY_FILENAME, configure_console_logger, \
    default, build_run_name, ensure_clean_results_dir, setup_determinism, get_logger, total_num_params
from lib.data.dataloader_utils import get_dataset_specs, create_dataloader
from lib.trainers import get_trainer, make_warmup_cosine_scheduler
from lib.models import Model
from torch.optim.lr_scheduler import LambdaLR
from dcasedataset import DCase2019Dataset, SoundAugmenter, PrecomputedValDataset, ChannelSliceDataset, seed_worker

class Training:
    def __init__(self, train_results_root, librimix_root=None, realm_root=None, dcase_root=None,
                 stft_frame_size=None,
                 stft_hop_size=None,
                 model_name=None,
                 model_load_path=None,
                 mixcycle_init_epochs=None,
                 snr_max=None,
                 train_batch_size=None,
                 valid_batch_size=None,
                 lr=None,
                 grad_clip=None,
                 train_subsample_ratio=None,
                 valid_subsample_ratio=None,
                 eval_method=None,
                 eval_blind_num_repeat=None,
                 eval_epochs=None,
                 patience=None,
                 min_epochs=None,
                 seed=None,
                 use_wandb=None,
                 run_name=None,
                 run_id=None,
                 num_val_samples=None,
                 num_train_samples=None,
                 device_name=None,
                 enable_augmentation=None,
                 augmentations=None,
                 mixit_head_multiplier=None,
                 use_scheduler=None,
                 scheduler_warmup_ratio=None,
                 desired_total_epochs=None,
                 continue_after_desired_total_epochs=None,
                 scheduler_final_lr=None,
                 scheduler_start_lr=None,
                 weight_decay=None,
                 in_channels=None,
                 out_channels=None,
                 w_out_mode=None,
                 num_workers=None,
                 n_overlapping_sounds=None,
                 use_intensity_features=None,
                 mixit_rotate=None,
                 mixit_rotate_prob=None,
                 use_scalar_steering=None,
                 scalar_steering_hidden_dim=None,
                 scalar_steering_detach_mask=None,
                 ):
        args = locals()
        del args['self']

        self.dataset_name, self.dataset_root = 'dcase19', DATASET_PATH
        if run_name is None:
            run_name = build_run_name(
                args=args,
                prepend_items={'train': self.dataset_name},
                exclude_keys=['train_results_root', 'realm_root', 'dcase_root', 'librimix_root', 'model_load_path', 'run_name']
            )

        self.results_dir = os.path.join(train_results_root, run_name)

        self.config = SimpleNamespace()
        self.config.run_name = run_name
        self.config.results_dir = default(self.results_dir, "results")
        self.config.stft_frame_size = default(stft_frame_size, 512)
        self.config.stft_hop_size = default(stft_hop_size, 128)
        self.config.model_name = default(model_name, 'mixit')
        self.config.model_load_path = default(model_load_path, None)
        self.config.mixcycle_init_epochs = default(mixcycle_init_epochs, 0)
        self.config.snr_max = default(snr_max, 30.0)
        self.config.train_batch_size = default(train_batch_size, 32)
        self.config.valid_batch_size = default(valid_batch_size, 32)
        self.config.lr = default(lr, 0.001)
        self.config.grad_clip = default(grad_clip, 5.0)
        self.config.train_subsample_ratio = default(train_subsample_ratio, 1.0)
        self.config.valid_subsample_ratio = default(valid_subsample_ratio, 1.0)
        self.config.eval_method = default(eval_method, 'reference')
        self.config.eval_blind_num_repeat = default(eval_blind_num_repeat, 1)
        self.config.eval_epochs = default(eval_epochs, 1)
        self.config.patience = default(patience, 6)
        self.config.min_epochs = default(min_epochs, 20)
        self.config.seed = default(seed, 42)
        self.config.device_name = default(device_name, 'cuda')
        self.config.device = 'cuda'
        self.config.use_wandb = default(use_wandb, True)
        self.config.num_val_samples = default(num_val_samples, 1024)
        self.config.num_train_samples = default(num_train_samples, 300000)
        self.config.enable_augmentation = default(enable_augmentation, False)
        self.config.augmentations = default(augmentations, "")
        self.config.head_multiplier = default(mixit_head_multiplier, 2)
        self.config.use_scheduler = default(use_scheduler, False)
        self.config.continue_after_desired_total_epochs = default(continue_after_desired_total_epochs, False)
        self.config.scheduler_warmup_ratio = default(scheduler_warmup_ratio, 0.05)
        self.config.desired_total_epochs = default(desired_total_epochs, 26)
        self.config.scheduler_final_lr = default(scheduler_final_lr, 1e-6)
        self.config.scheduler_start_lr = default(scheduler_start_lr, 3e-6)
        self.config.weight_decay = default(weight_decay, 0.0)
        self.config.in_channels  = default(in_channels, 1)
        self.config.out_channels = default(out_channels, self.config.in_channels)
        self.config.w_out_mode   = default(w_out_mode, False)
        self.config.num_workers = default(num_workers, 8)
        self.config.n_overlapping_sounds = default(n_overlapping_sounds, 2)
        self.config.use_intensity_features = default(use_intensity_features, False)
        self.config.mixit_rotate = default(mixit_rotate, False)
        self.config.mixit_rotate_prob = default(mixit_rotate_prob, 1.0)
        self.config.use_scalar_steering = default(use_scalar_steering, False)
        self.config.scalar_steering_hidden_dim = default(scalar_steering_hidden_dim, 128)
        self.config.scalar_steering_detach_mask = default(scalar_steering_detach_mask, True)
        if self.config.use_intensity_features and self.config.in_channels != 4:
            self.config.use_intensity_features = False
        if self.config.mixit_rotate and self.config.in_channels != 4:
            raise ValueError("--mixit-rotate requires --in-channels 4.")
        if self.config.use_scalar_steering and (self.config.in_channels != 4 or self.config.out_channels != 4):
            raise ValueError("--use-scalar-steering requires --in-channels 4 and --out-channels 4.")
        if self.config.in_channels > 1 and self.config.out_channels == 1 and not self.config.w_out_mode:
            raise ValueError("out_channels=1 with in_channels>1 requires --w_out_mode until downmix is implemented.")

    def start(self):
        ensure_clean_results_dir(self.config.results_dir)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name())


        self.logger = get_logger('train', self.config.results_dir)
        self.logger.info('config: %s', self.config)
        torch.save(self.config, os.path.join(self.config.results_dir, CONFIG_FILENAME))

        setup_determinism(self.config.seed)

        self.config.device = torch.device(self.config.device_name)
        #self.tensorboard = SummaryWriter(os.path.join(self.config.results_dir, 'tb'))

        self.cur_epoch = 0
        self.cur_step = 0
        self.cur_patience = self.config.patience
        self.last_best = {}
        self.metrics_history = []

        checkpoint_path = os.path.join(self.config.results_dir, BEST_CHECKPOINT_FILENAME)

        if self.config.enable_augmentation:
            self.logger.info("Data augmentation enabled")
            enable_flip = "flip" in self.config.augmentations
            enable_jitter = "jitter" in self.config.augmentations
            enable_noise = "noise" in self.config.augmentations
            enable_rotation = "rotate" in self.config.augmentations
            if enable_rotation and self.config.in_channels != 4:
                self.logger.warning("rotate augmentation requested with in_channels=%d; disabling.", self.config.in_channels)
                enable_rotation = False
            print("Data augmentation settings: flip={}, jitter={}, noise={}, rotate={}".format(
                enable_flip, enable_jitter, enable_noise, enable_rotation
            ))
            augmenter = SoundAugmenter(
                enable_flip=enable_flip,
                enable_jitter=enable_jitter,
                enable_noise=enable_noise,
                enable_rotation=enable_rotation,
                add_noise_prob=0.3,
                noise_snr_db=(25.0, 45.0),
            )
        else:
            augmenter = None

        self.logger.info(f"Creating training dataset...")
        train_dataset = DCase2019Dataset(
            audio_dir=self.dataset_root,
            n_overlapping_sounds=self.config.n_overlapping_sounds,
            chunk_size=2 * 16000,  # 2 seconds
            out_channels=self.config.in_channels,
            mode='train',
            seed=self.config.seed,
            augmentations=augmenter
        )
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.config.train_batch_size, shuffle=True,
            num_workers=self.config.num_workers, drop_last=True, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker,
        )
        if self.config.n_overlapping_sounds <= 2:
            base_valid_dataset = PrecomputedValDataset(
                preprocessed_dir=os.path.join(PREPROCESSED_ROOT, 'val_seed' + f"{self.config.seed}", 'val.pt'),
            )
            if self.config.num_val_samples and self.config.num_val_samples < len(base_valid_dataset):
                subset_indices = torch.arange(self.config.num_val_samples)
                base_valid_dataset = Subset(base_valid_dataset, subset_indices)
            val_in_memory = True
        else:
            val_audio_dir = Path(self.dataset_root).parent / 'validation'
            base_valid_dataset = DCase2019Dataset(
                audio_dir=str(val_audio_dir),
                n_overlapping_sounds=self.config.n_overlapping_sounds,
                chunk_size=2 * 16000,  # 2 seconds
                out_channels=self.config.in_channels,
                mode='val',
                num_val_samples=self.config.num_val_samples,
                seed=self.config.seed,
                augmentations=None
            )
            val_in_memory = False

        valid_dataset = ChannelSliceDataset(
            base_valid_dataset,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
        )
        #valid_dataset = DCase2019Dataset(
        #    audio_dir='./datasets/dcase2019/validation',
        #    n_overlapping_sounds=2,
        #    chunk_size=2 * 16000,  # 2 seconds
        #    out_channels=self.config.use_n_channels,
        #    mode='val',
        #    num_val_samples=self.config.num_val_samples, 
        #    seed=self.config.seed
        #)
        print(f"Number of validation samples: {len(valid_dataset)}")
        # PrecomputedValDataset is fully on RAM
        val_num_workers = 0 if val_in_memory else self.config.num_workers
        val_persistent = val_num_workers > 0
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.config.valid_batch_size, shuffle=False,
            num_workers=val_num_workers, drop_last=False, pin_memory=True,
            persistent_workers=val_persistent,
            worker_init_fn=seed_worker if val_num_workers > 0 else None,
        )

        mixture_batch, sources_batch = next(iter(self.train_dataloader))
        self.config.num_sources = sources_batch.size(1)
        assert self.config.in_channels == mixture_batch.size(1), "in_channels must match the data."
        assert self.config.out_channels in {1, 4}, "out_channels must be 1 or 4 for these experiments."
        self.config.sample_length = mixture_batch.size(2)
        if self.config.num_train_samples and self.config.num_train_samples < len(self.train_dataloader.dataset):
           self.config.num_batches = self.config.num_train_samples // self.config.train_batch_size
           print(f"Epoch length limited to {self.config.num_batches} batches")
        else:
            self.config.num_batches = len(self.train_dataloader)

        self.config.num_frequency_bins = 1 + self.config.stft_frame_size // 2

        if self.config.use_wandb:
            wandb.init(
                project='sound_source_separation',
                name=self.config.run_name,
                config=self.config
            )

        if self.config.model_load_path:
            self.logger.info(f"Loading checkpoint from: {self.config.model_load_path}")
            model = Model.load(
                path=os.path.join(self.config.model_load_path, BEST_CHECKPOINT_FILENAME),
                device=self.config.device
            ).train()
        else:
            model = None

        trainer_cls = get_trainer(self.config.model_name)
        max_underlying_batches = self.config.num_batches if self.config.num_train_samples else None
        self.trainer = trainer_cls(config=self.config, model=model, max_underlying_batches=max_underlying_batches)

        if self.config.use_scheduler:
            # Calculate steps per epoch for scheduler
            if max_underlying_batches is not None:
                steps_per_epoch = max_underlying_batches // 2
            else:
                steps_per_epoch = len(self.train_dataloader) // 2

            total_desired_steps = self.config.desired_total_epochs * steps_per_epoch

            self.trainer.scheduler, warmup_steps = make_warmup_cosine_scheduler(self.trainer.optimizer, total_steps=total_desired_steps, warmup_ratio=self.config.scheduler_warmup_ratio, base_lr=self.config.lr, start_lr=self.config.scheduler_start_lr, min_lr=self.config.scheduler_final_lr)
            self.logger.info(f"Using learning rate scheduler with warmup of {warmup_steps} steps and cosine annealing over {total_desired_steps} steps.")
            self.logger.info(
                f"LR scheduler initialized | "
                f"base_lr={self.config.lr:.2e}, "
                f"start_lr={self.config.scheduler_start_lr:.2e}, "
                f"min_lr={self.config.scheduler_final_lr:.2e}, "
                f"warmup_steps={warmup_steps}, "
                f"total_steps={total_desired_steps}"
            )
            self.logger.info(f"steps_per_epoch={steps_per_epoch} (len(dataloader)={len(self.train_dataloader)}, "f"max_underlying_batches={max_underlying_batches})")

        if self.config.use_wandb:
            self.logger.info("Watching model with wandb")
            #wandb.watch(
            #    models=self.trainer.get_model(),
            #    log='all',
            #    log_freq=self.config.num_batches, # Log every epoch
            #    log_graph=False
            #)

        self.logger.info('model parameter count: %d', total_num_params(self.trainer.get_model().parameters()))
        self._train_loop()
        self.logger.info('completed')

    def _train_loop(self):
        total_steps = self.config.eval_epochs * self.config.num_batches
        
        self.logger.info(f"Starting training for at least {self.config.min_epochs} epochs and {total_steps} steps")
        while self.cur_patience > 0:
            if not self.config.continue_after_desired_total_epochs and self.cur_epoch >= self.config.desired_total_epochs:
                self.logger.info(f"Reached desired_total_epochs={self.config.desired_total_epochs}. Stopping.")
                break
            self.train_start = time()
            start_step = self.trainer.global_step
            for _ in range(self.config.eval_epochs):
                steps = self.trainer.train(self.train_dataloader)
                with tqdm(steps, total=self.config.num_batches, leave=False) as progress_bar:
                    for increment in steps:
                        self.cur_step += increment
                        progress_bar.update(increment)
                self.logger.info(f"Epoch {self.cur_epoch} completed in {time() - self.train_start:.1f} seconds")
                if self.config.use_scheduler:
                    steps_this_epoch = self.trainer.global_step - start_step
                    self.logger.info(f"Optimizer steps this epoch: {steps_this_epoch}")
                    self.logger.info(f"LR after epoch {self.cur_epoch}: {self.trainer.optimizer.param_groups[0]['lr']:.2e}")
                self.cur_epoch += 1
            self._validate()
                

    def _validate(self):
        self.logger.info("Starting validation...")
        metrics = {
            'process': {
                'epoch': self.cur_epoch,
                'step': self.cur_step,
                'train_elapsed': time() - self.train_start,
                'train_loss': self.trainer.get_loss()
            }
        }
        validate_start = time()
        with torch.inference_mode():
            validation_results, audio_samples = self.trainer.validate(self.valid_dataloader)
            metrics['validation'] = validation_results
        metrics['process']['validate_elapsed'] = time() - validate_start

        #if self.config.use_wandb and audio_samples:
        #    wandb.log({
        #        'validation/audio_samples': wandb.Audio(
        #            audio_samples,
        #            sample_rate=16000,
        #            caption='Validation Audio Samples'
        #        )
        #    })

        self._update_best(metrics)
        self._update_patience(metrics)
        self._report(metrics)
        self.trainer.loss_accumulator.reset()

    def _update_best(self, metrics):
        metrics['best'] = {}
        for key, value in metrics['validation'].items():
            if key not in self.last_best or value > self.last_best[key]:
                self.last_best[key] = value
                metrics['best'][key] = True
            else:
                metrics['best'][key] = False
        return metrics

    def _update_patience(self, metrics):
        min_epoch = self.config.min_epochs and self.config.min_epochs > self.cur_epoch

        if self.config.eval_method == 'reference':
            main_metric_name = 'sisnri'
        elif self.config.eval_method == 'blind':
            main_metric_name = 'sisnri_blind'
        else:
            raise Exception(f'unknown eval_method: {self.config.eval_method}')

        is_main_best = metrics['best'][main_metric_name]

        if is_main_best:
            self.trainer.get_model().save(os.path.join(self.config.results_dir, BEST_CHECKPOINT_FILENAME))

        if min_epoch or is_main_best:
            self.cur_patience = self.config.patience
        else:
            self.cur_patience -= 1
        metrics['process']['patience'] = self.cur_patience

    def _report(self, metrics):
        log_line = ''
        for group in ['process', 'validation']:
            for key, value in metrics[group].items():
                format_spec = '{}={:.3f}' if isinstance(value, float) else '{}={}'
                log_line += format_spec.format(key, value)

                if group == 'validation' and metrics['best'][key]:
                    log_line += '*'
                else:
                    log_line += ' '

                if group == 'validation':
                    log_line += ' '

                #self.tensorboard.add_scalar(
                #    tag='{}/{}'.format(group, key),
                #    scalar_value=value,
                #    global_step=metrics['process']['step']
                #)
            if group == 'process':
                log_line += '| '

        self.logger.info(log_line)

        if self.config.use_wandb:
            wandb_metrics = {}
            for group, group_metrics in metrics.items():
                if group == 'best': continue
                for key, value in group_metrics.items():
                    wandb_metrics[f"{group}/{key}"] = value
            wandb_metrics['epoch'] = metrics['process']['epoch']
            wandb.log(wandb_metrics)
        
        #self.tensorboard.flush()

        metrics['time'] = time()
        self.metrics_history.append(metrics)
        metrics_history_path = os.path.join(self.config.results_dir, METRICS_HISTORY_FILENAME)
        os.makedirs(os.path.dirname(metrics_history_path), exist_ok=True)
        torch.save(self.metrics_history, metrics_history_path)


if __name__ == '__main__':
    configure_console_logger()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train-results-root', type=str, required=True)
    arg_parser.add_argument('--librimix-root', type=str)
    arg_parser.add_argument('--realm-root', type=str)
    arg_parser.add_argument('--dcase-root', type=str)
    arg_parser.add_argument('--stft-frame-size', type=int)
    arg_parser.add_argument('--stft-hop-size', type=int)
    arg_parser.add_argument('--model-name', choices=['mixit'])
    arg_parser.add_argument('--mixcycle-init-epochs', type=int)
    arg_parser.add_argument('--snr-max', type=float)
    arg_parser.add_argument('--train-batch-size', type=int)
    arg_parser.add_argument('--valid-batch-size', type=int)
    arg_parser.add_argument('--lr', type=float)
    arg_parser.add_argument('--grad-clip', type=float)
    arg_parser.add_argument('--train-subsample-ratio', type=float)
    arg_parser.add_argument('--valid-subsample-ratio', type=float)
    arg_parser.add_argument('--eval-method', choices=['reference', 'blind'])
    arg_parser.add_argument('--eval-epochs', type=int)
    arg_parser.add_argument('--patience', type=int)
    arg_parser.add_argument('--seed', type=int)
    arg_parser.add_argument('--run-name', type=str, help='Optional short name to use for the results directory and W&B run.')
    arg_parser.add_argument('--run-id', type=str)
    arg_parser.add_argument('--device-name', type=str)
    arg_parser.add_argument('--num-val-samples', type=int)
    arg_parser.add_argument('--num-train-samples', type=int)
    arg_parser.add_argument('--model-load-path', type=str, default=None,
                            help='Path to load the model from. If provided, training will resume from this checkpoint.')
    arg_parser.add_argument('--use-wandb', type=bool)
    arg_parser.add_argument('--enable-augmentation', type=bool)
    arg_parser.add_argument('--augmentations', type=str, help='Comma-separated list of augmentations to apply. Options: flip, jitter, noise, rotate (FOA-only)')
    arg_parser.add_argument('--mixit-head-multiplier', type=int, help='Multiplier for the number of output heads in MixIT models.')
    arg_parser.add_argument('--use-scheduler', action='store_true', help='If set to True, use learning rate cosine scheduler with warmup.')
    arg_parser.add_argument('--scheduler-warmup-ratio', type=float, help='Ratio of warmup steps for the learning rate scheduler.')
    arg_parser.add_argument('--desired-total-epochs', type=int, help='Desired total number of epochs for training (used for scheduler calculation).')
    arg_parser.add_argument('--continue-after-desired-total-epochs', action='store_true', help='If set to True, training will continue after reaching the desired total epochs.')
    arg_parser.add_argument('--scheduler-final-lr', type=float, help='Final learning rate for the cosine annealing scheduler.')
    arg_parser.add_argument('--scheduler-start-lr', type=float, help='Starting learning rate for the warmup scheduler.')
    arg_parser.add_argument('--weight-decay', type=float, help='Weight decay (L2 regularization) factor for the optimizer.')
    arg_parser.add_argument('--in-channels', type=int)
    arg_parser.add_argument('--out-channels', type=int)
    arg_parser.add_argument('--w_out_mode', action='store_true', help='If set, we train/eval with FOA input but W-only targets and losses.')
    arg_parser.add_argument('--num-workers', type=int, help='Number of worker threads for data loading.')
    arg_parser.add_argument('--n-overlapping-sounds', type=int, help='Number of overlapping sources per mixture (train/val).')
    arg_parser.add_argument('--use-intensity-features', dest='use_intensity_features', action='store_true', default=None,
                            help='Stack per-TF active-intensity features ([weighted unit-I, reliability]) onto the mask-generator input. FOA only; off by default.')
    arg_parser.add_argument('--no-intensity-features', dest='use_intensity_features', action='store_false', default=None,
                            help='Force the mask generator to use magnitude-only features even with FOA input.')
    arg_parser.add_argument('--mixit-rotate', action='store_true',
                            help='Apply independent random SO(3) rotations to each FOA mixture before forming the MoM in MixIT training.')
    arg_parser.add_argument('--mixit-rotate-prob', type=float,
                            help='Probability per pair of applying the rotation in rotation-MixIT (default 1.0).')
    arg_parser.add_argument('--use-scalar-steering', action='store_true',
                            help="Predict a mono dry source per output head plus 4 ambisonic gains. Forces a rank-1 (point-source) FOA model. Requires in_channels=out_channels=4.")
    arg_parser.add_argument('--scalar-steering-hidden-dim', type=int,
                            help='Hidden width of the scalar prediction head (default 128).')
    arg_parser.add_argument('--scalar-steering-detach-mask', dest='scalar_steering_detach_mask', action='store_true', default=None,
                            help='Detach the mono waveform on the YZX branch so YZX reconstruction loss only updates the scalar predictor (default).')
    arg_parser.add_argument('--no-scalar-steering-detach-mask', dest='scalar_steering_detach_mask', action='store_false', default=None,
                            help='Let YZX reconstruction loss flow into the mono mask too (couples direction learning back to separation).')

    cmd_args = arg_parser.parse_args()
    Training(**vars(cmd_args)).start()
