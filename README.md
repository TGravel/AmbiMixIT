# AmbiMixIT
Repository for Unsupervised Ambisonic Source Separation via Mixture-Invariant Training. Code for the main models and archived experimental results.

TODO: Include plot and maybe 2-3 audios

## Source
The original author of this MixIT (MixCycle) implementation is Ertuğ Karamatlı [MixCycle Repository](https://github.com/ertug/MixCycle). The PyTorch MixIT method has been adapted for Ambisonic Audio.

#### Source papers:

MixCycle paper:
```BibTex
@article{karamatli2022unsupervised,
  title={MixCycle: Unsupervised Speech Separation via Cyclic Mixture Permutation Invariant Training},
  author={Karamatl{\i}, Ertu{\u{g}} and K{\i}rb{\i}z, Serap},
  journal={IEEE Signal Processing Letters},
  volume={29},
  number={},
  pages={2637-2641},
  year={2022},
  doi={10.1109/LSP.2022.3232276}
}
```


MixIT paper:
```BibTex
@inproceedings{wisdom2022mixit,
 author = {Wisdom, Scott and Tzinis, Efthymios and Erdogan, Hakan and Weiss, Ron and Wilson, Kevin and Hershey, John},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
 pages = {3846--3857},
 publisher = {Curran Associates, Inc.},
 title = {Unsupervised Sound Separation Using Mixture Invariant Training},
 url = {https://proceedings.neurips.cc/paper_files/paper/2020/file/28538c394c36e4d5ea8ff5ad60562a93-Paper.pdf},
 volume = {33},
 year = {2020}
}
```


DCase 2019 Dataset (TAU Spatial Sound Events 2019 Ambisonic dataset):
```BibTex
@inproceedings{adavanne2019dataset,
title = {A Multi-room Reverberant Dataset for Sound Event Localization and Detection},
author = {Sharath Adavanne and Archontis Politis and Tuomas Virtanen},
year = {2019},
pages = {10--14},
booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019)},
doi = {10.5281/zenodo.4064792}
}
```

# Instructions:
Used dependencies:
```
python 3.12.5
numpy 2.2.6
torch 2.8.0.dev20250504+cu128
torchaudio 2.6.0.dev20250505+cu128
pandas 2.2.2
tqdm
wandb
torch_mir_eval
```
Cuda 12.8 strongly recommended!

### Dataset
The dataset used in the experiments can be downloaded [here](https://doi.org/10.5281/zenodo.2599196), the separate test split [here](https://doi.org/10.5281/zenodo.3377088). The dataset can be downsampled to 16kHz using the provided "resample16kHz.py". The dcasedataset.py script is used to extract individual ambisonic sounds and mix them together for MixIT.

The files need to be arranged in the following way:

`datasets/dcase2019_16kHz`
            ↳ `train`
                ↳ `split1_ir0_ov1_1.csv` ... `split3_ir4_ov2_100.wav` (must include .wav and .csv filed from split 1-3)
            ↳ validation
                ↳ `split4_ir0_ov1_1.csv` ... `split4_ir4_ov2_100.wav` (Split 4 was used for validation please copy and remove it from the train folder)
            ↳ test
                ↳ `split0_ir0_ov1_1.csv` ... `split0_ir4_ov2_100.wav` (available via test download)


Please edit the lib/utils.py file constants instead of using the arguments to set dataset paths for training. (lines 9-15)


Most important are:

```python
DATASET_PATH = r'./datasets/dcase2019_16kHz/train'
PREPROCESSED_ROOT = r'F:\Datasets\dcase2019\preprocessed'
```

Preprocessed datasets can be generated using the preprocess_dataset.py. The structure must be `preprocessed_path\val_seed<seeds>\val.pt` for a validation set for example.
The current code should be able to fully reproduce the THESIS_EVAL experiments 1, 2, 3, 6 and 8. Feel free to contact me if you require assistance to get the code working.

### Start a run:
To start training an Ambisonic MixIT model execute something similar to (used seeds were 42, 3206, 5858):

`python train.py --train-results-root results_seed_42 --run-name foa_baseline --model-name mixit --lr 0.0001 --patience 30 --in-channels 4 --out-channels 4 --seed 42`

`python train.py --train-results-root results_seed_42 --run-name intensity_features  --model-name mixit --lr 0.0001 --patience 30 --in-channels 4 --out-channels 4 --use-intensity-features --seed 42`

Mono baseline would be:

`python train.py --train-results-root results_seed_42 --run-name mono_baseline --model-name mixit --lr 0.0001 --patience 30 --in-channels 1 --out-channels 1 --seed 42`

To test a finished run:

`python test.py --train-results-dir THESIS_EVAL\8_final_multiseed\42_intensity_features --in-channels 4 --out-channels 4 --seed 42 --mode test`


# Acknowledgement

The code is an edit of the original [MixCycle](https://github.com/ertug/MixCycle) implementation by Ertuğ Karamatlı.