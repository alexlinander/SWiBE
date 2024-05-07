# SWiBE (Step-Wised Bandwidth Expansion)

## Training
```
python train.py --base_dir <dataset_dir> --save_dir <model_ckpt_dir>
```
Note: <dataset_dir> should be the directory containing subdirectories `train/`, `valid/` and `test/` which contains `clean/` and `noisy/` for each sibdirectory.

## Inference
```
python enhancement.py --test_dir <testset_dir> --enhanced_dir <enhanced_wav_dir> --ckpt <ckpt_dir>
```
Note: <testset_dir> should direct to `<dataset_dir>/test`