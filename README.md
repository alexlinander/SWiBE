# SWiBE (Step-Wised Bandwidth Expansion)

## Preprocessing
To prepare the low sampling rate auido for training, run  the command below
```
python src/util/downsample.py --src_dir <16k audio directory> --tgt_dir <8k audio directory>
```


## Training
```
python train.py --base_dir <dataset_dir> --save_dir <model ckpt dir>
```
Note: <dataset_dir> should be the directory containing subdirectories `train/`, `valid/` and `test/` which contains `clean/` and `noisy/` for each sibdirectory as shown below. The `clean/` folders contain the clean 16k speech files, while the `noisy/` folders contain the noisy 8k speech files.

```
base_dir
    |-- train
        |-- clean
        |-- noisy
    |-- valid
        |-- clean
        |-- noisy
    |-- test
        |-- clean
        |-- noisy
```

## Inference
```
python enhancement.py --test_dir <testset_dir> --enhanced_dir <enhanced_wav_dir> --ckpt <ckpt_dir>
```
Note: <testset_dir> should direct to `<dataset_dir>/test`

## Acknowledgement
We would like to thank the authors and contributors of the repository [sgmse](https://github.com/sp-uhh/sgmse), whose code and resources significantly helped in the development of this project.