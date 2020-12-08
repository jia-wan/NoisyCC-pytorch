# Modeling Noisy Annotations for Crowd Counting

## Data preparation
The dataset can be constructed followed by [Bayesian Loss](https://github.com/ZhihengCV/Bayesian-Crowd-Counting).

## Pretrained model
The pretrained model can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1QQsLgch_kaazqs0WW7pshJy5qmUmoo8g?usp=sharing).

## Test

```
python test.py --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT
```

## Train

```
python train.py --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT
```

### Citation
If you use our code or models in your research, please cite with:

```
@article{wan2020modeling,
  title={Modeling Noisy Annotations for Crowd Counting},
  author={Wan, Jia and Chan, Antoni},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
