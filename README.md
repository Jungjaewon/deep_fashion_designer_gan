# Deep Fashion Designer GAN (DFD_GAN)
This repository have code for DFD GAN. This GAN takes four difference categories of fashion items and generates a suitable fashion item.

## Requirements
* python3.7+
* pytorch 1.7.0
* others.

## Usage
training a model
```bash
python3 main.py --config config.yml
```

testing a model
```bash
Not implmented yet
```

## Architecture
![architecture](img/overview.jpg)

## Tops Results
![outer_result](img/outer_fig_gen.jpg)
![inner_result](img/inner_fig_gen.jpg)

## Bottoms Results
![bottoms_result](img/bottoms_fig_gen.jpg)

## Accessories Results
![bags_result](img/bags_fig_gen.jpg)
![earrings_result](img/earrings_fig_gen.jpg)

## Refinement Results
![refine_result](img/refine_fig_6.jpg)

## Mapping Network Results
![mapping_result](img/tsne_result.jpg)

## Wrong Results
![wrong_result](img/wrong_fig.jpg)

## Comments
None for now.
## Reference
1. [Dataset](https://github.com/xthan/polyvore)
2. [Spectral Normalization](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py)
3. [U-net](https://github.com/milesial/Pytorch-UNet)
4. [FINCH Clustering](https://github.com/ssarfraz/FINCH-Clustering)
5. [BIT](https://github.com/google-research/big_transfer)