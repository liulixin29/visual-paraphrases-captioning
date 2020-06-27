# Generating Diverse and Descriptive Image Captions Using Visual Paraphrases

This repository includes the code of the ICCV 2019 paper [**Generating Diverse and Descriptive Image Captions Using Visual Paraphrases**](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Generating_Diverse_and_Descriptive_Image_Captions_Using_Visual_Paraphrases_ICCV_2019_paper.pdf).

We refer to our model as **DCVP** (**D**iverse and descriptive **C**aptioning using **V**isual **P**araphrases).

Our code is based on Ruotian Luo's implementation of [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563), available [here](https://github.com/ruotianluo/self-critical.pytorch). Please refer to the project for more details.

## Requirements
Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3)

PyTorch 0.4 (along with torchvision, it seems to be OK with other versions like PyTorch 1.0)

coco-caption, PIL, h5py

## Train your own network on COCO

### Download COCO captions and preprocess them

Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then do:

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data.json --output_h5 data
```

`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data.json` and discretized caption data are dumped into `data_label.h5`.

### Download COCO dataset (Used for evaluation as VSE++ requires raw images)

Download the coco images from [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the `train2014/` and `val2014/` in the same directory, denoted as `$IMAGE_ROOT`.

### Download Bottom-up features

Download pre-extracted feature from [link](https://github.com/peteanderson80/bottom-up-attention). You can either download adaptive one (used in our paper) or fixed one.

For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip

```

Then:

```bash
python script/make_bu_data.py --output_dir data/cocobu
```

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`.

### Build Visual Paraphrase Pairs

We provide h5 file for visual paraphrase pairs using *len(0)*, *yngve(0)*, *Tdiv(0.1, 0.3)* and *IR(2)* scoring functions.
Please refer to our paper for the details to select visual paraphrase pairs.
The visual paraphrase pairs h5 file should have the same keys as the h5 file of preprocessing labels (`data_label.h5`). And values will be an HDF5 dataset from numpy arrays in shape (|P|, 2), where |P| is the number of valid paraphrase pairs for the specific image and scoring function. If one pair is [0, 3], it means the first caption (numbered 0) and the fourth caption (numbered 3) of the image constitutes a valid paraphrase pair. The order of captions should follow the order in  `data_label.h5`.
### Download Files

[Download Link](https://pan.baidu.com/s/1COUxtWUMwSe4bOQwl2uvSw) on BaiduYun (password: yppa ).

Download visual paraphrase pairs h5 file (in folder `visual_paraphrase_pairs/`), `df_processed.pkl`(document frequency file), and text-image retrieval model from [VSE++](https://github.com/fartashf/vsepp) for evaluation on R@K metric.

We also provide preprocessed captions used for the paper.

Folder `self_retrieval_model/` contains the cross-modal retrieval model used for IR scoring function.

We provide model checkpoints of *DCVP(Tdiv, 0.1)*, *DCVP(Tdiv, 0.3)* and *DCVP(IR, 2)*, and model output json results on Karpathy test set.

If you only need the evaluation code, please download `evaluation.zip`.


### Start training

```bash
$ python train_pair.py --id tdiv_0.1 --caption_model att2in2p --pair_h5 <PAIR_H5_PATH> --batch_size 16 --learning_rate 5e-4 --checkpoint_path cp_tdiv_0.1 --save_checkpoint_every 6000 --max_epochs 30
```
`<PAIR_H5_PATH>` is the path of h5 file containing information of visual paraphrase pairs, such as `pair_tdiv_0.1.h5`.

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `save/`). We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

If you have tensorflow, the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

If you'd like to evaluate BLEU/CIDEr/SPICE scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into the directory.

For more options, see `opts.py`. Please refer to [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) repo for more details.


## Generate image captions

### Evaluate on Karpathy test split

```bash
$ python eval.py  --model cp_tdiv_0.1/model-best.pth --infos_path cp_tdiv_0.1/infos_tdiv_0.1.pkl --beam_size 3
```

The default split to evaluate is test. The default inference method is greedy decoding (`--sample_max 1`), to sample from the posterior, set `--sample_max 0`.

**Beam Search**. Beam search can increase the performance of the search for greedy decoding sequence by ~5%. However, this is a little more expensive. To turn on the beam search, use `--beam_size N`, N should be greater than 1.

In default, the code will evaluate conventional automatic metrics results of final captions (from the second decoder) and preliminary captions (from the first decoder) generated from the two-stage decoder.


### Evaluate on automatic metrics

If you have model output results in json, please use `eval_diversity.py` and `eval_retrieval.py` from this repo or `evaluate.zip` from download link above to evaluate results on *Dist-N* metric for caption diversity and *R@K* metric for caption descriptiveness(distinctness, discriminability).

The evaluation of *R@K* relies on raw images from Karpathy test set and a pretrained cross-model retrieval model (We use VSE++ with fine-tuned ResNet-152. Other state-of-the-art retrieval models such as [ViLBERT-multi-task](https://github.com/facebookresearch/vilbert-multi-task) can also be used).



## Reference

If you find our work or this repo useful, please consider citing:

```
@InProceedings{Liu_2019_ICCV,
author = {Liu, Lixin and Tang, Jiajun and Wan, Xiaojun and Guo, Zongming},
title = {Generating Diverse and Descriptive Image Captions Using Visual Paraphrases},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

## Acknowledgements

Thanks Ruotian Luo for his [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and [DiscCaptioning](https://https://github.com/ruotianluo/DiscCaptioning) repositories.
