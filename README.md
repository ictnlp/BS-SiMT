# BS-SiMT
Source code for our ACL 2023 paper ["Learning Optimal Policy for Simultaneous Machine Translation via Binary Search"](https://arxiv.org/pdf/2305.12774)

Our method is implemented based on the open-source toolkit [Fairseq](https://github.com/facebookresearch/fairseq) .

## Requirements and Installation

* Python version = 3.8

* PyTorch version = 1.10

* Install fairseq:

```
git clone https://github.com/ictnlp/BS-SiMT.git
cd BS-SiMT
pip install --editable ./
```

## Quick Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)), IWSLT14 German-English (download [here](https://wit3.fbk.eu/2014-01)).

For IWSLT14 German-English, we tokenize the corpus via [mosesdecoder/scripts/tokenizer/normalize-punctuation.perl](https://github.com/moses-smt/mosesdecoder) and apply BPE with 10K merge operations via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt).

Then, we process the data into the fairseq format, adding ```--joined-dictionary``` for IWSLT14 German-English:

```
SRC=source_language
TGT=target_language
TRAIN_DATA=path_to_training_data
VALID_DATA=path_to_valid_data
TEST_DATA=path_to_test_data
DATA=path_to_processed_data

# add --joined-dictionary for IWSLT14 German-English
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
    --trainpref ${TRAIN_DATA} --validpref ${VALID_DATA} \
    --testpref ${TEST_DATA}\
    --destdir ${DATA}
```

### Multi-Path Training

Get the base translation model via multi-path training.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATAFILE=dir_to_data
MODELFILE=dir_to_save_model

python train.py \
  ${DATAFILE} \
  --arch transformer_iwslt_de_en \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --lr 0.0005  \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 8192 \
  --update-freq 1 \
  --no-progress-bar \
  --log-format json \
  --left-pad-source False \
  --ddp-backend=no_c10d \
  --dropout 0.3 \
  --log-interval 100 \
  --multipath-training \
  --save-dir ${MODELFILE}
```

### Constructing Optimal Policy

Employ binary search to determine the ideal number of source tokens to be read for each target token.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATAFILE=dir_to_data
MODELFILE=dir_to_save_model
LEFTBOUND=1
RIGHTBOUND=5

python train.py \
  ${DATAFILE} \
  --arch transformer_iwslt_de_en \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --lr 0.0005  \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 8192 \
  --update-freq 1 \
  --no-progress-bar \
  --log-format json \
  --left-pad-source False \
  --ddp-backend=no_c10d \
  --dropout 0.3 \
  --log-interval 100 \
  --reset-dataloader \
  --reset-optimizer \
  --reset-meters \
  --reset-lr-scheduler \
  --left-bound ${LEFTBOUND} \
  --right-bound ${RIGHTBOUND} \
  --save-dir ${MODELFILE}
```

### Learning Optimal Policy

Let the agent the learn the optimal policy, which is obtained by the translation model.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATAFILE=dir_to_data
MODELFILE=dir_to_save_model
LEFTBOUND=1
RIGHTBOUND=5

python train.py \
  ${DATAFILE} \
  --arch transformer_iwslt_de_en \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --lr 0.0005  \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 8192 \
  --update-freq 1 \
  --no-progress-bar \
  --log-format json \
  --left-pad-source False \
  --ddp-backend=no_c10d \
  --dropout 0.3 \
  --log-interval 100 \
  --reset-dataloader \
  --reset-optimizer \
  --reset-meters \
  --reset-lr-scheduler \
  --left-bound ${LEFTBOUND} \
  --right-bound ${RIGHTBOUND} \
  --classerifier-training \
  --action-loss-smoothing 0.1 \
  --save-dir ${MODELFILE}
```

### Inference
Evaluate the model with the following command:

```
export CUDA_VISIBLE_DEVICES=0

MODELFILE=dir_to_save_model
DATAFILE=dir_to_data
REFERENCE=oath_to_reference

python generate.py ${MODELFILE} --path $MODELFILE/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --remove-bpe > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${REFERENCE} < pred.translation
```


## Citation
```
@inproceedings{guo-etal-2023-learning,
    title = "Learning Optimal Policy for Simultaneous Machine Translation via Binary Search",
    author = "Guo, Shoutao  and
      Zhang, Shaolei  and
      Feng, Yang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.130",
    pages = "2318--2333",
    abstract = "Simultaneous machine translation (SiMT) starts to output translation while reading the source sentence and needs a precise policy to decide when to output the generated translation. Therefore, the policy determines the number of source tokens read during the translation of each target token. However, it is difficult to learn a precise translation policy to achieve good latency-quality trade-offs, because there is no golden policy corresponding to parallel sentences as explicit supervision. In this paper, we present a new method for constructing the optimal policy online via binary search. By employing explicit supervision, our approach enables the SiMT model to learn the optimal policy, which can guide the model in completing the translation during inference. Experiments on four translation tasks show that our method can exceed strong baselines across all latency scenarios.",
}
```
