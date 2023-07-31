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