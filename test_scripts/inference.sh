MODELFILE=dir_to_save_model
DATAFILE=dir_to_data

python generate.py ${data} --path $MODELFILE/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --remove-bpe > pred.out