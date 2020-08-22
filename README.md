# Japanese sentence scoring with BERT

This is repository of using BERT to score Japanese sentence following [this paper](http://proceedings.mlr.press/v101/shin19a/shin19a.pdf).

To clone this repository together with the required [BERT](https://github.com/google-research/bert).

    git clone --recurse-submodules https://github.com/dangne/japanese-sentence-scoring-with-bert

The pretrained BERT and trained SentencePiece were fork from [yoheikikuta's repo](https://github.com/yoheikikuta/bert-japanese), please download all pretrained objects to `model/` directory.

- **[`Pretrained BERT model and trained SentencePiece model`](https://drive.google.com/drive/folders/1Zsm9DD40lrUVu6iAnIuTH2ODIkh-WM-O?usp=sharing)** 

You can run the below scripts with TPU, follow this [Colab notebook](https://colab.research.google.com/drive/1ZhV7PGJFyCp0drOLUGE32Vu3a9-t05zS?usp=sharing) for instructions.

## Create data for predicting masked token task

Convert raw text (.txt file) to .tfrecord format that BERT can understand

```
python3 src/create_pretraining_data.py \
	--input_file="./your_path_to_input_dir/*.txt" \
	--output_file="./your_path_to_output_dir/train.tfrecord" \
	--model_file=./model/wiki-ja.model \
	--vocab_file=./model/wiki-ja.vocab \
	--do_lower_case=True \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--random_seed=12345 \
	--dupe_factor=10 \
	--masked_lm_prob=0.15 \
	--short_seq_prob=0.1
```

**Remark:** to see the full list of parameters, visit the code of [`create_pretraining_data.py`](https://github.com/dangne/japanese-sentence-scoring-with-bert/blob/master/src/create_pretraining_data.py)



## Finetune on predicting masked token task

```
python3 src/run_pretraining.py \
    --input_file="./your_path_to_input_dir/predict.tfrecord" \
    --output_dir="./your_path_to_output_dir" \
    --init_checkpoint="./model/model.ckpt-1400000" \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --do_train=True \
    --do_eval=True \
    --do_predict=False \
    --train_batch_size=32 \
    --eval_batch_size=8 \
    --learning_rate=5e-5 \
    --num_train_steps=100000 \
    --num_warmup_steps=10000 \
    --save_checkpoints_steps=1000 \
    --iterations_per_loop=1000 \
    --max_eval_steps=100 \
    --use_tpu=False \
```

**Remark:** to see the full list of parameters, visit the code of [`run_pretraining.py`](https://github.com/dangne/japanese-sentence-scoring-with-bert/blob/master/src/run_pretraining.py)



## Predict masked token

```
python3 src/run_pretraining.py \
    --input_file="./your_path_to_input_dir/predict.tfrecord" \
    --output_dir="./your_path_to_output_dir" \
    --init_checkpoint="./model/model.ckpt-1400000" \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --predict_batch_size=8 \
    --iterations_per_loop=1000 \
    --max_eval_steps=100 \
    --use_tpu=False \
```

**Remark:** to see the full list of parameters, visit the code of [`run_pretraining.py`](https://github.com/dangne/japanese-sentence-scoring-with-bert/blob/master/src/run_pretraining.py)