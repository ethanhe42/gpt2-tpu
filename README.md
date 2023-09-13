# gpt2-tpu
pretraining gpt2 on TPUs



```bash
pip3 install -r requirements.txt

# optional
python3 -m wandb login

# choose single machine, TPU, 8 cores
python3 -m accelerate config

git clone https://github.com/huggingface/transformers
cd transformers/examples/pytorch/language-modeling/

export ALLOW_MULTIPLE_LIBTPU_LOAD=1
export PJRT_DEVICE=TPU
accelerate launch run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name Skylion007/openwebtext \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```


wikitext-2-raw-v1