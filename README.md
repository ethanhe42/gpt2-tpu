# gpt2-tpu

pretraining gpt2 on TPUs. Verified on TPUv3-8

![image](https://github.com/yihui-he/gpt2-tpu/assets/10027339/2859929b-1340-4190-92dc-3c183f7a8b81)

[full log](https://wandb.ai/yihuihe/tpu-research/runs/8a33wa61?workspace=user-yihuihe)



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
export WANDB_PROJECT='tpu-research'
accelerate launch run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name Skylion007/openwebtext \
    --learning_rate 6e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 1e-1 \
    --warmup_steps 2000 \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 14 \
    --do_train \
    --do_eval \
    --preprocessing_num_workers 96 \
    --save_steps 0.01 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --eval_steps 0.01 \
    --bf16 True \
    --output_dir /tmp/test-clm
```


wikitext-2-raw-v1
