CUDA_VISIBLE_DEVICES=0 python run.py \
--model bioelectra \
--output_dir ./results \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 64 \
--warmup_steps 500 \
--weight_decay 0.01 \
--logging_dir ./logs \
--logging_steps 10