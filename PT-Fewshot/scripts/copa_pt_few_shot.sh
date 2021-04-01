export CUDA_VISIBLE_DEVICES=0

python3 cli.py \
--data_dir PATH_TO_DATA_DIR/COPA \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name copa \
--output_dir PATH_TO_OUTPUT_DIR/copa \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 1 \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 96 \
--pet_max_steps 3500 \
--pattern_ids 1 \
