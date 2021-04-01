export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 cli.py \
  --data_dir PATH_TO_DATA_DIR/MultiRC \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v2 \
  --task_name multirc \
  --output_dir PATH_TO_OUTPUT_DIR/multirc \
  --do_eval \
  --do_train \
  --pet_per_gpu_eval_batch_size 8 \
  --pet_per_gpu_train_batch_size 2 \
  --pet_gradient_accumulation_steps 1 \
  --pet_max_seq_length 512 \
  --pet_max_steps 250 \
  --pattern_ids 1 \
  --learning_rate 1e-4
