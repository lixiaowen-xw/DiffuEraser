validation_image="['data/eval/DAVIS/JPEGImages/480p/bear','data/eval/DAVIS/JPEGImages/480p/boat']"
validation_mask="['data/eval/DAVIS/Annotations/480p/bear','data/eval/DAVIS/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"
csv_file="['data/train/dataset1/metadata.csv']"

accelerate launch --mixed_precision "fp16"  \
train_DiffuEraser_stage2.py \
  --base_model_name_or_path="weights/stable-diffusion-v1-5" \
  --pretrained_stage1="weights/converted_weights/diffuEraser-model-stage1/checkpoint-1" \
  --vae_path="weights/sd-vae-ft-mse" \
  --motion_adapter_path="weights/animatediff-motion-adapter-v1-5-2" \
  --train_data_dir="data/train" \
  --csv_file="$csv_file" \
  --resolution=512 \
  --nframes=22 \
  --mixed_precision="fp16" \
  --train_batch_size=1 \
  --dataloader_num_workers=2 \
  --learning_rate=1e-05 \
  --resume_from_checkpoint="latest" \
  --validation_steps=2000 \
  --output_dir="diffuEraser-model-stage2" \
  --logging_dir="logs-stage2" \
  --validation_image="$validation_image" \
  --validation_mask="$validation_mask" \
  --validation_prompt="$validation_prompt" \
  --checkpointing_steps=2000  >train-DiffuEraser-stage2.log 2>&1