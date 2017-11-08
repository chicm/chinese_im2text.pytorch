python train_cnn.py \
--caption_model show_attend_tell --checkpoint_path checkpoints/sat_cnn --learning_rate 0.0004 --beam_size 1 \
--save_checkpoint_every 500 --val_images_use 3200 \
--start_from checkpoints/sat480
