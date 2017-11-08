python train.py --input_fc_h5 data/res152_480_fc.h5 --input_att_h5 data/res152_480_att.h5 \
--caption_model show_attend_tell --checkpoint_path checkpoints/sat480 --learning_rate 0.0004 --beam_size 1 \
--save_checkpoint_every 500 --val_images_use 3200 \
--start_from checkpoints/sat480 --language_eval 0
