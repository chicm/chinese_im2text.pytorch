python train.py --input_fc_h5 data/res152_224_fc.h5 --input_att_h5 data/res152_224_att.h5 \
--caption_model att2in2 --checkpoint_path checkpoints_att2in2 --learning_rate 0.0004 --beam_size 1 \
--learning_rate_decay_start 10000 --learning_rate_decay_every 1000 --learning_rate_decay_rate 0.8 \
--save_checkpoint_every 500 --val_images_use 3200 \
--start_from checkpoints_att2in2
