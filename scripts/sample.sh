#!/usr/bin/env sh
res="1024:1024x1024"
time_shifting_factor=6
cfg_scale=4.0
seed=0
steps=18
solver=midpoint # midpoint, euler, dpm
system_type=real
model_dir=/your/ckpt/path
cap_dir=/your/caption/path
out_dir=samples/infer_${system_type}_${solver}_${cfg_scale}_${time_shifting_factor}
python -u sample.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${time_shifting_factor} \
--cfg_scale ${cfg_scale} \
--system_type ${system_type} \
--batch_size 1 \
--hf_token xxx \
--rank 0 \
