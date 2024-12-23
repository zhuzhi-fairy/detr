output_dir=reports/test3
nohup python main_yokogawa.py \
    --dataset_file yokogawa \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume ${output_dir}/checkpoint.pth \
    --output_dir $output_dir > ${output_dir}/stdout_test.log &
