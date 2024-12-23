output_dir=reports/test4
mkdir -p ${output_dir}
nohup python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    main_yokogawa.py \
    --lr 1e-5 \
    --batch_size 96 \
    --dataset_file yokogawa \
    --output_dir ${output_dir} > ${output_dir}/stdout_train.log &
