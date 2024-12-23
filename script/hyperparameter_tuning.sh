lrs=(5e-4 1e-4 7e-5 4e-5 1e-5 5e-6)
for index in "${!lrs[@]}"
do
    trial_name="trial${index}"
    lr=${lrs[$index]}
    output_dir="reports/ResNet50/hypertuning/lr/${trial_name}"
    mkdir -p ${output_dir}
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env \
        main_yokogawa.py \
        --lr ${lr} \
        --batch_size 64 \
        --dataset_file yokogawa \
        --output_dir ${output_dir} > ${output_dir}/stdout_train.log
done
