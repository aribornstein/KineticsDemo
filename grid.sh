grid train \
    --g_name $1 \
    --g_disk_size 200 \
    --g_max_nodes 10 \
    --g_instance_type t2.medium \
    --g_gpus 0 \
    train.py \
    --fast_dev_run 1