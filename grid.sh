grid train \
    --g_name hasty-cobra-769 \
    --g_disk_size 200 \
    --g_max_nodes 10 \
    --g_instance_type g4dn.xlarge \
    --g_use_spot \
    --g_gpus 1 \
    train.py \
    --max_epochs 3 \
    --gpus 1