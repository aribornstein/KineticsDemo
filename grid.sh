git add -A && git commit -m 'i' && git push origin $1
grid train \
    --g_name $2 \
    --g_disk_size 200 \
    --g_max_nodes 10 \
    --g_instance_type g4dn.xlarge \
    --g_gpus 1 \
    train.py \
    --gpus 1 \
    --max_epochs 3