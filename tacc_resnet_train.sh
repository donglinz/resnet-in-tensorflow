
export CUDA_VISIBLE_DEVICES="0"
nohup python cifar10_train.py --train_steps 60000 --version 4e-4rr  --init_lr 4e-4 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python cifar10_train.py --train_steps 60000 --version 4e-4rf --deterministic_input --init_lr 4e-4 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python cifar10_train.py --train_steps 60000 --version 4e-4fr --deterministic_init --init_lr 4e-4 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python cifar10_train.py --train_steps 60000 --version 4e-4ff --deterministic_input --deterministic_init --init_lr 4e-4 1>/dev/null 2>&1 &

wait $(jobs -p)