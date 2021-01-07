export CUDA_VISIBLE_DEVICES="1"
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rr1  --init_lr 8e-4 1>/dev/null 2>&1 &
pid1=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rr2  --init_lr 8e-4 1>/dev/null 2>&1 &
pid2=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rr3  --init_lr 8e-4 1>/dev/null 2>&1 &
pid3=$!
wait $pid1
wait $pid2
wait $pid3
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rr4  --init_lr 8e-4 1>/dev/null 2>&1 &
pid4=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rr5  --init_lr 8e-4 1>/dev/null 2>&1 &
pid5=$!

wait $pid4
wait $pid5

export CUDA_VISIBLE_DEVICES="1"
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rf1  --init_lr 8e-4 --deterministic_input 1>/dev/null 2>&1 &
pid6=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rf2  --init_lr 8e-4 --deterministic_input 1>/dev/null 2>&1 &
pid7=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rf3  --init_lr 8e-4 --deterministic_input 1>/dev/null 2>&1 &
pid8=$!
wait $pid6
wait $pid7
wait $pid8
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rf4  --init_lr 8e-4 --deterministic_input 1>/dev/null 2>&1 &
pid9=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4rf5  --init_lr 8e-4 --deterministic_input 1>/dev/null 2>&1 &
pid10=$!



wait $pid9
wait $pid10


export CUDA_VISIBLE_DEVICES="1"
nohup python cifar10_train.py --train_steps 60000 --version 8e-4fr1  --init_lr 8e-4 --deterministic_init 1>/dev/null 2>&1 &
pid1=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4fr2  --init_lr 8e-4 --deterministic_init 1>/dev/null 2>&1 &
pid2=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4fr3  --init_lr 8e-4 --deterministic_init 1>/dev/null 2>&1 &
pid3=$!
wait $pid1
wait $pid2
wait $pid3
nohup python cifar10_train.py --train_steps 60000 --version 8e-4fr4  --init_lr 8e-4 --deterministic_init 1>/dev/null 2>&1 &
pid4=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4fr5  --init_lr 8e-4 --deterministic_init 1>/dev/null 2>&1 &
pid5=$!


wait $pid4
wait $pid5

export CUDA_VISIBLE_DEVICES="1"
nohup python cifar10_train.py --train_steps 60000 --version 8e-4ff1  --init_lr 8e-4 --deterministic_init --deterministic_input 1>/dev/null 2>&1 &
pid6=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4ff2  --init_lr 8e-4 --deterministic_init --deterministic_input 1>/dev/null 2>&1 &
pid7=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4ff3  --init_lr 8e-4 --deterministic_init --deterministic_input 1>/dev/null 2>&1 &
pid8=$!
wait $pid6
wait $pid7
wait $pid8
nohup python cifar10_train.py --train_steps 60000 --version 8e-4ff4  --init_lr 8e-4 --deterministic_init --deterministic_input 1>/dev/null 2>&1 &
pid9=$!
nohup python cifar10_train.py --train_steps 60000 --version 8e-4ff5  --init_lr 8e-4 --deterministic_init --deterministic_input 1>/dev/null 2>&1 &
pid10=$!


wait $pid9
wait $pid10
