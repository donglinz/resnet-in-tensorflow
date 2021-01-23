export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b128e1009 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 1>/dev/null 2>&1 &

wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b512e4009 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 1>/dev/null 2>&1 &

wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbar/b2048e16009 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 1>/dev/null 2>&1 &
wait $(jobs -p)

# TODO
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbar/b50000e10001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbar/b50000e10002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbar/b50000e10003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbar/b50000e10004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbar/b50000e10005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbar/b50000e10006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbar/b50000e10007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbar/b50000e10008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
wait $(jobs -p)

############################################################# deterministic tensorflow computation

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b128e1009 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 100 --deterministic_tf 1>/dev/null 2>&1 &

wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b512e4009 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 400 --deterministic_tf 1>/dev/null 2>&1 &

wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder pinkbardet/b2048e16009 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 1600 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)

# TODO
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbardet/b50000e10001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbardet/b50000e10002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbardet/b50000e10003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbardet/b50000e10004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder pinkbardet/b50000e10005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder pinkbardet/b50000e10006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder pinkbardet/b50000e10007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder pinkbardet/b50000e10008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)