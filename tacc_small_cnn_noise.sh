export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder noise_soft1 --lr 4e-4 --save_ckpt --save_pred --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_soft2 --lr 4e-4 --save_ckpt --save_pred --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder noise_soft3 --lr 4e-4 --save_ckpt --save_pred --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_soft4 --lr 4e-4 --save_ckpt --save_pred --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder noise_soft5 --lr 4e-4 --save_ckpt --save_pred --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_soft6 --lr 4e-4 --save_ckpt --save_pred --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder noise_soft7 --lr 4e-4 --save_ckpt --save_pred --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_soft8 --lr 4e-4 --save_ckpt --save_pred --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &

wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder noise_hard1 --lr 4e-4 --save_ckpt --save_pred --deterministic_init --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_hard2 --lr 4e-4 --save_ckpt --save_pred --deterministic_init --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder noise_hard3 --lr 4e-4 --save_ckpt --save_pred --deterministic_init --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_hard4 --lr 4e-4 --save_ckpt --save_pred --deterministic_init --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder noise_hard5 --lr 4e-4 --save_ckpt --save_pred --deterministic_init --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_hard6 --lr 4e-4 --save_ckpt --save_pred --deterministic_init --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder noise_hard7 --lr 4e-4 --save_ckpt --save_pred --deterministic_init --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_hard8 --lr 4e-4 --save_ckpt --save_pred --deterministic_init --deterministic_input --deterministic_tf 1>/dev/null 2>&1 &

wait $(jobs -p)


export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder noise_softhard1 --lr 4e-4 --save_ckpt --save_pred --deterministic_input 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_softhard2 --lr 4e-4 --save_ckpt --save_pred --deterministic_input 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder noise_softhard3 --lr 4e-4 --save_ckpt --save_pred --deterministic_input 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_softhard4 --lr 4e-4 --save_ckpt --save_pred --deterministic_input 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder noise_softhard5 --lr 4e-4 --save_ckpt --save_pred --deterministic_input 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_softhard6 --lr 4e-4 --save_ckpt --save_pred --deterministic_input 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder noise_softhard7 --lr 4e-4 --save_ckpt --save_pred --deterministic_input 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder noise_softhard8 --lr 4e-4 --save_ckpt --save_pred --deterministic_input 1>/dev/null 2>&1 &

wait $(jobs -p)