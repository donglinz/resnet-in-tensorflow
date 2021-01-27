folder1="pinkbartest"
folder2="pinkbardettest"

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b128e21 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b128e22 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b128e23 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b128e24 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b128e25 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b128e26 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b128e27 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b128e28 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 1>/dev/null 2>&1 &

wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b512e81 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b512e82 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b512e83 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b512e84 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b512e85 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b512e86 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b512e87 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b512e88 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 1>/dev/null 2>&1 &

wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b2048e321 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b2048e322 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b2048e323 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b2048e324 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b2048e325 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b2048e326 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b2048e327 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b2048e328 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 1>/dev/null 2>&1 &
wait $(jobs -p)

# wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b8192e1281 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b8192e1282 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b8192e1283 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b8192e1284 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b8192e1285 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b8192e1286 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b8192e1287 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder1/b8192e1288 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 1>/dev/null 2>&1 &
wait $(jobs -p)



# TODO
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b50000e10001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b50000e10002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b50000e10003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b50000e10004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b50000e10005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b50000e10006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder1/b50000e10007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder1/b50000e10008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 1>/dev/null 2>&1 &
wait $(jobs -p)

############################################################# deterministic tensorflow computation

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b128e21 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b128e22 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b128e23 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b128e24 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b128e25 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b128e26 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b128e27 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b128e28 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 128 --epochs 2 --deterministic_tf 1>/dev/null 2>&1 &

wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b512e81 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b512e82 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b512e83 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b512e84 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b512e85 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b512e86 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b512e87 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b512e88 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 512 --epochs 8 --deterministic_tf 1>/dev/null 2>&1 &

wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b2048e321 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b2048e322 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b2048e323 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b2048e324 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b2048e325 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b2048e326 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b2048e327 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b2048e328 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 2048 --epochs 32 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)

# wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b8192e1281 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b8192e1282 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b8192e1283 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b8192e1284 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b8192e1285 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b8192e1286 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b8192e1287 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 --deterministic_tf 1>/dev/null 2>&1 &
nohup python small_cnn.py --ckpt_folder $folder2/b8192e1288 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 8192 --epochs 128 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)

# TODO
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b50000e10001 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b50000e10002 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b50000e10003 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b50000e10004 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b50000e10005 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b50000e10006 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder $folder2/b50000e10007 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder $folder2/b50000e10008 --lr 4e-4 --deterministic_init --save_ckpt --save_pred --batch_size 50000 --epochs 1000 --deterministic_tf 1>/dev/null 2>&1 &
wait $(jobs -p)
