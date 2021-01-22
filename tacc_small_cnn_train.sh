export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder 4e-4rr1 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 4e-4rr2 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 4e-4rr3 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 4e-4rr4 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 4e-4rr5 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder 4e-4rr6 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 4e-4rr7 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 4e-4rr8 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 4e-4rr9 --lr 4e-4 --save_ckpt 1>/dev/null 2>&1 &
pid9=$!

export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder 4e-4rf1 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 4e-4rf2 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 4e-4rf3 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 4e-4rf4 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 4e-4rf5 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder 4e-4rf6 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 4e-4rf7 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 4e-4rf8 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 4e-4rf9 --lr 4e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid9=$!

wait $(jobs -p)


export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder 4e-4fr1 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 4e-4fr2 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 4e-4fr3 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 4e-4fr4 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 4e-4fr5 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder 4e-4fr6 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 4e-4fr7 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 4e-4fr8 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 4e-4fr9 --lr 4e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid9=$!


export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder 4e-4ff1 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 4e-4ff2 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 4e-4ff3 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 4e-4ff4 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 4e-4ff5 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder 4e-4ff6 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 4e-4ff7 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 4e-4ff8 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 4e-4ff9 --lr 4e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid9=$!

wait $(jobs -p)

###################################

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder 8e-4rr1 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 8e-4rr2 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 8e-4rr3 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 8e-4rr4 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 8e-4rr5 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder 8e-4rr6 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 8e-4rr7 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 8e-4rr8 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 8e-4rr9 --lr 8e-4 --save_ckpt 1>/dev/null 2>&1 &
pid9=$!

export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder 8e-4rf1 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 8e-4rf2 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 8e-4rf3 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 8e-4rf4 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 8e-4rf5 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder 8e-4rf6 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 8e-4rf7 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 8e-4rf8 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 8e-4rf9 --lr 8e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid9=$!

wait $(jobs -p)

export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder 8e-4fr1 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 8e-4fr2 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 8e-4fr3 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 8e-4fr4 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 8e-4fr5 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder 8e-4fr6 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 8e-4fr7 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 8e-4fr8 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 8e-4fr9 --lr 8e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid9=$!



export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder 8e-4ff1 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 8e-4ff2 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 8e-4ff3 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 8e-4ff4 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 8e-4ff5 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder 8e-4ff6 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 8e-4ff7 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 8e-4ff8 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 8e-4ff9 --lr 8e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid9=$!

wait $(jobs -p)


#############################
export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder 16e-4rr1 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 16e-4rr2 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 16e-4rr3 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 16e-4rr4 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 16e-4rr5 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder 16e-4rr6 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 16e-4rr7 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 16e-4rr8 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 16e-4rr9 --lr 16e-4 --save_ckpt 1>/dev/null 2>&1 &
pid9=$!


export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder 16e-4rf1 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 16e-4rf2 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 16e-4rf3 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 16e-4rf4 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 16e-4rf5 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder 16e-4rf6 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 16e-4rf7 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 16e-4rf8 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 16e-4rf9 --lr 16e-4 --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid9=$!

wait $(jobs -p)


export CUDA_VISIBLE_DEVICES="0"
nohup python small_cnn.py --ckpt_folder 16e-4fr1 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 16e-4fr2 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 16e-4fr3 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 16e-4fr4 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 16e-4fr5 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="1"
nohup python small_cnn.py --ckpt_folder 16e-4fr6 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 16e-4fr7 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 16e-4fr8 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 16e-4fr9 --lr 16e-4 --deterministic_init --save_ckpt 1>/dev/null 2>&1 &
pid9=$!


export CUDA_VISIBLE_DEVICES="2"
nohup python small_cnn.py --ckpt_folder 16e-4ff1 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid1=$!
nohup python small_cnn.py --ckpt_folder 16e-4ff2 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid2=$!
nohup python small_cnn.py --ckpt_folder 16e-4ff3 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid3=$!
nohup python small_cnn.py --ckpt_folder 16e-4ff4 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid4=$!
nohup python small_cnn.py --ckpt_folder 16e-4ff5 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid5=$!
export CUDA_VISIBLE_DEVICES="3"
nohup python small_cnn.py --ckpt_folder 16e-4ff6 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid6=$!
nohup python small_cnn.py --ckpt_folder 16e-4ff7 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid7=$!
nohup python small_cnn.py --ckpt_folder 16e-4ff8 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid8=$!
nohup python small_cnn.py --ckpt_folder 16e-4ff9 --lr 16e-4 --deterministic_init --deterministic_input --save_ckpt 1>/dev/null 2>&1 &
pid9=$!

wait $(jobs -p)