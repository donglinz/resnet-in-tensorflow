export CUDA_VISIBLE_DEVICES="0"
python cifar10_train.py --test True --test_ckpt_path logs_16e-4rr1/model.ckpt-3999
python cifar10_train.py --test True --test_ckpt_path logs_16e-4rr2/model.ckpt-3999
python cifar10_train.py --test True --test_ckpt_path logs_16e-4rr3/model.ckpt-3999

python cifar10_train.py --test True --test_ckpt_path logs_16e-4rf1/model.ckpt-3999
python cifar10_train.py --test True --test_ckpt_path logs_16e-4rf2/model.ckpt-3999
python cifar10_train.py --test True --test_ckpt_path logs_16e-4rf3/model.ckpt-3999

python cifar10_train.py --test True --test_ckpt_path logs_16e-4fr1/model.ckpt-3999
python cifar10_train.py --test True --test_ckpt_path logs_16e-4fr2/model.ckpt-3999
python cifar10_train.py --test True --test_ckpt_path logs_16e-4fr3/model.ckpt-3999

python cifar10_train.py --test True --test_ckpt_path logs_16e-4ff1/model.ckpt-3999
python cifar10_train.py --test True --test_ckpt_path logs_16e-4ff2/model.ckpt-3999
python cifar10_train.py --test True --test_ckpt_path logs_16e-4ff3/model.ckpt-3999

