python train_fer.py --model ResNet18 --kd_T 5 -s 1 --mu 1.0 #seed 1
python train_fer.py --model ResNet18 --kd_T 5 -s 5 --mu 1.0 #seed 5
python train_fer.py --model ResNet18 --kd_T 5 -s 7 --mu 1.0 #seed 7

python train_fer.py --model wrn_40_2 --kd_T 5 -s 1 --mu 0.3
python train_fer.py --model wrn_40_2 --kd_T 5 -s 5 --mu 0.3
python train_fer.py --model wrn_40_2 --kd_T 5 -s 7 --mu 0.3

python train_fer.py --model ShuffleV2 --kd_T 5 -s 1 --mu 0.5
python train_fer.py --model ShuffleV2 --kd_T 5 -s 5 --mu 0.5
python train_fer.py --model ShuffleV2 --kd_T 5 -s 7 --mu 0.5

python train_fer.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3
python train_fer.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3
python train_fer.py --model MobileNetV2 --kd_T 5 -s 1 --mu 0.3
