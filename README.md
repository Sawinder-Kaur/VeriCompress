# VeriCompress
VeriCompress: A Tool to Streamline the Synthesis of Verified Robust Compressed Neural Networks from Scratch

install requirements mentioned in requirements.txt

# For Benchmark Vision Datasets - CIFAR( or MNIST or SVHN)
## Use following command for training dense model:
python Benchmark/main.py --data CIFAR --cnn_4layer --compress False

## Use following command for training structured sparse model:
python Benchmark/main.py --data CIFAR --model cnn_4layer --parameter_budget 100000  --deploy True

# For Pedestrian Detection 
python -u Pedestrain-Detection/main.py --parameter_budget 100000 --deploy True
