#!/bin/bash

python3 --version
python3 -c 'import tensorflow as tf; print("Tensor Flow: %s" %tf.__version__)'
python3 -c 'import numpy; print("Numpy: %s" %numpy.__version__)'
python3 -c 'import gym; print("Gym: %s" %gym.__version__)'
python3 -c 'import tqdm; print("tqdm: %s" %tqdm.__version__)'
#python3 -c 'import bunch; print("bunch: %s" %bunch.__version__)'
#python3 -c 'import matplotlib; print("matplotlib: %s" %matplotlib.__version__)'
#python3 -c 'import Pillow; print("Pillow: %s" %Pillow.__version__)'
