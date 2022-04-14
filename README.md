# CSC2541 Final Project

## Authors

* Jiacheng Yang
* Xin Li

## Install Dependencies

```shell
pip3 install -r requirements.txt
```

Note that this will install the default pip version of `jax`. 
Please change it if a specific CUDA/CUDNN version of `jax` is needed. 

Before running any scripts, make sure all those non-package dependencies are
included in `PYTHONPATH`:
```shell
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:third_party/FlexBricks"
```

## Preparing the Datasets

The 225-bird-species dataset is constantly updated, and we provide the version that we used for 
our experiments [here](https://cs.toronto.edu/~jcyang/assets/birds.tar.gz). The other dataset 
can be downloaded and reorganized on their corresponding website.

All datasets should be organized to follow the pattern `<dataset_root>/<split>/<class>/<image>` 
so that the `ImageFolder` can read all the datasets correctly. 

## Reproducing the figures

For the results on the linear regression models, please refer to [here](linear_regression_MNIST.ipynb)

In the following training commands, please remove the `-G` option if GPU is not available. 

```shell
export DATASET_PATH=~/Dataset  # Change it to the your dataset path
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+225-bird-species/global=0.001/sgd
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+scenes/global=0.001/sgd
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+flowers-102/global=0.001/sgd
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+cars/global=0.001/sgd
python3 scripts/train.py -G -D $DATASET_PATH -C configs/densenet121/imagenet+225-bird-species/global=0.001/sgd
python3 scripts/train.py -G -D $DATASET_PATH -C configs/mobilenetv2/imagenet+225-bird-species/global=0.001/sgd
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+225-bird-species/global=0.001/sgd/lr=0.0006
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+225-bird-species/global=0.001/sgd/lr=0.0002
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+225-bird-species/global=0.001/sgd_with_momentum
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+225-bird-species/global=0.001/adam
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+225-bird-species/global=0.001/sgd
python3 scripts/train.py -G -D $DATASET_PATH -C configs/resnet18/imagenet+225-bird-species/global=0.001/sgd/rayleigh.yml
```

Then, use the `scripts/plot.py` to get the figures:

```shell
python3 scripts/plot.py -C configs/@figures/different_datasets.yml
python3 scripts/plot.py -C configs/@figures/different_dnns.yml
python3 scripts/plot.py -C configs/@figures/different_learning_rate.yml
python3 scripts/plot.py -C configs/@figures/different_optimizer.yml

# Note that data of the rayleigh quotient figure should be obtained from the JSON file 
# in TensorBoard. We provide our version at the assets/ folder. Any changes to the file 
# should also be mentioned in the figure configuration file to get the right path to the data.  
python3 scripts/plot.py -C configs/@figures/rayleigh_quotient.yml
```

### Tracking in TensorBoard

The training script will generate a folder `sessions/`, which in which it will 
store the TensorBoard events. To view these events, just run:
```shell
tensorboard --logdir sessions/
```
