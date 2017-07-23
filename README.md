# tensorflow-cifar-10-NiN
Cifar-10 Network in Network implementation example using TensorFlow library.

## Requirement
**Library** | **Version**
--- | ---
**Python** | **^2.7**
**Tensorflow** | **^1.0.1** 
**Numpy** | **^1.12.0** 
**Pickle** |  *  

## Usage
### Download code:
```sh
git clone https://github.com/exelban/tensorflow-cifar-10

cd tensorflow-cifar-10
```

### Train cnn:
Batch size: 128

Prediction made on per epoch basis. 

161 epochs takes about 3h on GTX 1080.

```sh
python train.py
```

#### Make prediction:
```sh
python predict.py
```

Example output:
```sh
Trying to restore last checkpoint ...
Restored checkpoint from: ./tensorboard/cifar-10/-20000
Accuracy on Test-Set: 75.73% (7573 / 10000)
[848   9  42  12  16   3   8   8  38  16] (0) airplane
[ 21 841   7   6   1   8   5   1  35  75] (1) automobile
[ 55   2 720  47  78  29  26  26   6  11] (2) bird
[ 33  10  83 587  74 118  47  24   8  16] (3) cat
[ 18   0  89  56 755  16  18  40   7   1] (4) deer
[ 18   5  77 194  58 581  15  40   4   8] (5) dog
[ 15   4  65  69  39  18 771   6   8   5] (6) frog
[ 23   0  36  36  75  30   3 789   1   7] (7) horse
[ 61  18  10   9   8   6   6   2 858  22] (8) ship
[ 41  70  10  14   3   4   2   6  27 823] (9) truck
 (0) (1) (2) (3) (4) (5) (6) (7) (8) (9)
```

## Tensorboard
```sh
tensorboard --logdir=./tensorboard
```

## Model

| **Convolution layer 1** |
| :---: |
| Conv_2d |
| ReLu |
| MLP |
| ReLu |
| MLP |
| ReLu |
| MaxPool |
| **Convolution layer 2** |
| Conv_2d |
| ReLu |
| MLP |
| ReLu |
| MLP |
| ReLu |
| MaxPool |
| **Convolution layer 3**  |
| Conv_2d |
| ReLu |
| MLP |
| ReLu |
| MLP |
| ReLu |
| AvgPool |
| **Softmax_linear** |

## License
[Apache License 2.0](https://github.com/eugenelet/tensorflow-cifar-10-NiN/blob/master/LICENSE)
