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
git clone https://github.com/eugenelet/tensorflow-cifar-10-NiN

cd tensorflow-cifar-10-NiN
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
('Restored checkpoint from:', u'./tensorboard/aug-decay-RMS/-188692')
Accuracy on Test-Set: 86.18% (8618 / 10000)
[890   4  25   5  12   1  23   4  29   7] (0) airplane
[  8 895   2   3   3   3  30   2  18  36] (1) automobile
[ 19   0 759  15  59  24 106  13   4   1] (2) bird
[ 13   1  28 696  46  86  99  22   5   4] (3) cat
[  4   0  19  13 872  13  62  17   0   0] (4) deer
[  2   1  18  71  40 807  38  21   0   2] (5) dog
[  3   0   4   9   5   2 974   2   1   0] (6) frog
[  3   0  14  14  29  19  15 905   0   1] (7) horse
[ 32   1   4   3   8   3  26   0 913  10] (8) ship
[ 12  20   3   5   1   1  21   3  27 907] (9) truck
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

## Implementation Details
[My Blog](https://embedai.wordpress.com/2017/07/23/network-in-network-implementation-using-tensorflow/)
