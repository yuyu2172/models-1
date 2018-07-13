# Multi-label Image Classification

Chainer implementation of multi-label image classification.
Currently, the training and evaluation is conducted with ResNet50 and PASCAL VOC07.

## Dependency

- Chainer==4.2.0
- CuPy==4.2.0
- ChainerCV==0.10.0

## Train
```
$ python train.py [--gpu <gpu>] [--batchsize <bsize>] [--out <out>]
```

## Evaluation
```
$ python eval_voc07.py [--gpu <gpu>] [--pretrained-model <path>]
```

## Demo
```
$ python demo.py [--gpu <gpu>] [--pretrained-model <model_path>] <image>.jpg
```

## Performance
The mAP of the network trained with VOC07 train-val and evaluated on VOC07 test.

```
mAP: 0.822417
aeroplane: 0.928972
bicycle: 0.881060
bird: 0.859881
boat: 0.852588
bottle: 0.550730
bus: 0.840429
car: 0.932287
cat: 0.887794
chair: 0.607687
cow: 0.783687
diningtable: 0.760139
dog: 0.860601
horse: 0.909372
motorbike: 0.867240
person: 0.958908
pottedplant: 0.696536
sheep: 0.770368
sofa: 0.752435
train: 0.944287
tvmonitor: 0.803339
```


## References
1. Maksim Lapin, Matthias Hein, and Bernt Schiele. "Analysis and Optimization of Loss Functions for Multiclass, Top-k, and Multilabel Classification" In IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.
The row "Multi-label" of Table 10 is particularly relevant [link](https://arxiv.org/pdf/1612.03663.pdf).
