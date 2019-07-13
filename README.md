# Backpropagation-in-neural-networks

#### Overview

We train feed-forward artificial neural networks by implementing backpropagation
by hand. We compare results by varying the number of hidden units, number of layers,
loss functions for classification (hinge vs cross-entropy) as well as different
activation functions (ReLU, ELU, tanh, sigmoid). We carry out our experiments on the 
MNIST dataset. The results and graphs are in `Report.pdf`.

#### To run:

`python3 a4.py`

Optional arguments:

```
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs Default: 50
  --std STD             Initialize parameters in this range Default: 1e-5
  --load                Load saved params Default: False
  --load_name LOAD_NAME
                        Parameter File to load (p0/p1/p2...) Default: p
  --save                Save params Default: False
  --save_name SAVE_NAME
                        Save params to file (p0/p1/p2...) Default: p
  --batch_size BATCH_SIZE
                        Batch size for training Default: 200
  --lr LR               Learning Rate for SGD Default: 0.01
  --alpha ALPHA         alpha for ELU Default: 1.0
  --activation ACTIVATION
                        Activation Type (ReLU/tanh/sigmoid/ELU) Default: ReLU
  --HiddenLayers HIDDENLAYERS
                        Number of hidden layers (One/Two/Three) Default: One
  --nH1 NH1             Hidden size 1 Default: 100
  --nH2 NH2             Hidden size 2 Default: 100
  --nH3 NH3             Hidden size 3 Default: 100
  --loss_type LOSS_TYPE
                        Loss Function (Hinge/CE) Default: Hinge
```
