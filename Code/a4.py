import numpy as np
import random
import argparse
import scipy.io

############################# Parsing Args ####################################
def _parse_args():
  parser = argparse.ArgumentParser(description='a4.py')

  parser.add_argument('--num_epochs', type=int, default=50, \
                      help='Number of epochs \
                      Default: 50')

  parser.add_argument('--std', type=float, default=1e-5, \
                      help='Initialize parameters in this range \
                      Default: 1e-5')

  parser.add_argument('--load', dest='load_saved', default=False, \
                     action='store_true', help='Load saved params \
                     Default: False')

  parser.add_argument('--load_name', type=str, default='p', \
                      help='Parameter File to load (p0/p1/p2...) \
                      Default: p')

  parser.add_argument('--save', dest='save_data', default=False, \
                     action='store_true', help='Save params \
                     Default: False')

  parser.add_argument('--save_name', type=str, default='p', \
                      help='Save params to file (p0/p1/p2...) \
                      Default: p')

  parser.add_argument('--batch_size', type=int, default=200, \
                      help='Batch size for training \
                      Default: 200')

  parser.add_argument('--lr', type=float, default=0.01, \
                      help='Learning Rate for SGD \
                      Default: 0.01')

  parser.add_argument('--alpha', type=float, default=1.0, \
                      help='alpha for ELU \
                      Default: 1.0')

  parser.add_argument('--activation', type=str, default='ReLU', \
                      help='Activation Type (ReLU/tanh/sigmoid/ELU) \
                      Default: ReLU')

  parser.add_argument('--HiddenLayers', type=str, default='One', \
                      help='Number of hidden layers (One/Two/Three) \
                      Default: One')

  parser.add_argument('--nH1', type=int, default=100, \
                      help='Hidden size 1 \
                      Default: 100')
  
  parser.add_argument('--nH2', type=int, default=100, \
                      help='Hidden size 2 \
                      Default: 100')

  parser.add_argument('--nH3', type=int, default=100, \
                      help='Hidden size 3 \
                      Default: 100')

  parser.add_argument('--loss_type', type=str, default='Hinge', \
                      help='Loss Function (Hinge/CE)\
                      Default: Hinge')

  args = parser.parse_args()
  return args
args = _parse_args()
############################# Parsing Args ####################################

#################### Saving and Loading variables #############################
def try_load(params):
  try:
    fname = args.load_name + '.npy'
    p = np.load(fname)
    return p.item()
  except:
    return params

def save_params(params):
  fname = args.save_name + '.npy'
  np.save(fname, nn.params)
#################### Saving and Loading variables #############################

################# Instantiate and return network object #######################
def initialize_network(num_hid = args.HiddenLayers):
  if num_hid == 'One':
    return OneHiddenLayerNet()
  if num_hid == 'Two':
    return TwoHiddenLayerNet()
  if num_hid == 'Three':
    return ThreeHiddenLayerNet()
################# Instantiate and return network object #######################

############################### Activations ###################################
def activation(X):
  if args.activation == 'ReLU':
    return np.maximum(0,X)

  if args.activation == 'tanh':
    return np.tanh(X)

  if args.activation == 'sigmoid':
    return np.exp(X)/(1 + np.exp(X))

  if args.activation == 'ELU':
    return np.where(X <= 0, args.alpha*(np.exp(X) - 1), X)

def grad_activation_single(g):
  if args.activation == 'ReLU':
    if g >= 0: return 1
    return 0
  
  if args.activation == 'tanh':
    return 1 - np.square(g)

  if args.activation == 'sigmoid':
    return g*(1-g)

  if args.activation == 'ELU':
    if g > 0: return 1
    return args.alpha + g

def grad_activation(g):
  matrix_grad_activation = np.vectorize(grad_activation_single)
  return matrix_grad_activation(g)
  
############################### Activations ###################################

################################## Losses #####################################
def CE_loss(scores, nn, X, y):   
  N = scores.shape[0]    

  # Loss Calculation
  exp_scores = np.exp(scores) # N x C
  exp_sums = np.sum(exp_scores, axis = 1) # N x 1
  loss = np.sum(-np.log(exp_scores[np.arange(N),y] / exp_sums)) / N

  # Gradient Calculation
  mask = np.zeros(scores.shape)
  mask[np.arange(N), y] = 1    
  dS = ((exp_scores.T/exp_sums).T - mask)/N
  
  grads = nn.calculate_grads(dS, X)

  return loss, grads
  
def hinge_loss(scores,nn, X, y):
  N = scores.shape[0]
  
  # Loss Calculation
  sub = scores[np.arange(N),y]
  new_scores = np.transpose(np.maximum(0,np.transpose(scores) - sub + 1))
  new_scores[np.arange(N),y] -=1
  loss = np.sum(new_scores)/N

  # Gradient Calculation
  mask = np.zeros(new_scores.shape)
  mask[new_scores>0] = 1
  mask[np.arange(N),y] = -np.sum(mask, axis = 1)
  dS = mask/N

  grads = nn.calculate_grads(dS, X)

  return loss, grads
################################## Losses #####################################

########################### One Hidden Layer ##################################
class OneHiddenLayerNet(object):
  def __init__(self, nH1 = args.nH1):
    # Initialize parameters to small random values
    self.params = {}
    self.params['W1'] = args.std * np.random.randn(784, nH1)
    self.params['b1'] = np.zeros(nH1)
    self.params['W2'] = args.std * np.random.randn(nH1, 10)
    self.params['b2'] = np.zeros(10)

  def forward_pass(self, X, y=None):
    W1 = self.params['W1'] # D x nH1 
    b1 = self.params['b1'] # nH1
    W2 = self.params['W2'] # nH1 x C 
    b2 = self.params['b2'] # C
  
    self.preH1 = X.dot(W1) + b1 # N x nH1 
    self.H1 = activation(self.preH1) # N x nH1
    self.scores = self.H1.dot(W2) + b2 # N x C
    
    if y is None: return self.scores

    if args.loss_type == 'Hinge': return hinge_loss(self.scores, nn, X, y)
    if args.loss_type == 'CE': return CE_loss(self.scores, nn, X, y)

  def calculate_grads(self, dS, X):
    grads = {}
    N = dS.shape[0]

    grads['W2'] = self.H1.T.dot(dS)
    grads['b2'] = dS.T.dot(np.ones(N))

    dH1 = dS.dot(self.params['W2'].T)
    d_preH1 = np.multiply(dH1, grad_activation(self.H1))
    grads['W1'] = X.T.dot(d_preH1)
    grads['b1'] = d_preH1.T.dot(np.ones(N))
  
    return grads
########################### One Hidden Layer ##################################

########################### Two Hidden Layers #################################
class TwoHiddenLayerNet(object):
  def __init__(self, nH1 = args.nH1, nH2 = args.nH2):
    # Initialize parameters to small random values
    self.params = {}
    self.params['W1'] = args.std * np.random.randn(784, nH1)
    self.params['b1'] = np.zeros(nH1)
    self.params['W2'] = args.std * np.random.randn(nH1, nH2)
    self.params['b2'] = np.zeros(nH2)
    self.params['W3'] = args.std * np.random.randn(nH2, 10)
    self.params['b3'] = np.zeros(10)

  def forward_pass(self, X, y=None):
    W1 = self.params['W1'] # D x nH1 
    b1 = self.params['b1'] # nH1
    W2 = self.params['W2'] # nH1 x nH2 
    b2 = self.params['b2'] # nH2
    W3 = self.params['W3'] # nH2 x C
    b3 = self.params['b3'] # C
  
    self.preH1 = X.dot(W1) + b1 # N x nH1 
    self.H1 = activation(self.preH1) # N x nH1
    self.preH2 = self.H1.dot(W2) + b2 # N x nH2
    self.H2 = activation(self.preH2) # N x nH2
    self.scores = self.H2.dot(W3) + b3 # N x C 
    
    if y is None: return self.scores

    if args.loss_type == 'Hinge': return hinge_loss(self.scores, nn, X, y)
    if args.loss_type == 'CE': return CE_loss(self.scores, nn, X, y)
  
  def calculate_grads(self, dS, X):
    grads = {}
    N = dS.shape[0]

    grads['W3'] = self.H2.T.dot(dS)
    grads['b3'] = dS.T.dot(np.ones(N))

    dH2 = dS.dot(self.params['W3'].T)
    d_preH2 = np.multiply(dH2, grad_activation(self.H2))
    grads['W2'] = self.H1.T.dot(d_preH2)
    grads['b2'] = d_preH2.T.dot(np.ones(N))

    dH1 = d_preH2.dot(self.params['W2'].T)
    d_preH1 = np.multiply(dH1, grad_activation(self.H1))
    grads['W1'] = X.T.dot(d_preH1)
    grads['b1'] = d_preH1.T.dot(np.ones(N))

    return grads
########################### Two Hidden Layers #################################

########################## Three Hidden Layers ################################
class ThreeHiddenLayerNet(object):
  def __init__(self, nH0 = args.nH1, nH1 = args.nH2, nH2 = args.nH3):
    # Initialize parameters to small random values
    self.params = {}
    self.params['W0'] = args.std * np.random.randn(784, nH0)
    self.params['b0'] = np.zeros(nH0)
    self.params['W1'] = args.std * np.random.randn(nH0, nH1)
    self.params['b1'] = np.zeros(nH1)
    self.params['W2'] = args.std * np.random.randn(nH1, nH2)
    self.params['b2'] = np.zeros(nH2)
    self.params['W3'] = args.std * np.random.randn(nH2, 10)
    self.params['b3'] = np.zeros(10)

  def forward_pass(self, X, y=None):
    W0 = self.params['W0'] # D x nH0 
    b0 = self.params['b0'] # nH0
    W1 = self.params['W1'] # nH0 x nH1 
    b1 = self.params['b1'] # nH1
    W2 = self.params['W2'] # nH1 x nH2 
    b2 = self.params['b2'] # nH2
    W3 = self.params['W3'] # nH2 x C
    b3 = self.params['b3'] # C
  
    self.preH0 = X.dot(W0) + b0 # N x nH0
    self.H0 = activation(self.preH0) # N x nH0
    self.preH1 = self.H0.dot(W1) + b1 # N x nH1 
    self.H1 = activation(self.preH1) # N x nH1
    self.preH2 = self.H1.dot(W2) + b2 # N x nH2
    self.H2 = activation(self.preH2) # N x nH2
    self.scores = self.H2.dot(W3) + b3 # N x C 
    
    if y is None: return self.scores

    if args.loss_type == 'Hinge': return hinge_loss(self.scores, nn, X, y)
    if args.loss_type == 'CE': return CE_loss(self.scores, nn, X, y)
  
  def calculate_grads(self, dS, X):
    grads = {}
    N = dS.shape[0]

    grads['W3'] = self.H2.T.dot(dS)
    grads['b3'] = dS.T.dot(np.ones(N))

    dH2 = dS.dot(self.params['W3'].T)
    d_preH2 = np.multiply(dH2, grad_activation(self.H2))
    grads['W2'] = self.H1.T.dot(d_preH2)
    grads['b2'] = d_preH2.T.dot(np.ones(N))

    dH1 = d_preH2.dot(self.params['W2'].T)
    d_preH1 = np.multiply(dH1, grad_activation(self.H1))
    grads['W1'] = self.H0.T.dot(d_preH1)
    grads['b1'] = d_preH1.T.dot(np.ones(N))

    dH0 = d_preH1.dot(self.params['W1'].T)
    d_preH0 = np.multiply(dH0, grad_activation(self.H0))
    grads['W0'] = X.T.dot(d_preH0)
    grads['b0'] = d_preH0.T.dot(np.ones(N))
    
    return grads
########################## Three Hidden Layers ################################

############################### Training ######################################
def update_params(nn, grads, lr):
  nn.params['W1'] -= grads['W1'] * lr
  nn.params['b1'] -= grads['b1'] * lr
  nn.params['W2'] -= grads['W2'] * lr
  nn.params['b2'] -= grads['b2'] * lr

  if args.HiddenLayers != 'One':
    nn.params['W3'] -= grads['W3'] * lr
    nn.params['b3'] -= grads['b3'] * lr
    
  if args.HiddenLayers == 'Three':
    nn.params['W0'] -= grads['W0'] * lr
    nn.params['b0'] -= grads['b0'] * lr

def train(nn, X, y, X_test, y_test, batch_size=args.batch_size, \
          learning_rate = args.lr, num_epochs = args.num_epochs):
  loss_history = []; train_acc_history = []; test_acc_history = []
  N = X.shape[0]
  iterations_per_epoch = int(max(N/batch_size, 1))
  for epoch in range(num_epochs):  
    for it in range(iterations_per_epoch):
      mask = np.random.choice(N, batch_size)
      X_batch = X[mask]
      y_batch = y[mask]
    
      loss, grads = nn.forward_pass(X_batch, y_batch)
      loss_history.append(loss)

      update_params(nn, grads, learning_rate)

    train_acc = (predict(nn, X) == y).mean()*100
    train_acc_history.append(train_acc)

    test_acc = (predict(nn, X_test) == y_test).mean()*100
    test_acc_history.append(test_acc)
    
    print("Epoch ", epoch + 1, ":", sep = '')
    print("Loss:", loss)
    print("Train Accuracy:", "{0:.2f}".format(train_acc))
    print("Test Accuracy:", "{0:.2f}".format(test_acc))
    save_histories(loss_history, train_acc_history, test_acc_history)

  return loss_history, train_acc_history, test_acc_history
############################### Training ######################################

############################### Inference #####################################
def predict (nn, X):
  scores = nn.forward_pass(X) # N x C
  y_pred = np.argmax(scores, axis = 1)
  return y_pred
############################### Inference #####################################

############################### Loading Data ##################################
mat = scipy.io.loadmat('digits.mat')
num_train = 60000; num_test = 10000

train_label = mat['trainLabels']; test_label = mat['testLabels']
train_set = mat['trainImages']; test_set = mat['testImages']
train_set = train_set.reshape(-1, num_train)
train_set = train_set.T # 60k x 784
test_set = test_set.reshape(-1, num_test)
test_set = test_set.T # 10k x 784  
train_labels = train_label.reshape(num_train) # 60k
test_labels = test_label.reshape(num_test) # 10k
train_set = train_set/255.0; test_set = test_set/255.0
############################### Loading Data ##################################

#################### Saving and Loading variables #############################
def try_load(params):
  try:
    fname = args.load_name + '.npy'
    p = np.load(fname)
    return p.item()
  except:
    return params

def save_params(params):
  fname = args.save_name + '.npy'
  np.save(fname, nn.params)

def save_histories(loss_history, train_acc_history, test_acc_history):
  np.save('plots/loss_history.npy', loss_history)
  np.save('plots/train_acc_history.npy', train_acc_history)
  np.save('plots/test_acc_history.npy', test_acc_history)
#################### Saving and Loading variables #############################

############################## Run Experiments ################################
nn = initialize_network()
 
if args.load_saved: nn.params = try_load(nn.params)

loss_history, train_acc_history, test_acc_history = \
  train(nn, train_set, train_labels, test_set, test_labels)

save_histories(loss_history, train_acc_history, test_acc_history)

if args.save_data: save_params(params)
############################## Run Experiments ################################
