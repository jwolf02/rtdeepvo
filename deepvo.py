import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, TimeDistributed, Flatten, Dense, LSTM, MaxPool2D, LeakyReLU, Dropout, BatchNormalization
from datetime import datetime
import cv2
from scipy.spatial.transform import Rotation as R
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
import torch
import sys

def move_axis(tensor):
  return np.moveaxis(tensor, [0, 1, 2, 3], [-1, -2, -4, -3])

if len(sys.argv) < 3:
  print("Usage:", sys.argv[0], "<flownet_pytorch_file> <output weights file>")
  exit(1)
  
flownet_torch_file = sys.argv[1]
weights_file = sys.argv[2]

flownet_weights = dict()
flownet_betas = dict()
flownet_gammas = dict()
flownet_moving_mean = dict()
flownet_moving_var = dict()

pretrained_flownet = torch.load(flownet_torch_file, map_location=torch.device('cpu'))
state_dict = pretrained_flownet['state_dict']
for layer in ["conv1", "conv2", "conv3", "conv3_1", "conv4", "conv4_1", "conv5", "conv5_1", "conv6"]:
  flownet_weights[layer] = move_axis(state_dict[layer + ".0.weight"].numpy())
  flownet_betas[layer] = state_dict[layer + ".1.bias"].numpy()
  flownet_gammas[layer] = state_dict[layer + ".1.weight"].numpy()
  flownet_moving_mean[layer] = state_dict[layer + ".1.running_mean"].numpy()
  flownet_moving_var[layer] = state_dict[layer + ".1.running_var"].numpy()

class FNWeights(tf.keras.initializers.Initializer):
  def __init__(self, layer):
    self.layer = layer

  def __call__(self, shape, dtype=None):
    return flownet_weights[self.layer]

class FNBetas(tf.keras.initializers.Initializer):
  def __init__(self, layer):
    self.layer= layer
    
  def __call__(self, shape, dtype=None):
    return flownet_betas[self.layer]

class FNGammas(tf.keras.initializers.Initializer):
  def __init__(self, layer):
    self.layer= layer
    
  def __call__(self, shape, dtype=None):
    return flownet_gammas[self.layer]
    
class FNMovingMean(tf.keras.initializers.Initializer):
  def __init__(self, layer):
    self.layer = layer

  def __call__(self, shape, dtype=None):
    return flownet_moving_mean[self.layer]

class FNMovingVar(tf.keras.initializers.Initializer):
  def __init__(self, layer):
    self.layer = layer
    
  def __call__(self, shape, dtype=None):
    return flownet_moving_var[self.layer]

WIDTH = 1280
HEIGHT = 384
CHANNELS = 6

BATCH_SIZE = 1
TS_LEN = 1

def euclidean_distance(y_true, y_pred):
  return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
  
def conv(x, name, filters, size, stride, activation='relu', trainable=True):
  x = TimeDistributed(Conv2D(filters, (size, size), strides=(stride, stride), padding="same", name=name, 
    kernel_initializer=FNWeights(name), use_bias=False, activation=activation, trainable=trainable), name="dt_" + name)(x)
  return TimeDistributed(BatchNormalization(beta_initializer=FNBetas(name), gamma_initializer=FNGammas(name),
    moving_mean_initializer=FNMovingMean(name), moving_variance_initializer=FNMovingVar(name), trainable=trainable, name="bn_" + name), 
    name="dt_bn_" + name)(x)

def build_rcnn(batch_size=BATCH_SIZE, ts_len=TS_LEN, trainable=False):
  print("building rcnn model")
  
  input_layer = keras.Input(batch_shape=(batch_size, ts_len, HEIGHT, WIDTH, CHANNELS), name="input")
  x = conv(input_layer, "conv1", 64, 7, 2, trainable=trainable)
  x = conv(x, "conv2", 128, 5, 2, trainable=trainable)
  x = conv(x, "conv3", 256, 5, 2, trainable=trainable)
  x = conv(x, "conv3_1", 256, 3, 1, trainable=trainable)
  x = conv(x, "conv4", 512, 3, 2, trainable=trainable)
  x = conv(x, "conv4_1", 512, 3, 1, trainable=trainable)
  x = conv(x, "conv5", 512, 3, 2, trainable=trainable)
  x = conv(x, "conv5_1", 512, 3, 1, trainable=trainable)
  x = conv(x, "conv6", 1024, 3, 2, activation=None, trainable=trainable)
  x = TimeDistributed(Flatten(name="flatten"), name="dt_flatten")(x)
  x = LSTM(1000, return_sequences=True, stateful=True, name="lstm1")(x)
  x = LSTM(1000, return_sequences=True, stateful=True, name="lstm2")(x)
  trans = TimeDistributed(Dense(2, name="translation"), name="dt_translation")(x)
  rot = TimeDistributed(Dense(1, name='rotation'), name="dt_rotation")(x)
  model = keras.Model(inputs=[input_layer], outputs=[trans, rot], name='RTDeepVO')
  losses = { 'dt_rotation': 'mse', 'dt_translation': euclidean_distance }
  loss_weights = { 'dt_rotation': 100.0, 'dt_translation': 1.0 }
  model.compile(optimizer='adagrad', loss=losses, loss_weights=loss_weights)
  return model

def test_performance(model, n=30):
  print("initial run")
  tf.keras.backend.set_learning_phase(0)
  tensor = np.zeros([BATCH_SIZE, TS_LEN, HEIGHT, WIDTH, CHANNELS])
  model.predict(tensor)

  total_ms = 0.0
  values = []
  print("0/", n, sep="", end="", flush=True)
  for i in range(n):
    begin = datetime.now()
    var = model.predict(tensor)
    end = datetime.now()
    elapsed = end - begin
    values.append(elapsed.seconds * 1000 + elapsed.microseconds / 1000)
    print("\r", i + 1, "/", n, sep="", end="", flush=True)
  print("\nmedian time per run:", round(np.median(values) if n > 0 else 0, 2), "ms")

model = build_rcnn()
model.save_weights(weights_file)
print(model.summary())
test_performance(model, 10)

