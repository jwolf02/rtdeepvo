import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, TimeDistributed, Flatten, Dense, LSTM, MaxPool2D, LeakyReLU
from datetime import datetime
import cv2
from scipy.spatial.transform import Rotation as R
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
import torchfile

def get_kernel(model, index):
  return model.modules[index].__dict__['_obj'][b'weight']
  
def get_bias(model, index):
  return model.modules[index].__dict__['_obj'][b'bias']

def move_axis(tensor):
  return np.moveaxis(tensor, [0, 1, 2, 3], [-1, -2, -4, -3])

flownet = torchfile.load("FlowNetS_from_scratch.t7")

flownet_kernels = []
flownet_biases = []

for i in range(1, 18, 2):
  k = move_axis(get_kernel(flownet, i))
  b = get_bias(flownet, i)
  print("layer ", i // 2, ": kernel_shape=", k.shape, " bias_shape=", b.shape, sep="")
  flownet_kernels.append(k)
  flownet_biases.append(b)
  
class FlowNetKernelInitializer(tf.keras.initializers.Initializer):
  def __init__(self, layer):
    self.layer = layer

  def __call__(self, shape, dtype=None):
    return flownet_kernels[self.layer]

class FlowNetBiasInitializer(tf.keras.initializers.Initializer):
  def __init__(self, layer):
    self.layer= layer
    
  def __call__(self, shape, dtype=None):
    return flownet_biases[self.layer]

WIDTH = 1280
HEIGHT = 384
CHANNELS = 6

BATCH_SIZE = 1
TS_LEN = 1

def euclidean_distance(y_true, y_pred):
  return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def build_rcnn(batch_size=BATCH_SIZE, ts_len=TS_LEN):
  print("building rcnn model")
  
  input_layer = keras.Input(batch_shape=(batch_size, ts_len, HEIGHT, WIDTH, CHANNELS), name="input")
  x = TimeDistributed(Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1", 
    kernel_initializer=FlowNetKernelInitializer(0), bias_initializer=FlowNetBiasInitializer(0)), name="dt_conv1")(input_layer)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky1"), name="dt_leaky1")(x)
  x = TimeDistributed(Conv2D(128, (5, 5), strides=(2, 2), padding="same", name="conv2", 
    kernel_initializer=FlowNetKernelInitializer(1), bias_initializer=FlowNetBiasInitializer(1)), name="dt_conv2")(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky2"), name="dt_leaky2")(x)
  x = TimeDistributed(Conv2D(256, (5, 5), strides=(2, 2), padding="same", name="conv3", 
    kernel_initializer=FlowNetKernelInitializer(2), bias_initializer=FlowNetBiasInitializer(2)), name="dt_conv3")(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky3"), name="dt_leaky3")(x)
  x = TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv3_1", 
    kernel_initializer=FlowNetKernelInitializer(3), bias_initializer=FlowNetBiasInitializer(3)), name="dt_conv3_1")(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky3_1"), name="dt_leaky3_1")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv4", 
    kernel_initializer=FlowNetKernelInitializer(4), bias_initializer=FlowNetBiasInitializer(4)), name="dt_conv4")(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky4"), name="dt_leaky4")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv4_1", 
    kernel_initializer=FlowNetKernelInitializer(5), bias_initializer=FlowNetBiasInitializer(5)), name="dt_conv4_1")(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky4_1"), name="dt_leaky4_1")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv5", 
    kernel_initializer=FlowNetKernelInitializer(6), bias_initializer=FlowNetBiasInitializer(6)), name="dt_conv5")(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky5"), name="dt_leaky5")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_1", 
    kernel_initializer=FlowNetKernelInitializer(7), bias_initializer=FlowNetBiasInitializer(7)), name="dt_conv5_1")(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky5_1"), name="dt_leaky5_1")(x)
  x = TimeDistributed(Conv2D(1024, (3, 3), strides=(2, 2), padding="same", name="conv6", 
    kernel_initializer=FlowNetKernelInitializer(8), bias_initializer=FlowNetBiasInitializer(8)), name="dt_conv6")(x)
  x = TimeDistributed(Flatten(name="flatten"), name="dt_flatten")(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(1000, return_sequences=True, stateful=True, name="lstm1")(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(1000, return_sequences=True, stateful=True, name="lstm2")(x)
  trans = TimeDistributed(Dense(2, name="translation"), name="dt_translation")(x)
  rot = TimeDistributed(Dense(1, name='rotation'), name="dt_rotation")(x)
  model = keras.Model(inputs=[input_layer], outputs=[trans, rot], name='RTDeepVO')
  losses = { 'dt_rotation': 'mse', 'dt_translation': euclidean_distance }
  loss_weights = { 'dt_rotation': 100.0, 'dt_translation': 1.0 }
  model.compile(optimizer='adagrad', loss=losses, loss_weights=loss_weights)
  return model

def test_performance(model, n=30):
  print("initial run")
  model.predict(np.zeros([BATCH_SIZE, TS_LEN, HEIGHT, WIDTH, CHANNELS]))

  total_ms = 0.0
  values = []
  print("0/", n, sep="", end="", flush=True)
  for i in range(n):
    begin = datetime.now()
    var = model.predict(np.zeros([BATCH_SIZE, TS_LEN, HEIGHT, WIDTH, CHANNELS]))
    end = datetime.now()
    elapsed = end - begin
    values.append(elapsed.seconds * 1000 + elapsed.microseconds / 1000)
    print("\r", i + 1, "/", n, sep="", end="", flush=True)
  print("\nmedian time per run:", round(np.median(values) if n > 0 else 0, 2), "ms")

weights_file = "rtdeepvo.h5"

tf.keras.backend.set_learning_phase(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
model = build_rcnn()
model.save_weights("deepvo.h5")
print(model.summary())
test_performance(model, 30)


