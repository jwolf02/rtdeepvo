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

def conv(x, name, filters, size, stride, activation='relu', trainable=True):
  x = TimeDistributed(Conv2D(filters, (size, size), strides=(stride, stride), padding="same", name=name, 
    use_bias=False, activation=activation, trainable=trainable), name="dt_" + name)(x)
  return TimeDistributed(BatchNormalization(trainable=trainable, name="bn_" + name), 
    name="dt_bn_" + name)(x)

def build_rcnn(trainable=False):
  print("building rcnn model")
  
  input_layer = keras.Input(batch_shape=(1, 1, HEIGHT, WIDTH, CHANNELS), name="input")
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
  #x = TimeDistributed(Dense(256, activation='relu', name="dense1"), name="dt_dense1")(x)
  x = LSTM(100, return_sequences=True, stateful=True, name="lstm1", trainable=False)(x)
  x = LSTM(100, return_sequences=True, stateful=True, name="lstm2", trainable=False)(x)
  trans = TimeDistributed(Dense(2, name="translation", trainable=False), name="dt_translation")(x)
  rot = TimeDistributed(Dense(1, name='rotation', trainable=False), name="dt_rotation")(x)
  model = keras.Model(inputs=[input_layer], outputs=[trans, rot], name='RTDeepVO')
  losses = { 'dt_rotation': 'mse', 'dt_translation': euclidean_distance }
  model.compile(optimizer='adagrad', loss=losses)
  return model
  
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage:", sys.argv[0], "<weights file> <output file>")
    exit(1)
    
  model = build_rcnn(1, 1)
  print(model.summary())
  model.load_weights(sys.argv[1])
  
