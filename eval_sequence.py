import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, TimeDistributed, Flatten, Dense, LSTM, MaxPool2D, LeakyReLU, Dropout
from datetime import datetime
import cv2
from scipy.spatial.transform import Rotation as R
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
import random
import sys
import kitti

WIDTH = 384
HEIGHT = 256
CHANNELS = 6

BATCH_SIZE = 1
TS_LEN = 50

def euclidean_distance(y_true, y_pred):
  return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def build_rcnn(batch_size=BATCH_SIZE, ts_len=TS_LEN):
  print("building rcnn model")
  
  input_layer = keras.Input(batch_shape=(batch_size, ts_len, HEIGHT, WIDTH, CHANNELS), name="input")
  x = TimeDistributed(Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1"), name="dt_conv1", trainable=False)(input_layer)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky1"), name="dt_leaky1")(x)
  x = TimeDistributed(Conv2D(128, (5, 5), strides=(2, 2), padding="same", name="conv2"), name="dt_conv2", trainable=False)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky2"), name="dt_leaky2")(x)
  x = TimeDistributed(Conv2D(256, (5, 5), strides=(2, 2), padding="same", name="conv3"), name="dt_conv3", trainable=False)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky3"), name="dt_leaky3")(x)
  x = TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv3_1"), name="dt_conv3_1", trainable=False)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky3_1"), name="dt_leaky3_1")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv4"), name="dt_conv4", trainable=False)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky4"), name="dt_leaky4")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv4_1"), name="dt_conv4_1", trainable=False)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky4_1"), name="dt_leaky4_1")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv5"), name="dt_conv5", trainable=False)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky5"), name="dt_leaky5")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_1"), name="dt_conv5_1", trainable=False)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky5_1"), name="dt_leaky5_1")(x)
  x = TimeDistributed(Conv2D(1024, (3, 3), strides=(2, 2), padding="same", name="conv6"), name="dt_conv6", trainable=False)(x)
  x = TimeDistributed(Flatten(name="flatten"), name="dt_flatten")(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(256, return_sequences=True, stateful=True, name="lstm1", trainable=False)(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(256, return_sequences=True, stateful=True, name="lstm2", trainable=False)(x)
  trans = TimeDistributed(Dense(2, name="translation"), name="dt_translation", trainable=False)(x)
  rot = TimeDistributed(Dense(1, name='rotation'), name="dt_rotation", trainable=False)(x)
  model = keras.Model(inputs=[input_layer], outputs=[trans, rot], name='RTDeepVO')
  losses = { 'dt_rotation': 'mse', 'dt_translation': euclidean_distance }
  model.compile(optimizer='adagrad', loss=losses)
  return model

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "<weights file> <kitti dir> <sequence>")
    exit(1)
    
    weights_file = sys.argv[1]
    kitti_dir = sys.argv[2]
    seq = sys.argv[3]

    model = build_rcnn()
    model.load_weights(weights_file)
    
    trans = []
    rots = []
    
    for i in range(0, kitti.SEQ_LEN[seq], TS_LEN):
      frames = kitti.load_poses(kitti_dir + "/sequence/", sequence, i, i + TS_LEN)
      t, r = kitti.load_frames(kitti_dir + "/poses/", sequence, i, i + TS_LEN, start_from_zero=False)
      frames.reshape([1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
      t.reshape([1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
      r.reshape([1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
      t_out, r_out = model.predict(frames)
      trans.append(t_out)
      rots.append(r_out)
      
    with open(sequence + "_eval.txt", "w") as f:
      for i in range(trans):
        f.write(str(trans[i, 0]) + " " + str(trans[i, 1]) + " " + str(rots[i, 0]) + "\n")

