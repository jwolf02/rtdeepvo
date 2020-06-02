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

WIDTH = 256
HEIGHT = 192
CHANNELS = 6

BATCH_SIZE = 2
TS_LEN = 20

def euclidean_distance(y_true, y_pred):
  return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def build_rcnn(trainable_encoder_layers=[False, False, False, False, False, False, False, False, False]):
  print("building rcnn model")
  
  input_layer = keras.Input(batch_shape=(BATCH_SIZE, TS_LEN, HEIGHT, WIDTH, CHANNELS), name="input")
  x = TimeDistributed(Conv2D(64, (7, 7), strides=(2, 2), padding="same", name="conv1"), name="dt_conv1", trainable=trainable_encoder_layers[0])(input_layer)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky1"), name="dt_leaky1")(x)
  x = TimeDistributed(Conv2D(128, (5, 5), strides=(2, 2), padding="same", name="conv2"), name="dt_conv2", trainable=trainable_encoder_layers[1])(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky2"), name="dt_leaky2")(x)
  x = TimeDistributed(Conv2D(256, (5, 5), strides=(2, 2), padding="same", name="conv3"), name="dt_conv3", trainable=trainable_encoder_layers[2])(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky3"), name="dt_leaky3")(x)
  x = TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv3_1"), name="dt_conv3_1", trainable=trainable_encoder_layers[3])(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky3_1"), name="dt_leaky3_1")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv4"), name="dt_conv4", trainable=trainable_encoder_layers[4])(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky4"), name="dt_leaky4")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv4_1"), name="dt_conv4_1", trainable=trainable_encoder_layers[5])(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky4_1"), name="dt_leaky4_1")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv5"), name="dt_conv5", trainable=trainable_encoder_layers[6])(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky5"), name="dt_leaky5")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_1"), name="dt_conv5_1", trainable=trainable_encoder_layers[7])(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky5_1"), name="dt_leaky5_1")(x)
  x = TimeDistributed(Conv2D(1024, (3, 3), strides=(2, 2), padding="same", name="conv6"), name="dt_conv6", trainable=trainable_encoder_layers[8])(x)
  x = TimeDistributed(Flatten(name="flatten"), name="dt_flatten")(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(100, return_sequences=True, stateful=True, name="lstm1")(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(100, return_sequences=True, stateful=True, name="lstm2")(x)
  trans = TimeDistributed(Dense(2, name="translation"), name="dt_translation")(x)
  rot = TimeDistributed(Dense(1, name='rotation'), name="dt_rotation")(x)
  model = keras.Model(inputs=[input_layer], outputs=[trans, rot], name='RTDeepVO')
  losses = { 'dt_rotation': 'mse', 'dt_translation': euclidean_distance }
  loss_weights = { 'dt_rotation': 100.0, 'dt_translation': 1.0 }
  model.compile(optimizer='adagrad', loss=losses, loss_weights=loss_weights)
  return model

def eval_rcnn(base_dir, model):
  print("evaulating model")
  with open("val_loss.txt", "a") as f:
    model.reset_states()
    frames, t, r = load_batch(base_dir, "06", "07", 160)
    loss = model.evaluate(frames, {"dt_translation": t, "dt_rotation": r}, batch_size=BATCH_SIZE, verbose=1)
    f.write(str(loss[1]) + " " + str(loss[2]) + "\n")

def load_batch(base_dir, seq0, seq1, num):
  idx0 = random.randrange(0, dataset_len[seq0] - (num + 5))
  frames0 = load_kitti_frames(base_dir, seq0, idx0, idx0 + num).reshape([-1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
  t0, r0 = load_kitti_poses(base_dir, seq0, idx0, idx0 + num)
  t0.reshape([-1, TS_LEN, 2])
  r0.reshape([-1, TS_LEN, 1])
  idx1 = random.randrange(0, dataset_len[seq1] - (num + 5))
  frames1 = load_kitti_frames(base_dir, seq1, idx1, idx1 + num).reshape([-1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
  t1, r1 = load_kitti_poses(base_dir, seq1, idx1, idx1 + num)
  t1.reshape([-1, TS_LEN, 2])
  r1.reshape([-1, TS_LEN, 1])
  frames = np.stack([frames0, frames1], axis=0).reshape([-1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
  t = np.stack([t0, t1], axis=1).reshape([-1, TS_LEN, 2])
  r = np.stack([r0, r1], axis=1).reshape([-1, TS_LEN, 1])
  return frames, t, r

def train_rcnn(base_dir, model, weights_file):
  seqs = ["00", "02", "05", "08", "09"]
  eval_rcnn(base_dir, model)
  while True:
    for i in range(len(seqs)):
      model.reset_states()
      print("training on sequence", seqs[i])
      frames, t, r = load_batch(base_dir , seqs[i], seqs[(i + 1) % 5], 160)
      model.fit(frames, { "dt_translation": t, "dt_rotation": r }, batch_size=BATCH_SIZE, epochs=4, verbose=1)
    model.save_weights(weights_file)
    eval_rcnn(base_dir, model)

if len(sys.argv) < 3:
  print("Usage:", sys.argv[0], "<Kitti dir> <weights file> [--train-encoder]")
  exit(1)

kitti_dir = sys.argv[1]
weights_file = sys.argv[2]
train_encoder = len(sys.argv) > 3 and sys.argv[3] == "--train-encoder"

random.seed()
model = build_rcnn(train_encoder)
model.build([BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
model.load_weights(weights_file)
print(model.summary())

train_rcnn(kitti_dir, model, weights_file)

