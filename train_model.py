#! /usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, TimeDistributed, Flatten, Dense, LSTM, MaxPool2D, LeakyReLU, Dropout, BatchNormalization, AveragePooling2D
from datetime import datetime
import cv2
from scipy.spatial.transform import Rotation as R
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
import random
import sys
import kitti
import matplotlib.pyplot as plt

WIDTH = 256
HEIGHT = 192
CHANNELS = 6

BATCH_SIZE = 4
TS_LEN = 15

SEQ_SIZE = 600

def euclidean_distance(y_true, y_pred):
  return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
  
def conv(x, name, filters, size, stride, dropout, batch_norm, activation=True, trainable=True):
  if batch_norm:
    x = TimeDistributed(Conv2D(filters, (size, size), strides=(stride, stride), padding="same", name=name, 
      use_bias=False, trainable=trainable), name="dt_" + name)(x)
    if activation:
      x = TimeDistributed(LeakyReLU(0.1, name="leaky_" + name), name="dt_leaky_" + name)(x)
    x = TimeDistributed(BatchNormalization(trainable=trainable, name="bn_" + name), 
      name="dt_bn_" + name)(x)
  else:
    x = TimeDistributed(Conv2D(filters, (size, size), strides=(stride, stride), padding="same", name=name, 
      trainable=trainable), name="dt_" + name)(x)
    if activation:
      x = TimeDistributed(LeakyReLU(0.1, name="leaky_" + name), name="dt_leaky_" + name)(x)
  return TimeDistributed(Dropout(dropout, name="dropout_" + name), name="dt_dropout_" + name)(x)
  
def rnn(x, num_states, num_layers, dropout):
  for i in range(num_layers):
    x = tf.compat.v1.keras.layers.CuDNNLSTM(num_states, return_sequences=True, stateful=True, name="lstm" + str(i + 1))(x)
  x = TimeDistributed(Dropout(dropout, name="dropout_lstm_out"), name="dt_dropout_lstm_out")(x)
  return x

def build_rcnn(batch_size=BATCH_SIZE, ts_len=TS_LEN, batch_norm=True, trainable=False):
  print("building rcnn model")  
  input_layer = keras.Input(batch_shape=(batch_size, ts_len, HEIGHT, WIDTH, CHANNELS), name="input")
  x = conv(input_layer, "conv1", 64, 7, 2, 0.2, batch_norm, trainable=trainable)
  x = conv(x, "conv2", 128, 5, 2, 0.2, batch_norm, trainable=trainable)
  x = conv(x, "conv3", 256, 5, 2, 0.2, batch_norm, trainable=trainable)
  x = conv(x, "conv3_1", 256, 3, 1, 0.2, batch_norm, trainable=trainable)
  x = conv(x, "conv4", 512, 3, 2, 0.2, batch_norm, trainable=trainable)
  x = conv(x, "conv4_1", 512, 3, 1, 0.2, batch_norm, trainable=trainable)
  x = conv(x, "conv5", 512, 3, 2, 0.2, batch_norm, trainable=trainable)
  x = conv(x, "conv5_1", 512, 3, 1, 0.2, batch_norm, trainable=trainable)
  x = conv(x, "conv6", 1024, 3, 2, 0.5, batch_norm, activation=False, trainable=trainable)
  x = TimeDistributed(AveragePooling2D(pool_size=(3, 4), name="gap"), name="dt_gap")(x)
  x = TimeDistributed(Flatten(name="flatten"), name="dt_flatten")(x)
  x = rnn(x, 1000, 2, 0.5)
  trans = TimeDistributed(Dense(2, name="translation"), name="dt_translation")(x)
  rot = TimeDistributed(Dense(1, name='rotation'), name="dt_rotation")(x)
  model = keras.Model(inputs=[input_layer], outputs=[trans, rot], name='RTDeepVO')
  losses = { 'dt_rotation': 'mae', 'dt_translation': 'mse' }
  loss_weights = { 'dt_rotation': 100.0, 'dt_translation': 1.0 }
  model.compile(optimizer='adagrad', loss=losses, loss_weights=loss_weights, metrics={"dt_translation": euclidean_distance, "dt_rotation": 'mae'})
  return model
    
def load_sample_sequence(base_dir, seq, size, offset=0, rand=False, start_from_zero=True):
  start_frame = random.randrange(kitti.SEQ_LEN[seq] - size - size - 1) if rand else offset
  print("loading sequence", seq, "starting with frame", start_frame)
  frames = kitti.load_frames(base_dir + "/sequences", seq, start_frame, start_frame + size)
  t, r = kitti.load_poses(base_dir + "/poses", seq, start_frame, start_frame + size, start_from_zero=start_from_zero)
  frames = frames.reshape([-1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
  t = t.reshape([-1, TS_LEN, 2])
  r = r.reshape([-1, TS_LEN, 1])
  return frames, t, r
  
def load_sample_batch(base_dir):
  frames = []
  trans = []
  rot = []
  seqs = ["00", "02", "05", "08", "09"]
  for i in range(BATCH_SIZE):
    seq = seqs[random.randrange(0, len(seqs))]
    f, t, r = load_sample_sequence(base_dir, seq, SEQ_SIZE//BATCH_SIZE, rand=True)
    frames.append(f)
    trans.append(t)
    rot.append(r)
  frames = np.stack(frames, axis=1).reshape([-1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
  t = np.stack(trans, axis=1).reshape([-1, TS_LEN, 2])
  r = np.stack(rot, axis=1).reshape([-1, TS_LEN, 1])
  return frames, t, r

def eval_model(base_dir, model):
  print("---------------- EVAL MODEL ----------------")
  model.reset_states()
  loss = []
  for offset in range(0, kitti.SEQ_LEN['06'] - 150, 150):
    frames = []
    trans = []
    rot = []
    for seq in ['06', '07']:
      f = kitti.load_frames(base_dir + "/sequences", seq, offset, offset + 150)
      t, r = kitti.load_poses(base_dir + "/poses", seq, offset, offset + 150, start_from_zero=False)
      frames.append([f, f])
      trans.append([t, t])
      rot.append([r, r])
    frames = np.stack(frames, axis=1).reshape([-1, TS_LEN, HEIGHT, WIDTH, CHANNELS])
    t = np.stack(trans, axis=1).reshape([-1, TS_LEN, 2])
    r = np.stack(rot, axis=1).reshape([-1, TS_LEN, 1])
    l = model.evaluate(frames, {'dt_translation': t, 'dt_rotation': r}, verbose=1, batch_size=BATCH_SIZE)
    loss.append([l[3], l[4]])
  t_loss = 0.0
  r_loss = 0.0
  for l in loss:
    t_loss += l[0]
    r_loss += l[1]
  t_loss /= len(loss)
  r_loss /= len(loss)
  with open("val_loss.txt", "a") as f:
    f.write(str(t_loss) + " " + str(r_loss) + "\n")
  print("t_loss:", t_loss, "r_loss:", r_loss)

def push_changes():
  os.system('git add . && git commit -m "some message" && git push')

def train_rcnn(base_dir, model, weights_file):
  #eval_model(base_dir, model)
  while True:
    for _ in range(5):
      for _ in range(20):
        model.reset_states()
        frames, t, r = load_sample_batch(base_dir)
        model.fit(frames, { "dt_translation": t, "dt_rotation": r }, batch_size=BATCH_SIZE, epochs=2, verbose=1)
      print("--------------- SAVING MODEL ---------------")
      model.save_weights(weights_file)
    eval_model(base_dir, model)
    push_changes()

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage:", sys.argv[0], "<kitti dir> <weights file> [--train-encoder, --batch-norm]")
    exit(1)

  kitti_dir = sys.argv[1]
  weights_file = sys.argv[2]
  train_encoder = False
  batch_norm = False
  for i in range(3, len(sys.argv)):
    arg = sys.argv[i]
    if arg == "--train-encoder":
      train_encoder = True
    elif arg == "--batch-norm":
      batch_norm = True

  random.seed()
  model = build_rcnn(BATCH_SIZE, TS_LEN, batch_norm=batch_norm, trainable=train_encoder)
  model.load_weights(weights_file)
  print(model.summary())

  train_rcnn(kitti_dir, model, weights_file)

