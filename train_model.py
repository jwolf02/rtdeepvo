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

WIDTH = 384
HEIGHT = 256
CHANNELS = 6

BATCH_SIZE = 2
TS_LEN = 20

MEAN = [0.4418668, 0.4422875, 0.41850266]

dataset_len = { "00": 4541, "02": 4661, "05": 2761, "06": 1091, "07": 1091, "08": 4071, "09": 1591 }

def load_frame(base_dir, sequence, i):
  iname = base_dir + "/sequences/" + sequence + "/" + str(i).zfill(6) + '.png'
  frame = cv2.imread(iname)
  if frame is None or frame.shape != (HEIGHT, WIDTH, 3):
    print("cannot load", iname)
    exit(1)
  return (frame / 255.) - MEAN

def load_kitti_frames(base_dir, sequence, begin=0, end=1000000):
  frames = []
  last_frame = load_frame(base_dir, sequence, begin) # load initial frame
  end = min(dataset_len[sequence], end)
  print("\r", 0, "/", end - begin, sep="", end="", flush=True)
  for i in range(begin + 1, end + 1):
    frame = load_frame(base_dir, sequence, i)
    frames.append(np.concatenate([last_frame, frame], axis=-1))
    last_frame = frame
    print("\r", i - begin, "/", end - begin, sep="", end="", flush=True)
  print("\rdone loading", end - begin, "frames from sequence", sequence)

  return np.asarray(frames)
  
def angle(x):
  if x < -180.:
    return x + 360.
  elif x > 180.:
    return x - 360.
  else:
    return x

def load_kitti_poses(base_dir, sequence, begin=0, end=1000000):
  end = min(dataset_len[sequence], end)
  with open(base_dir + '/poses/' + sequence + '.txt', 'r') as f:
    lines = f.readlines()
		
    r = []
    t = []
    for line in lines:
      tokens = line.split(' ')
      y = float(tokens[0])
      x = float(tokens[1])
      theta = float(tokens[2])
      t.append(np.asarray([x, y]))
      r.append(theta)
    t_out = []
    r_out = []
    for i in range(begin, end):
      t_out.append(t[i] - t[begin - 1] if begin > 0 else np.array([0.0, 0.0]))
      r_out.append(angle(r[i] - r[begin - 1] if begin > 0 else np.asarray([0.0])))
    return np.asarray(t_out), np.asarray(r_out)

def load_kitti_sequence(base_dir, sequence, begin, num):
  frames = load_kitti_frames(base_dir, sequence, begin, num)
  t, r = load_kitti_poses(base_dir, sequence, begin, num)
  return frames, t, r

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
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(2, 2), padding="same", name="conv5"), name="dt_conv5", trainable=True)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky5"), name="dt_leaky5")(x)
  x = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_1"), name="dt_conv5_1", trainable=True)(x)
  x = TimeDistributed(LeakyReLU(alpha=0.1, name="leaky5_1"), name="dt_leaky5_1")(x)
  x = TimeDistributed(Conv2D(1024, (3, 3), strides=(2, 2), padding="same", name="conv6"), name="dt_conv6", trainable=True)(x)
  x = TimeDistributed(Flatten(name="flatten"), name="dt_flatten")(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(256, return_sequences=True, stateful=True, name="lstm1")(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(256, return_sequences=True, stateful=True, name="lstm2")(x)
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
      model.fit(frames, { "dt_translation": t, "dt_rotation": r }, batch_size=BATCH_SIZE, epochs=1, verbose=1)
    model.save_weights("rtdeepvo_rcnn.h5")
    eval_rcnn(base_dir, model)

BASE_DIR = "/home/jwolf/kitti"

weights_file = sys.argv[1] if len(sys.argv) else "rtdeepvo.h5"

random.seed()
model = build_rcnn()
model.build([BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
model.load_weights(weights_file)
print(model.summary())

train_rcnn(BASE_DIR, model, weights_file)

