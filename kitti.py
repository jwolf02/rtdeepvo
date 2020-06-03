import numpy as np
import cv2
import math
from datetime import datetime

WIDTH = 256
HEIGHT = 192
CHANNELS = 6

MEAN = np.asarray([0.4418668, 0.4422875, 0.41850266])

SEQ_LEN = { 
  "00": 4541, 
  "02": 4661, 
  "05": 2761, 
  "06": 1091, 
  "07": 1091, 
  "08": 4071, 
  "09": 1591 
}

def map_angle(x):
  if x < -180.:
    x += 360.
  elif x > 180.:
    x -= 360.
  return x / 180. * math.pi

def load_poses(kitti_poses_dir, sequence, begin=0, end=1000000, start_from_zero=True):
  """
  load converted poses from the kitti dataset
  """
  end = min(SEQ_LEN[sequence], end)
  with open(kitti_poses_dir + '/' + sequence + '.txt', 'r') as f:
    lines = f.readlines()
		
    r = []
    t = []
    for line in lines:
      pose = [float(x) for x in line.split(' ')]
      # pose = "x y z rot(x) rot(y) rot(z)
      t.append(np.asarray([pose[0], pose[2]]))
      r.append(pose[4])
  t_out = np.asarray(t[begin:end]) - (np.asarray(t[begin - 1]) if begin - 1 > 0 and start_from_zero else np.asarray([0.0, 0.0]))
  r_out = []
  for i in range(begin, end):
    r_d = r[i] - (r[begin - 1] if begin - 1 > 0 and start_from_zero else 0.0)
    if r_d < -math.pi:
      r_d += 2 * math.pi
    elif r_d > math.pi:
      r_d -= 2 * math.pi
    r_out.append(r_d)
  return t_out, np.asarray(r_out)

def load_frame(kitti_sequence_dir, sequence, i, preprocess=True):
  iname = kitti_sequence_dir + "/" + sequence + "/" + str(i).zfill(6) + '.png'
  frame = cv2.imread(iname)
  if frame is None or frame.shape != (HEIGHT, WIDTH, 3):
    raise Exception("failed to read frame " + iname)
  return (frame / 255.) - MEAN if preprocess else frame

def load_frames(kitti_sequence_dir, sequence, begin=0, end=1000000):
  last_frame = load_frame(kitti_sequence_dir, sequence, begin) # load initial frame
  end = min(SEQ_LEN[sequence], end)
  frames = np.empty([end - begin, HEIGHT, WIDTH, CHANNELS])
  stime = datetime.now()
  print("\r", 0, "/", end - begin, sep="", end="", flush=True)
  for i in range(begin + 1, end + 1):
    frame = load_frame(kitti_sequence_dir, sequence, i)
    frames[i - (begin + 1), :, :, :] = np.concatenate([last_frame, frame], axis=-1)
    last_frame = frame
    print("\r", i - begin, "/", end - begin, sep="", end="", flush=True)
  etime = datetime.now()
  elapsed_time = etime - stime
  millis = (elapsed_time.seconds * 1000 + elapsed_time.microseconds // 1000)
  print("\rdone loading ", end - begin, " frames from sequence ", sequence, " in ", millis, " ms (", millis / (end - begin), " ms/frame)", sep='')

  return np.asarray(frames)



