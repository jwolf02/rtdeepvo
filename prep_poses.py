#! /usr/bin/python3
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
  print("Usage:", sys.argv[0], "<orig poses file> <output poses file> [--plot-output]")
  exit(1)

orig_poses_file = sys.argv[1]
out_poses_file = sys.argv[2]
plot_output = len(sys.argv) > 3 and sys.argv[3] == "--plot-output"

poses = []

with open(orig_poses_file, "r") as f:
  lines = f.readlines()
		
  for line in lines:
    m = np.fromstring(line, dtype=float, sep=' ')
    m = m.reshape(3, 4)
    r = R.from_matrix(m[0:3, 0:3]).as_euler("yxz", degrees=False)
    t = np.reshape(m[0:3, 3:4], (3,))
    poses.append([t[0], t[1], t[2], r[1], r[0], r[2]])
    
with open(out_poses_file, "w") as f:
  for p in poses:
    line = str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + " " + str(p[4]) + " " + str(p[5]) + "\n"
    f.write(line)

np_poses = np.asarray(poses)
print("trans (max):", np_poses[:, 0:3].max(axis=0))
print("trans (min):", np_poses[:, 0:3].min(axis=0))
print("rot (max)  :", np_poses[:, 3:6].max(axis=0))
print("rot (min)  :", np_poses[:, 3:6].min(axis=0))

if plot_output:
  plot1 = plt.figure(1)
  plt.plot([[p[0], p[1], p[2]] for p in poses])
  plot2 = plt.figure(2)
  plt.plot([[p[3], p[4], p[5]] for p in poses])
  plt.show()
    
print("done")
  

