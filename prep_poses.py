import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys

POSES_FILE = "/home/jwolf/kitti/orig_poses/"

if len(sys.argv) < 2:
  print("missing sequence")
  exit(1)

sequence = sys.argv[1]

poses = [[0.0, 0.0, 0.0]]

with open(POSES_FILE + sequence + ".txt", "r") as f:
  lines = f.readlines()
		
  for line in lines:
    m = np.fromstring(line, dtype=float, sep=' ')
    m = m.reshape(3, 4)
    r = R.from_matrix(m[0:3, 0:3]).as_euler("yzx", degrees=True)
    t = np.reshape(m[0:3, 3:4], (3,))
    poses.append([t[0], t[2], r[0]])
    
with open("/home/jwolf/" + sequence+ ".txt", "w") as f:
  for p in poses:
    line = str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + "\n"
    f.write(line)
    
plt.plot([p[2] for p in poses])#[p[0] for p in poses], [p[1] for p in poses])
plt.show()
    
print("done")
  

