import cv2
import os
import sys
from datetime import datetime

if len(sys.argv) < 3:
  print("Usage:", sys.argv[0], "<sequence_dir> <output dir>")
  exit(0)
  
sequence_dir = sys.argv[1]
output_dir = sys.argv[2]

i = 0
print("\r", 0, sep="", end="", flush=True)
stime = datetime.now()
for f in os.listdir(sequence_dir):
  frame = cv2.imread(sequence_dir + '/' + f)
  if frame is None:
    print("cannot load", f)
    continue
  #frame = frame[0:376, 280:844, 0:3]
  frame = frame[0:256, 21:362, 0:3] # resize frames from (256, 384) to (192, 256)
  frame = cv2.resize(frame, (256, 192)) # extract inner patch of size (564, 376) and resize to (384, 256)
  cv2.imwrite(output_dir + '/' + f, frame)
  i += 1
  print("\r", i, sep="", end="", flush=True)
etime = datetime.now()
elapsed_time = etime - stime
millis = elapsed_time.seconds * 1000 + elapsed_time.microseconds // 1000
print("\rdone processing ", i, " frames in ", millis, " ms (", round(millis / i, 2), " ms/frame)", sep='')
