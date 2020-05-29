import numpy as np
import cv2
import sys

kitti_dir = "/home/jwolf/kitti/sequences/"

dataset_len = { "00": 4541, "02": 4661, "05": 2761, "06": 1101,"07": 1101, "08": 4071, "09": 1591 }

means = []
for seq in ["00", "02", "05", "08", "09"]:
	n = dataset_len[seq]
	print("0/", n, sep="", end="", flush=True)
	for i in range(n):
		iname = kitti_dir + seq + "/" + str(i).zfill(6) + '.png'
		frame = cv2.imread(iname)
		if frame is None:
			print("cannot load", iname)
			continue
		m = (frame.astype(np.float32) / 255.).mean(axis=(-3,-2), keepdims=2).reshape([3])
		means.append(m)
		print("\r", i + 1, "/", n, sep="", end="", flush=True)
	print("\n")

mean = np.asarray(means).mean(axis=(-2), keepdims=1)
print(mean)
