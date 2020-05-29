# rtdeepvo
Bachelor Thesis Project for making DeepVO working in real time

## Download link for pretrained FlowNet I used
https://drive.google.com/drive/folders/0B5EC7HMbyk3CbjFPb0RuODI3NmM

## Link to the author's paper
https://arxiv.org/abs/1709.08429

## Info
It has been tried to make DeepVO work multiple times but to my knowledge nobody  
made it work unless for some small toy dataset, like moving forward and backward.  
I cannot prove the author's faked something about their paper and I recognized  
some serious errors in the code of the other guys who tries to implement it.  
The original network architecture featured and input size of 1280x384 which leads  
to an encoder output size of 6x20x1024 and two LSTM layers each with 1000 cells.  
This makes the network contain over 500M variables when the training dataset  
containing less than 15K images each with a label of effectively 3 degrees of freedom.  
Even with dropout, early stopping and data augmentation I do not believe it is possible  
to train such a network (AlexNet has 60M variables trained on a dataset with 10M images  
each having a 1000 class label, just for comparison).  
  
So I propose an architecture with an input size of 384x256 and LSTMs with 256 cells which  
makes the network contain ~40M variables of which 14M come pretrained from FlowNet.  
This leads to a smalle enough network that it can be run in real time on a reasonably  
sized computer (5 FPS on the Nvidia Jetson Nano, 20 FPS on Intel i5, 45 FPS on GPU).  
The original architecture was so big that I could only make it run on my 16GB laptop  
where it took 1200ms to run one iteration (compared to 45ms of my architecture).  
It was too big for both my Desktop computer with 8GB RAM and 5GB GPU-RAM and the  
Jetson Nano with just 4GB shared RAM. The parameters itself take up 2GB of memory.   

The authors did not publish their code but there is a pretrained model flowing around  
the internet which I have not tried out yet.  
You can find it here: https://drive.google.com/file/d/1l0s3rYWgN8bL0Fyofee8IhN-0knxJF22/view  
Maybe I am wrong but there is a strong indication that there is something wrong about this paper.  
It's just too straight forward for all the problems I and all the other who tried to  
reimplement it faced.
