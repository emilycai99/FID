import numpy as np

# x = np.load('/home/qindafei/KX/FID/stats/cifar.npz')
x = np.load('/home/qindafei/KX/image_diffusion/baseline/samples_10000x32x32x3.npz',
            mmap_mode='r')
for k in x.files:
    print(k)
