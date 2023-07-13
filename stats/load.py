import numpy as np
from PIL import Image
import random

# x = np.load('/home/qindafei/KX/FID/stats/cifar.npz')
x = np.load('/home/qindafei/KX/image_diffusion/baseline/samples_10000x32x32x3.npz',
            mmap_mode='r')
for k in x.files:
    print(k)

for i in random.sample(range(x[x.files[0]].shape[0]),10):
    y = x[x.files[0]][i]
    img = Image.fromarray(y, 'RGB')
    img.save('/home/qindafei/KX/FID/stats/sample/sample{}.png'.format(i))


    