import subprocess
from subprocess import STDOUT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# subprocess.run([
#     'python', '/home/qindafei/KX/FID/src/pytorch_fid/fid_score.py',
#     '--save-stats',
#     '/home/qindafei/KX/data/cifar',
#     '/home/qindafei/KX/FID/stats/cifar_100.npz',
#     '--batch-size', '100',
#     '--num_samples', '100'
# ], stderr=STDOUT)

subprocess.run([
    'python', '/home/qindafei/KX/FID/src/pytorch_fid/fid_score.py',
    '/home/qindafei/KX/data/cifar',
    '/home/qindafei/KX/data/cifar',
    '--batch-size', '100',
    '--num_samples', '10000'
], stderr=STDOUT)

# subprocess.run([
#     'python', '/home/qindafei/KX/FID/src/pytorch_fid/fid_score.py',
#     '/home/qindafei/KX/FID/stats/cifar_10000_stat.npz',
#     '/home/qindafei/KX/image_diffusion/baseline/samples_10000x32x32x3.npz',
#     '--batch-size', '100',
#     '--num_samples', '10000'
# ], stderr=STDOUT)