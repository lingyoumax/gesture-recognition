import torch
from tools import *
import matplotlib.pyplot as plt

data = plt.imread('test.jpg')
'''
data=torch.from_numpy(data)
data=distinguish_otsu_gray_2(data)
'''
plt.imshow(data)
plt.show()