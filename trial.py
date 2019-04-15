import numpy as np
import matplotlib.pyplot as plt
from math import pi
l_min = 5
l_max = 15


factor = (l_max - l_min) * 2.0 / (pi)

l_values = np.arange(0, l_max + 1)
l_mask = np.zeros(len(l_values))
def mask(l):
    if(l < l_min):
        return 1.0
    else:
        arg = (l - l_min)/factor
        # print(arg, l)
        return pow(np.cos(arg), 1.0/8.0)


for i, l in enumerate(l_values):
    l_mask[i] = mask(l)

print(l_values, l_mask)

plt.plot(l_values, l_mask)
plt.savefig("pic.png")
