
import numpy as np

l = np.array([1, 2, 3, 4, -1])
print(l.shape)

l_new = (l > 0).astype(float)
print(l_new)