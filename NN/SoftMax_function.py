import math
import numpy as np
layer_output = [[4.8, 1.21, 2.385],
                [4.2, 1, 2.5],
                [4, 9.8, -2.385]]

print(np.max(layer_output))

exp_vals = np.exp(layer_output)
norm_base = exp_vals / np.sum(exp_vals, axis= 1, keepdims=True)
#usage of axis depicts we are adding by rows
#usage of keepdims keeps our dimensions of new outputs as previous outputs, 3 lists in a list
print(norm_base)





