from math import e, log
import numpy as np

softmax_output = [0.7, 0.1, 0.2]
one_hot = [1,0,0]

list_l = [1,2,3,4,5]
print(list_l[0:3:])

y = np.array([1,2,3,4,5])
print(y)

for i in range(0,10):
    value = i + 1

print(value)




new_arr = np.array([2,3])
print(new_arr)
new_shape  = new_arr.reshape(2,1)
print(new_shape)

print("-------------------------------------------------")
print(np.random.rand(3,4))
"""
#wherver the index is  = 1 means thats where our true value lies, we want to see how wrong we were
for i in one_hot:
    if i == 1:
        index = one_hot.index(i)
        predicted_value = softmax_output[index]


print(predicted_value)
                
#calculate the loss
#-(actual * log(predicted)) ---> -log(predicted)   as actual will always be 1

print(f"cross entropy loss value is : {-log(predicted_value)}")     
"""

            


