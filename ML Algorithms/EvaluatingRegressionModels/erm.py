import math
import numpy as nm
l1=nm.array([2,3,4,5])
l2=nm.array([2,5,6,8])

def rmse(p,a):
    return math.sqrt((sum(p-a)**2)//len(a))
print(rmse(l1,l2))

