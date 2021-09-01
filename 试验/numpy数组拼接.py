import numpy as np

a=np.array([[1,2],[3,4],[5,6]])

print(a)

c=[]
print(c)
for i,b in enumerate(a):
    #c=np.c_[c,b]
    print(b)
    # print('befor c:',c)
    # print('b:',b)
    # c=np.concatenate((c,b),axis=0)
    # print('after c:', c)
    c.append(b)

c=np.concatenate(c,axis=0)
print(c)
