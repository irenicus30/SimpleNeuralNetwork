import numpy as np

def lin(x, deriv=False):
  if(deriv==True):
     return np.ones(x.shape)
  return x

X = np.array([
      [0,0],
      [0,1],
      [1,0],
      [1,1]
    ])

y = np.array([
      [0], 
      [1], 
      [1], 
      [0]
    ])

np.random.seed(1)

syn0 = 2*np.random.random((2,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1


print("This are weights syn0 when the training is finished")
print("syn0")
print(syn0)

print("This are weights syn1 when the training is finished")
print("syn1")
print(syn1)


#training step
for j in range(100000):  
    
    l0 = X
    l1 = lin(np.dot(l0, syn0))
    l2 = lin(np.dot(l1, syn1))
    # print(l1) 
    l2_error = y - l2
    if(j % 10000) == 0: 
        print("Error: " + str(np.mean(np.abs(l2_error))))
        
    l2_delta = l2_error*lin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * lin(l1,deriv=True)
    
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    

print("This are weights syn0 when the training is finished")
print("syn0")
print(syn0)

print("This are weights syn1 when the training is finished")
print("syn1")
print(syn1)

print("This is input")
print("X")
print(X)

print("This are intermediate values when the training is finished")
print("l1")
print(l1)

print("This is the output when the training is finished")
print("l2")
print(l2)

print("This is output")
print("y")
print(y)
