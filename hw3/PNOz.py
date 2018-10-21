
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# Modified by Kevin S. Xu, 2017

# The Palmerston North Ozone time series example

import matplotlib.pyplot as pl
import numpy as np

PNoz = np.loadtxt('PNoz.txt',skiprows=1)

# Plot raw time series data
pl.ion()
pl.plot(np.arange(np.shape(PNoz)[0]),PNoz[:,2],'.')
pl.xlabel('Time (Days)')
pl.ylabel('Ozone (Dobson units)')

# Normalise data
PNoz[:,2] = PNoz[:,2]-PNoz[:,2].mean()
PNoz[:,2] = PNoz[:,2]/PNoz[:,2].max()

# Assemble input vectors
t = 1
k = 7

lastPoint = np.shape(PNoz)[0]-t*(k+1)
inputs = np.zeros((lastPoint,k))
targets = np.zeros((lastPoint,1))
for i in range(lastPoint):
    inputs[i,:] = PNoz[i:i+t*k:t,2]
    targets[i] = PNoz[i+t*(k+1),2]
    
test = inputs[-800:,:]
testTargets = targets[-800:]
train = inputs[:-800:2,:]
trainTargets = targets[:-800:2]
valid = inputs[1:-800:2,:]
validTargets = targets[1:-800:2]

# Train the network
import mlp
net = mlp.mlp(train,trainTargets,3,outtype='linear')
net.earlystopping(train,trainTargets,valid,validTargets,0.25)

test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
testOutMlp = net.mlpfwd(test)

pl.figure()
pl.plot(np.arange(np.shape(test)[0]),testOutMlp,'.')
pl.plot(np.arange(np.shape(test)[0]),testTargets,'x')
pl.legend(('Predictions','Targets'))
print(0.5*np.sum((testTargets-testOutMlp)**2))
pl.show()
