import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import pandas as pd
display.set_matplotlib_formats('svg')

ANN = nn.Sequential(
    nn.Linear(2, 1),   # Input Layer
    nn.ReLU(),          # activation
    nn.Linear(1, 1),  # hidden Layer
    nn.Sigmoid()
)

lossfun = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(ANN.parameters(), lr=.01)

#final forward padd



fig, ax = plt.subplots(1,2,figsize=(13, 4))
# create data

# numepochs = 1000
# losses = torch.zeros(numepochs)
# ongoingAcc = []
# for epochi in range(numepochs):
#     # forward pass
#     yHat = ANN(data)

#     #compute loss
#     loss = lossfun(yHat, labels)
#     losses[epochi] = loss

#     # backprop
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     # end training model

#     # compute accuracy
#     matches = torch.argmax(yHat, axis=1) == labels
#     matchesNumeric = matches.float()
#     accuracyPct = 100*torch.mean(matchesNumeric)
#     ongoingAcc.append( accuracyPct )

# #final forward padd
# prdeictions = ANN(data)

# predlabels = torch.argmax(prdeictions, axis=1)
# totalacc = 100*torch.mean((predlabels == labels).float())

# print('Final accuracy: %g%%' %totalacc)

nPerClust = 100
blur = 1

A = [  1, 1 ]
B = [  5, 1 ]

# generate data
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

# concatanate into a matrix
data_np = np.hstack((a,b)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show the data
fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs')
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko')
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.ylabel('qwerty dimension 2')
plt.show()

ax[1].plot(ongoingAcc)
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].set_title('Accuracy')
plt.show()

