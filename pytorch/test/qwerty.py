import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

#----- data generation ----------
cuda_device = torch.device('cuda:0')

import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()
# sns.pairplot(iris, hue='species')
# plt.show()
iris = iris.sample(frac=1)
data = torch.tensor( iris[iris.columns[0:4]].values, device=cuda_device).float()
labels = torch.zeros(len(data), dtype=torch.long, device=cuda_device)
labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2
#------------------------------------------


#------------ Neural Network ------------------

class classANN(nn.Module):
    #layer creation
    def __init__(self, hiddenode, hiddenlayer):
        super().__init__()

        self.layers = nn.ModuleDict().cuda('cuda:0')
        self.hiddenlayer = hiddenlayer

        #input layer
        self.layers['input'] = nn.Linear(4,hiddenode, device=cuda_device)

        #hidden layers
        for i in range(hiddenlayer):
            self.layers[f'hidden{i}'] = nn.Linear(hiddenode,hiddenode, device=cuda_device)

        # output layer
        self.layers['output'] = nn.Linear(hiddenode,3, device=cuda_device)
     
    #forward propogation
    def forward(self, x):
        
        #input
        x = self.layers['input'](x)

        #sigmoid
        x = F.relu(x).cuda('cuda:0')

        #hidden layer
        for i in range(self.hiddenlayer):
            x = F.relu( self.layers[f'hidden{i}'](x) ).cuda('cuda:0')

        #output
        x = self.layers['output'](x)

        # #sigmoid
        # x = torch.sigmoid(x)
        
        return x


#----------------------------------------------------

#------------ Training -----------------------
def train(ANN):
    #loss function
    lossfun = nn.CrossEntropyLoss().cuda('cuda:0')
    # optimizer

    numepochs = 10000
    optimizer = torch.optim.SGD(ANN.parameters(),lr=.01)
    # initialize losses
    losses = torch.zeros(numepochs, device=cuda_device)
    # loop over epochs
    for epochi in range(numepochs):
        # forward pass
        yHat = ANN(data)
        # compute loss
        loss = lossfun(yHat,labels)
        losses[epochi] = loss
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #-------- final output -----------
            
    # final forward pass
    predictions = ANN(data)
    predlabels = predictions >.5

    #calculate accuracy
    predlabels = torch.argmax(predictions,axis=1)
    print(predlabels)
    totalacc = 100*torch.mean((predlabels == labels).float()).cuda('cuda:0')
    return totalacc


layers = range(1,6)                             # number of hidden layers
units  = torch.arange(1,101).cuda('cuda:0')     # units per hidden layer

accuracies  = torch.zeros((len(units),len(layers))).cuda('cuda:0')

for unitidx in range(len(units)):
  for layeridx in range(len(layers)):

    # create a fresh model instance
    net = classANN(units[unitidx],layers[layeridx]).cuda('cuda:0')

    # run the model and store the results
    acc = train(net).cuda('cuda:0')
    accuracies[unitidx,layeridx] = acc

fig = plt.figure(figsize=(5,5))
plt.plot(units,accuracies,'o-',markerfacecolor='w',markersize=9)
plt.plot(units[[0,-1]],[33,33],'--',color=[.8,.8,.8])
plt.plot(units[[0,-1]],[67,67],'--',color=[.8,.8,.8])
plt.legend(layers)
plt.xlabel('Hidden Units')
plt.ylabel('Accuracy')
plt.show()