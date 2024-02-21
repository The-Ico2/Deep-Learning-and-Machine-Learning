import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn


def model(x, y):
    ANN = nn.Sequential( # creating NN layers
        nn.Linear(1, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    )
    
    lossFunc = nn.MSELoss() # claculates losses of the function
    optimizer = torch.optim.SGD(ANN.parameters(),lr=.05) # optimizez NN learning

    numepochs = 100
    losses = torch.zeros(numepochs) # array of zeroes == to number of epochs for losses to be "appended" to
    for epochi in range(numepochs): # for loop running the function
        # forward pass
        yHat = ANN(x) # data predicted by the NN model

        #compute loss
        loss = lossFunc(yHat, y) # loss calculation
        losses[epochi] = loss # "appending" calculated loss to the loss array

        # backprop
        optimizer.zero_grad() # unknown
        loss.backward() # backprop function
        optimizer.step() # forwardprop function
        # end training model

        #compute model predictions
        predictions = ANN(x)


        # output:
    return predictions, losses


def createData(m):
    N = 50
    x = torch.randn(N, 1)
    y = m*x + torch.randn(N, 1)/2
    return x, y

# create a dataset
x,y = createData(.8)

# run the model
yHat,losses = model(x,y)
fig,ax = plt.subplots(1,2,figsize=(12,4))

ax[0].plot(losses.detach(), 'o', markerfacecolor='w',linewidth=.1)
ax[0].set_xlabel('Epoch')
ax[0].set_title('Loss')

ax[1].plot(x,y,'bo',label='Real Data')
ax[1].plot(x,yHat.detach(),'rs',label='Predictions')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title(f'prediction-data corrr = {np.corrcoef(y.T,yHat.detach().T)[0,1]:.2f}')
ax[1].legend()

plt.show()