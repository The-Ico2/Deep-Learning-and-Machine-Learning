import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
                
def model(x, y):
    ANNreg = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    )

    LearningRate = .05
    
    lossFunc = nn.MSELoss()
    optimizer = torch.optim.SGD(ANNreg.parameters(),lr=.05)

    numepochs = 500
    losses = torch.zeros(numepochs)
    for epochi in range(numepochs):
        # forward pass
        yHat = ANNreg(x)

        #compute loss
        loss = lossFunc(yHat, y)
        losses[epochi] = loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # end training model

        ### compute model predictions
        predictions = ANNreg(x)


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