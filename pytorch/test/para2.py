import numpy as np
import matplotlib.pyplot as plt

from IPython import display
display.set_matplotlib_formats('svg')

# function (Note: over-writing varibale names)
def fx(x):
     fx_res = np.sin(x)*np.exp(-x**2*.05)
     return fx_res

# derivative function
def deriv(x):
    deriv_res = np.cos(x)*np.exp(-x**2*.05) - np.sin(x)*.1*x*np.exp(-x**2*.05)
    return deriv_res




# Experiment 3: interaction between learning rates and training epochs

# setup parameters
learningrates = np.linspace(1e-10,1e-1,50)
training_epochs = np.round(np.linspace(10,500,40))

# Initialize matrix to store results
finalres = np.zeros((len(learningrates),len(training_epochs)))


# loop over learning rates
for Lidx,learningRate in enumerate(learningrates):
    # loop over training epochs
    for Eidx,trainEpoch in enumerate(training_epochs):

        locmin = 0
        for i in range(int(trainEpoch)):
            grad = deriv(locmin)
            locmin = locmin - learningRate*grad
        # store the final guess
        finalres[Lidx,Eidx] = locmin
        print(finalres)

# plot the result
fig,ax = plt.subplots(figsize=(7,5))

plt.imshow(finalres,extent=[learningrates[0],learningrates[-1],training_epochs[0],training_epochs[-1]],
           aspect='auto',origin='lower',vmin=-1.45,vmax=-1.2)
plt.xlabel('Learning Rate')
plt.xlabel('Training Epochs')
plt.title('Final Guess')
plt.colorbar()
plt.show()

# Antoher visualization
plt.plot(learningrates,finalres)
plt.xlabel('Learning Rates')
plt.ylabel('Final Function Estimate')
plt.title('Each Line is a training epochs N')
plt.show()
