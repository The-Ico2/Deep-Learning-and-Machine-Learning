import time
import torch
import matplotlib.pyplot as plt

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

start = time.time()
print('}--------------------------------------------------------------------------------| Program Has Started |--------------------------------------------------------------------------------{')

def ts(inp):
    sin = torch.sin(inp)
    return sin

def tc(inp):
    cos = torch.cos(inp)
    return cos

def te(inp):
    exp = torch.exp(inp)
    return exp

def tl(inp1, inp2, inp3):
    lins = torch.linspace(inp1, inp2, inp3, device='cuda:0')
    return lins

def tr(inp):
    round = torch.round(inp)
    return round

def tz(inp):
    zero = torch.zeros(inp, device='cuda:0')
    return zero



# function (Note: over-writing varibale names)
def fx(x):
     fx_res = ts(x)*te(-x**2*.05)
     return fx_res

# derivative function
def deriv(x):
    deriv_res = tc(x)*te(-x**2*.05) - ts(x)*.1*x*te(-x**2*.05)
    return deriv_res




# Experiment 3: interaction between learning rates and training epochs

# setup parameters
learningrates = tl(1e-10,1e-1,10)
training_epochs = tr(tl(10,500,10))

# Initialize matrix to store results
finalres = tz((len(learningrates),len(training_epochs)))
# loop over learning rates
for Lidx,learningRate in enumerate(learningrates):
    #loop over training epochs
    for Eidx,trainEpoch in enumerate(training_epochs):
        locmin = torch.tensor(0)
        for i in range(int(trainEpoch)):
            grad = deriv(locmin)
            locmin = locmin - learningRate*grad
        # store the final guess
        finalres[Lidx,Eidx] = locmin

        print('Learning Rate- ', f'{Lidx+1:0>2.0f}', ' : ', '{:.5f}'.format(learningRate.item()), ' ]=[ ', 'Epoch- ', f'{Eidx+1:0>2.0f}' , ' : ', int(trainEpoch.item()))
end = time.time()
tte = end-start
print('}--------------------------------------------------------------------------------| Program Has Ended |--------------------------------------------------------------------------------{')
print('Time To Finish Execution: ', round(tte),'s')

# plot the result
fig,ax = plt.subplots(figsize=(7,5))

plt.imshow(finalres.cpu(),extent=[learningrates[0].cpu(),learningrates[-1].cpu(),training_epochs[0].cpu(),training_epochs[-1].cpu()], aspect='auto',origin='lower',vmin=-1.45,vmax=-1.2)
plt.xlabel('Learning Rate')
plt.ylabel('Training Epochs')
plt.title('Final Guess')
plt.colorbar()
plt.show()