# ported from https://medium.com/mlearning-ai/learning-xor-with-pytorch-c1c11d67ba8e

import torch
import torch.nn as nn
from torch.autograd import Variable
import time

# create data
Xs = torch.Tensor([[0., 0.],
                   [1., 0.],
                   [0., 1.],
                   [1., 1.]])

y = torch.Tensor([0., 1., 1., 0.]).reshape(Xs.shape[0], 1)


class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.linear = nn.Linear(2, 2)
        self.Sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 1)

    def forward(self, input):
        x = self.linear(input)
        sig = self.Sigmoid(x)
        yh = self.linear2(sig)
        return yh


xor_network = XOR()
epochs = 10000
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(xor_network.parameters(), lr=0.03)

start = time.perf_counter()

for epoch in range(epochs):

    # input training example and return the prediction
    yhat = xor_network.forward(Xs)

    # calculate MSE loss
    loss = mseloss(yhat, y)

    # backpropogate through the loss gradiants
    loss.backward()

    # update model weights
    optimizer.step()

    # remove current gradients for next iteration
    optimizer.zero_grad()

print((time.perf_counter()-start)*1000, " miliseconds")

print(xor_network(torch.tensor([0., 0.])))
print(xor_network(torch.tensor([1., 0.])))
print(xor_network(torch.tensor([0., 1.])))
print(xor_network(torch.tensor([1., 1.])))