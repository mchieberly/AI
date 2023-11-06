# import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import mlp

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Running on " + dev)

train_dataset = MNIST(".", train=True, 
                      download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, 
                          batch_size=64, shuffle=True)

# Instantiate model, optimizer, and hyperparameter(s)
in_dim, feature_dim, out_dim = 784, 256, 10
lr=1e-3
loss_fn = nn.CrossEntropyLoss()
epochs=40
classifier = mlp.BaseClassifier(in_dim, feature_dim, out_dim)
classifier = classifier.to(dev)
optimizer = optim.SGD(classifier.parameters(), lr=lr)

def train(classifier=classifier,
          optimizer=optimizer,
          epochs=epochs,
          loss_fn=loss_fn):

  classifier.train()
  loss_lt = []
  for epoch in range(epochs):
    running_loss = 0.0
    for minibatch in train_loader:
      data, target = minibatch
      target = target.to(dev)
      data = data.flatten(start_dim=1)
      data = data.to(dev)
      out = classifier(data)
      computed_loss = loss_fn(out, target)
      computed_loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      # Keep track of sum of loss of each minibatch
      running_loss += computed_loss.item()
    loss_lt.append(running_loss/len(train_loader))
    print("Epoch: {} train loss: {}".format(epoch+1, running_loss/len(train_loader)))

#   plt.plot([i for i in range(1,epochs+1)], loss_lt)
#   plt.xlabel("Epoch")
#   plt.ylabel("Training Loss")
#   plt.title(
#       "MNIST Training Loss: optimizer {}, lr {}".format("SGD", lr))
#   plt.show()

  # Save state to file as checkpoint
  print("Saving network in mnist.plt")
  torch.save(classifier.state_dict(), 'mnist.pt')

train()