import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import slp

test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate model, optimizer, and hyperparameter(s)
in_dim, out_dim = 784, 10
loss_fn = nn.CrossEntropyLoss()
classifier = slp.BaseClassifier(in_dim, out_dim)
classifier.load_state_dict(torch.load('mnist.pt'))

def test(classifier=classifier, loss_fn = loss_fn):
    classifier.eval()
    accuracy = 0.0
    computed_loss = 0.0
    confusion_matrix = np.zeros((10, 10))

    with torch.no_grad():

        for data, target in test_loader:
            data = data.flatten(start_dim=1)
            out = classifier(data)
            _, preds = out.max(dim=1)
         
            # Get loss and accuracy
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds==target)
            for i in range(len(preds)): 
                confusion_matrix[preds[i]][target[i]] += 1               
          
        print("Test loss: {}, test accuracy: {}".format(
        computed_loss.item()/(len(test_loader)*64), accuracy*100.0/(len(test_loader)*64)))
        df = pd.DataFrame(confusion_matrix, columns = ['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
        print(df)
      
test()