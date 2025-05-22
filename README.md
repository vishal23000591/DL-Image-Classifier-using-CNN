# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
![Screenshot 2025-04-24 092951](https://github.com/user-attachments/assets/ad1adeeb-7c7d-4708-9850-b268a390f5e3)

## DESIGN STEPS
### STEP 1: Import all the required libraries (PyTorch, TorchVision, NumPy, Matplotlib, etc.).

### STEP 2: Download and preprocess the MNIST dataset using transforms.

### STEP 3: Create a CNN model with convolution, pooling, and fully connected layers.
### STEP 4: Set the loss function and optimizer. Move the model to GPU if available.
### STEP 5: Train the model using the training dataset for multiple epochs.

### STEP 6: Evaluate the model using the test dataset and visualize the results (accuracy, confusion matrix, classification report, sample prediction).

## PROGRAM

### Name: Vishal S

### Register Number: 212223110063


```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


```

```
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='../Data', train=True, download=True, transform=transform)

test_data = datasets.MNIST(root='../Data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)


```

```
for i, (X_train, y_train) in enumerate(train_data):
    break

x = X_train.view(1,1,28,28)

class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(42)
model = ConvolutionalNetwork()
model

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
```
import time
start_time = time.time()

# Variables ( Trackers)
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# for loop epochs
for i in range(epochs):

    trn_corr = 0
    tst_corr = 0


    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1

        # Apply the model
        y_pred = model(X_train)  # we not flatten X-train here
        loss = criterion(y_pred, y_train)


        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()  # Trure 1 / False 0 sum()
        trn_corr += batch_corr

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b%600 == 0:
            print(f'epoch: {i}  batch: {b} loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')

train_losses = [t.detach().numpy() for t in train_losses]
test_losses = [t.detach().numpy() for t in test_losses]

plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend();
plt.show()

plt.plot([t/600 for t in train_correct], label='training accuracy')
plt.plot([t/100 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend();
plt.show()

test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

```
```
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)  # we don't flatten the data this time
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()

correct.item()

correct.item()/len(test_data)

# print a row of values for reference
np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))
print(np.arange(10).reshape(1,10))
print()

# print the confusion matrix
print(confusion_matrix(predicted.view(-1), y_test.view(-1)))

```
```
# single image for test
plt.imshow(test_data[2019][0].reshape(28,28))
plt.show()

model.eval()
with torch.no_grad():
    new_prediction = model(test_data[2019][0].view(1,1,28,28))

new_prediction.argmax()

torch.save(model.state_dict(), 'Vishal.pt')

new_model = ConvolutionalNetwork() # Replace Model with ConvolutionalNetwork
new_model.load_state_dict(torch.load('Bharathwaj.pt'))
new_model.eval()

```

### OUTPUT

## Training Loss per Epoch
![image](https://github.com/user-attachments/assets/b8425cb4-5856-410e-aada-c9bc292ac948)

## Confusion Matrix
![image](https://github.com/user-attachments/assets/0aef0283-57df-405c-b124-0cbf391de00b)

## Classification Report
![image](https://github.com/user-attachments/assets/101b437e-2e5b-4fec-9f43-cdb4adf7fe5e)

### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/949f1918-513f-44ee-b38d-e361237ec1c3)


## RESULT
Thus the CNN model was trained and tested successfully on the MNIST dataset.
