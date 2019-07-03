import time
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

from Tensorboard import Logger

log_dir = '/Users/SirJerrick/Documents/logs'

now = datetime.now()

nowstr = now.isoformat()

run_path = os.path.join(log_dir, nowstr)

os.mkdir(run_path)

logger = Logger(run_path)

train_dir = '/Users/SirJerrick/Downloads/data/dogs-vs-cats/trainset/train'
val_dir = '/Users/SirJerrick/Downloads/data/dogs-vs-cats/trainset/val'

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(30),
     transforms.RandomVerticalFlip(),
     transforms.RandomResizedCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ])

test_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

batch_size = 4

trainset = datasets.ImageFolder(root=train_dir, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
val_set = torchvision.datasets.ImageFolder(root=val_dir, transform=test_transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

print('Number of training samples: ', len(trainset))
print('Number of validation samples: ', len(val_set))

classes = ('cat', 'dog')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

model = models.resnet50(pretrained=True)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.fc = nn.Linear(in_features=2048, out_features=len(classes))

model.to(device)

import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
'''
Using the cross-entropy function along with the softmax function, we are able to calculate the loss of the model.
It calculates loss by taking the log-softmax probabilities of its prediction. The softmax function normalizes the
probabilities which results in the probabilities sum to 1. Cross entropy loss, or log loss then measures those
outputted probabilities with respect to the true label. Cross entropy loss increases as the predicted probability
diverges from the actual label, with the true label being 1.
This is me just trying to explain it to myself
'''

optimizer = optim.Adam(model.parameters(), lr=0.0005)

"""
An optimizer is an algorithm that updates the weights and parameters of the nerual network
in response to the output of the loss function explained above. The loss function tells the optimizer
if its moving in the right direction and the optimizer updates the network's weights in response. If
the loss is decreasing, then the optimizer is doing a good job. In order to actually calculate the gradient, we do 
backpropagation. Backpropagation, combined with optimization, is basically how a machine learns. Backpropagation 
involves taking the loss found in the loss function and sending it backwards through the network so the optimizer 
can alter the weights and parameters. Backpropagataion works by calculating a gradient (think slope) of each parameter 
in the network. We use differential calculus to find the gradient. In our case, we want to find the change in loss with 
respect to the change in a weight and we want to move both variables in the direction of less loss. Once we
know the direction, the optimizer updates the weights in accordance to the direction of less loss. The most common 
algorithm is gradient descent. Gradient descents seeks to find the minimum of the loss function. 
In gradient descent, we:
1.) Calculate how much a small change in each individual weight would do to the loss function
2.) Adjust each weight based on its gradient. A gradient is a partial derivative that measure
the change in loss with respect to the change in the weights. The gradients of the weights tell
us what to do with the weights - add .5 or subtract .1 for example - which will decrease loss and make the model more 
accurate.
3.) Repeat steps 1 and 2 until loss is as low as possible. 
Keep in mind that this is for the weights of only 1 layer. In a model like Resnet50 that has 50 layers and millions of 
weights,
gradient descent and the optimizer performs the process and calculates the gradient for all the weights in each layer, 
an application of the chain rule. 
In our case, we are using the Adam optimizer because it can adapt the learning rate for each parameter individually
which will increase accuracy and decrease loss. 
"""


start_time = time.time()

img_cnt = len(trainloader)

best_val_loss = float('inf')

for epoch in range(2):  # loop over the dataset multiple times

    step = 0
    running_loss = 0.0
    model.train()

    for inputs, labels in trainloader:
        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients, so no previous gradients are stored and will not build up.
        optimizer.zero_grad()

        # forward + backward + optimize
        prediction = model(inputs)  #Forward. In other words, the machine makes a prediction. In machine learning, this is called forward propagating.
        loss = loss_function(prediction, labels) #Calculate the loss of the prediction
        loss.backward() #Backpropagation takes the output of the loss function and uses it to compute the gradients of the weights.
        optimizer.step() #Updates the weights based on the gradient.

        end_time = time.time()

        # print statistics
        running_loss += loss.item()

        stats = []

        if step > 0:
            average_loss = running_loss / step

            if end_time > start_time:
                stats.append("Epoch: {}".format(epoch))
                stats.append("Average loss: {}".format(average_loss))
                stats.append("{}%".format(100 * step / img_cnt))
                print(stats)

            if average_loss < best_val_loss:
                best_val_loss = average_loss
                torch.save(model.state_dict(), '/Users/SirJerrick/Documents/Saved_models/ResNet_cats_dogs.pth')

        info = {'Loss': loss.item()}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)

        step += 1

print('Finished Training')

correct = 0
total = 0

with torch.no_grad():
    model.eval()
    print("Testing...")
    for data in val_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))

class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))


with torch.no_grad():
    model.eval()
    for data in val_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
