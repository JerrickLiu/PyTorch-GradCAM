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
test_dir = '/Users/SirJerrick/Downloads/data/dogs-vs-cats/testset'

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(30),
     transforms.RandomVerticalFlip(),
     transforms.RandomResizedCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ])

test_transform = transforms.Compose([
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

batch_size = 4

trainset = datasets.ImageFolder(root=train_dir, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
val_set = torchvision.datasets.ImageFolder(root = val_dir, transform = test_transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle = True, num_workers = 4 )

testset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
classes = ('cat', 'dog')

import matplotlib.pyplot as plt
import numpy as np

#functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
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

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(in_features = 2048, out_features= 2)

import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

start_time = time.time()

img_cnt = len(trainloader)

step = 0

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        model.train()
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        prediction = model(inputs)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()

        end_time = time.time()

        # print statistics
        running_loss += loss.item()

        stats =[]

        if i > 0:
            average_loss = running_loss / i

            if end_time > start_time:
                stats.append("Epoch: {}".format(epoch))
                stats.append("Average loss: {}".format(average_loss))
                stats.append("{}%".format(100 * i / img_cnt))
                print(stats)

        info = {'Loss': loss.item()}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)

        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        step += 1

print('Finished Training')

torch.save(model.state_dict(), '/Users/SirJerrick/Documents/Saved_models')

dataiter = iter(val_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' %classes[labels[j]] for j in range(4)))

outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    print("Testing...")
    for data in val_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    model.eval()
    for data in val_loader:
        images, labels = data
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