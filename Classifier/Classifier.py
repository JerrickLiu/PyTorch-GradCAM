import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision
import csv
import glob
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
import os
from torchvision.models.resnet import ResNet, BasicBlock
import time
import inspect
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

data_set_path = '/home/jerrick/Desktop/dogs-vs-cats/train/'

batch_size = 100
validation_split = .2
test_split = .1
shuffle_dataset = True
random_seed= 42

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset = ImageFolder(root = data_set_path, transform = data_transforms['train'])

dataset_size = len(dataset)


indices = list(range(dataset_size))
split1 = int(np.floor(validation_split * dataset_size))
split2 = int(np.floor((validation_split + test_split) * dataset_size))


# shuffle_dataset = False

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[split2:], indices[:split1], indices[split1:split2]

# print (f'val first {val_indices[0]}')
# print (f'val last {val_indices[-1]}')

# print (f' test first{test_indices[0]}')
# print (f' test last{test_indices[-1]}')


# print (f' train FIRST{train_indices[0]}')
# print (f' train last{train_indices[-1]}')
# exit()

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          sampler = test_sampler)

def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    return x


model = models.resnet18(pretrained = True)
#model.cuda()

epochs = 2

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())

start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

losses = []
batches = len(train_loader)
val_batches = len(val_loader)


def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")


for epoch in range(epochs):
    total_loss = 0

    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    model.train()

    for i, data in progress:
        X, y = data[0].to(device), data[1].to(device)


        model.zero_grad()
        outputs = model(X)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()


        current_loss = loss.item()
        total_loss += current_loss


        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []


    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = model(X) # this get's the prediction from the network

            val_losses += loss_function(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1]


            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )

    print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(total_loss/batches)

print(f"Training time: {time.time()-start_ts}s")

torch.save(model.state_dict(), '/home/jerrick/Documents/Saved_Models/ResNet_model.pt')

model = models.resnet18(pretrained = True)

saved_model = '/home/jerrick/Documents/Saved_Models/ResNet_model.pt'

model.load_state_dict(torch.load(saved_model))
model.eval()

data_set_path = '/home/jerrick/Desktop/dogs-vs-cats/train/'

transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
dataset = ImageFolder(root = data_set_path, transform = transforms)

testloader = torch.utils.data.DataLoader(dataset)


correct = 0
total = 0



with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))



#https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets/50544887#50544887
