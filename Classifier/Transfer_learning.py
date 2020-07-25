import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.models.resnet import ResNet
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
from tqdm import tqdm
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder

epochs = 2
batch_size = 10
learning_rate = 0.003
data_path = ./datset
test_data_path = ./test_data
validation_split = .2
shuffle_dataset = True
random_seed = 42

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

dataset = ImageFolder(root = data_path, transform = data_transforms)

dataset_size = len(dataset)

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=data_transforms)
test_data_loader  = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

if __name__ == '__main__':

    print("Number of train samples: ", len(dataset))
    print("Number of test samples: ", len(test_data))
    print("Detected Classes are: ", dataset.class_to_idx) # classes are detected by folder structure

model = models.resnet50(pretrained = True)
#model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

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

    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batch_size)

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
