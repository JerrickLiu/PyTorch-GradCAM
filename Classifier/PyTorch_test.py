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

test_dir = '/Users/SirJerrick/Downloads/data/dogs-vs-cats/testset'

test_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

classes = ('cat', 'dog')

batch_size = 20

testset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)