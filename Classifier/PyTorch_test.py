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

test_dir = '/path/to/test/data'

test_transform = transforms.Compose([
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

classes = ('dog', 'cat')

batch_size = 400

testset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained = True)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
model.fc = nn.Linear(in_features=2048, out_features=len(classes))
model.load_state_dict(torch.load('/path/to/checkpoint.pth'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    print("Testing...")
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))        
