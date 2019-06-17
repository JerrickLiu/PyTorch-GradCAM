import torchvision.models as models

model = models.resnet152(pretrained = True)

print(model)

other_model = models.vgg19(pretrained = True)

print(other_model)