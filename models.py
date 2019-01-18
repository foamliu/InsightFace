import torchvision
from torch import nn
from torchsummary import summary
from torchvision import transforms

from config import *

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ArcFaceModel(nn.Module):
    def __init__(self):
        super(ArcFaceModel, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=True)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.bn1 = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048 * 4 * 4, 512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, images):
        x = self.resnet(images)  # [N, 512, 4, 4]
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 512]
        x = self.fc(x)
        x = self.bn2(x)
        return x


if __name__ == "__main__":
    model = ArcFaceModel().to(device)
    summary(model, (3, 112, 112))
