import torch.nn.functional as F
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


class ArcFaceEncoder(nn.Module):
    def __init__(self):
        super(ArcFaceEncoder, self).__init__()

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


class ArcMarginModel(nn.Module):
    def __init__(self):
        super(ArcMarginModel, self).__init__()

        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, embedding):
        x = self.fc(embedding)
        id_out = F.softmax(x, dim=1)
        return id_out


if __name__ == "__main__":
    model = ArcFaceEncoder().to(device)
    summary(model, (3, 112, 112))
