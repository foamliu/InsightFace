import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import Parameter
from torchsummary import summary
from torchvision import transforms

from config import device, num_classes
from utils import parse_args

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


class ArcFaceModel18(nn.Module):
    def __init__(self, args):
        super(ArcFaceModel18, self).__init__()

        resnet = torchvision.models.resnet18(pretrained=args.pretrained)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.bn1 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 4 * 4, args.emb_size)
        self.bn2 = nn.BatchNorm1d(args.emb_size)

    def forward(self, images):
        x = self.resnet(images)  # [N, 512, 4, 4]
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 512]
        x = self.fc(x)
        x = self.bn2(x)
        return x


class ArcFaceModel34(nn.Module):
    def __init__(self, args):
        super(ArcFaceModel34, self).__init__()

        resnet = torchvision.models.resnet34(pretrained=args.pretrained)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.bn1 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 4 * 4, args.emb_size)
        self.bn2 = nn.BatchNorm1d(args.emb_size)

    def forward(self, images):
        x = self.resnet(images)  # [N, 512, 4, 4]
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 512]
        x = self.fc(x)
        x = self.bn2(x)
        return x


class ArcFaceModel50(nn.Module):
    def __init__(self, args):
        super(ArcFaceModel50, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=args.pretrained)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.bn1 = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048 * 4 * 4, args.emb_size)
        self.bn2 = nn.BatchNorm1d(args.emb_size)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 4, 4]
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 512]
        x = self.fc(x)
        x = self.bn2(x)
        return x


class ArcFaceModel101(nn.Module):
    def __init__(self, args):
        super(ArcFaceModel101, self).__init__()

        resnet = torchvision.models.resnet101(pretrained=args.pretrained)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.bn1 = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048 * 4 * 4, args.emb_size)
        self.bn2 = nn.BatchNorm1d(args.emb_size)

    def forward(self, images):
        x = self.resnet(images)  # [N, 512, 4, 4]
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 512]
        x = self.fc(x)
        x = self.bn2(x)
        return x


class ArcMarginModel(nn.Module):
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s
        self.softmax = args.softmax

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        if self.softmax:
            output = F.softmax(output, dim=1)
        return output


if __name__ == "__main__":
    args = parse_args()
    model = ArcFaceModel101(args).to(device)
    summary(model, (3, 112, 112))
