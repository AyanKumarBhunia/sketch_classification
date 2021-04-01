import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        backbone = backbone_.vgg16(pretrained=True) #vgg16, vgg19_bn

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        # if hp.pool_method is not None:
        #     self.pool_method = eval('nn.' + str(hp.pool_method) + '(1)')
        #     # AdaptiveMaxPool2d, AdaptiveAvgPool2d, AvgPool2d
        # else:
        #     self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default

        backbone.classifier._modules['6'] = nn.Linear(4096, 250)
        self.classifier = backbone.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x




class Resnet_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True) #resnet50, resnet18, resnet34
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)

        if hp.pool_method is not None:
            self.pool_method = eval('nn.' + str(hp.pool_method) + '(1)')
        else:
            self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        # AdaptiveMaxPool2d, AdaptiveAvgPool2d, AvgPool2d
        self.classifier = nn.Linear(2048, 250)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)


    def forward(self, x):

        x = x.sub(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        x = self.features(x)
        x = self.pool_method(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

class Resnet_Network18(nn.Module):
    def __init__(self, hp):
        super(Resnet_Network18, self).__init__()
        backbone = backbone_.resnet18(pretrained=True) #resnet50, resnet18, resnet34
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)

        if hp.pool_method is not None:
            self.pool_method = eval('nn.' + str(hp.pool_method) + '(1)')
        else:
            self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        # AdaptiveMaxPool2d, AdaptiveAvgPool2d, AvgPool2d
        self.classifier = nn.Linear(512, 250)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)


    def forward(self, x):

        x.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        x = self.features(x)
        x = self.pool_method(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x


class InceptionV3_Network(nn.Module):
    def __init__(self, hp):
        super(InceptionV3_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        self.backbone = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['AuxLogits', 'fc']:
                self.backbone.add_module(name, module)

        if hp.pool_method is not None:
            self.pool_method = eval('nn.' + str(hp.pool_method) + '(1)')
        else:
            self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        # AdaptiveMaxPool2d, AdaptiveAvgPool2d, AvgPool2d

    def forward(self, x, every_layer=True):
        feature_list = {}
        batch_size = x.shape[0]
        for name, module in self.backbone._modules.items():
            x = module(x)
            if every_layer:
                feature_list[name] = self.pool_method(x).view(batch_size, -1)

        if not feature_list:
            feature_list['pre_logits'] = self.pool_method(x).view(batch_size, -1)
        return feature_list


