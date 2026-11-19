"""SimpleBaseline (Xiao et al., ECCV 2018): ImageNet ResNet backbone + 3 deconvolution layers -> 17 heatmaps."""
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

BACKBONES = {"resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
             "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048)}


class SimpleBaseline(nn.Module):
    def __init__(self, backbone="resnet50", num_joints=17, pretrained=True, deconv_ch=256):
        super().__init__()
        fn, w, cin = BACKBONES[backbone]
        r = fn(weights=w if pretrained else None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        layers = []
        for _ in range(3):
            layers += [nn.ConvTranspose2d(cin, deconv_ch, 4, 2, 1, bias=False), nn.BatchNorm2d(deconv_ch),
                       nn.ReLU(True)]
            cin = deconv_ch
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv2d(deconv_ch, num_joints, 1)
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(self.head.weight, std=0.001); nn.init.zeros_(self.head.bias)

    def forward(self, x):                      # (B, 3, 256, 192) -> (B, 17, 64, 48)
        return self.head(self.deconv(self.backbone(x)))
