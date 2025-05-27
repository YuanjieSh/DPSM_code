import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['densenet']


from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        # print(f"Before torch.cat: x version={x._version}, out version={out._version}")
        out = torch.cat((x, out), 1)
        # print(f"After torch.cat: out version={out._version}")

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, dropRate=0, num_classes=10, growthRate=12, compressionRate=2, use_fc_single = False):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 6 if block == Bottleneck else (depth - 4) // 3

        self.growthRate = growthRate
        self.dropRate = dropRate
        self.num_classes = num_classes 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Default to GPU 0

        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to handle different input sizes
        
        # self.fc = None  # Fully connected layer will be defined dynamically in forward
        self.fc = nn.Linear(self.inplanes, num_classes)
        
        self.use_fc_single = use_fc_single
        if self.use_fc_single:
            self.fc_single = nn.Linear(self.inplanes, 1)  # Add only if needed



        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(int(blocks)):
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward_single(self, x):

        # print(f"Input to conv1_single: shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")
        x = self.conv1(x)
        # print(f"After conv1_single: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        x = self.dense1(x)
        # print(f"After dense1_single: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        x = self.trans1(x)
        # print(f"After trans1_single: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        x = self.dense2(x)
        # print(f"After dense2_single: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        x = self.trans2(x)
        # print(f"Before trans2_single: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")
        
        x = self.dense3(x)
        # print(f"After dense3_single: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        
        assert x.dim() == 4, f"Expected 4D input to BatchNorm, got {x.dim()}D"
        assert x.size(0) > 0, "Batch size is zero, which is invalid for BatchNorm"

        # print(f"Before BatchNorm_single: version={x._version}, requires_grad={x.requires_grad}")
        x = self.bn(x)
        # print(f"After BatchNorm_single: version={x._version}, requires_grad={x.requires_grad}")
        x = self.relu(x)  # Avoid in-place modifications

        x = self.avgpool(x)

        # Debug before view
        assert x.dim() == 4, f"Expected 4D tensor before view, got {x.dim()}D"
        x = x.view(x.size(0), -1)

        single_value = self.fc_single(x)

        return single_value.squeeze(dim=1)


    def forward(self, x):

        # print(f"Input to conv1_forward: shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")
        x = self.conv1(x)
        # print(f"After conv1_forward: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        x = self.dense1(x)
        # print(f"After dense1_forward: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        x = self.trans1(x)
        # print(f"After trans1_forward: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        x = self.dense2(x)
        # print(f"After dense2_forward: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")

        x = self.trans2(x)
        # print(f"Before trans2_forward: x shape={x.shape}, requires_grad={x.requires_grad}, version={x._version}")
        
        x = self.dense3(x)

        # print(f"Before BatchNorm_forward: version={x._version}, requires_grad={x.requires_grad}")
        x = self.bn(x)
        # print(f"After BatchNorm_forward: version={x._version}, requires_grad={x.requires_grad}")
        x = self.relu(x)

        # Adaptive average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def densenet(**kwargs):
    """
    Constructs a DenseNet model.
    """
    return DenseNet(**kwargs)
