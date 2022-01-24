import torch.nn as nn
import models.layers as nl
import pdb
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nl.SharableConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nl.SharableConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Note that here we rename `fc` layer from official torchvision model as `classifier` layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_tasks, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = []
        for i in range(num_tasks):
            exec("self.bn1"+str(i)+" = norm_layer(planes)")
            exec("self.bn1.append(self.bn1"+str(i)+")")
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = []
        for i in range(num_tasks):
            exec("self.bn2"+str(i)+" = norm_layer(planes)")
            exec("self.bn2.append(self.bn2"+str(i)+")")
        self.downsample = downsample
        self.stride = stride

    def forward(self, pair):
        x = pair[0]
        k = pair[1]
        identity = x

        out = self.conv1(x)
        out = self.bn1[k](out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2[k](out)

        if self.downsample is not None :
            identity = self.downsample[0](identity)
            identity = self.downsample[1][k](identity)

        out += identity
        out = self.relu(out)

        return [out, k]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_tasks, stride=1, downsample=None,  groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(int(inplanes), width)
        self.bn1 = []
        for i in range(num_tasks):
            exec("self.bn1"+str(i)+" = norm_layer(width)")
            exec("self.bn1.append(self.bn1"+str(i)+")")
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = []
        for i in range(num_tasks):
            exec("self.bn2"+str(i)+" = norm_layer(width)")
            exec("self.bn2.append(self.bn2"+str(i)+")")
        self.conv3 = conv1x1(width, int(planes * self.expansion))
        self.bn3 = []
        for i in range(num_tasks):
            exec("self.bn3"+str(i)+" = norm_layer(int(planes * self.expansion))")
            exec("self.bn3.append(self.bn3"+str(i)+")")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, pair):
        x = pair[0]
        k = pair[1]
        identity = x

        out = self.conv1(x)
        out = self.bn1[k](out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2[k](out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3[k](out)

        if self.downsample is not None:
            identity = self.downsample[0](identity)
            identity = self.downsample[1][k](identity)

        out += identity
        out = self.relu(out)

        return [out, k]


class ResNet(nn.Module):

    def __init__(self, block, layers,  M, num_tasks, num_classes, dataset_history, dataset2num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.M = M
        self.num_classes = num_classes
        self.num_tasks = num_tasks

        self.convs1 = []
        self.bns1 = []
        self.layers1 = []
        self.layers2 = []
        self.layers3 = []
        self.layers4 = []

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        for j in range(self.M):
            self.inplanes = 64
            self.dilation = 1
            exec("self.conv1"+ str(j) +" = nl.SharableConv2d(3, self.inplanes ,kernel_size=7, stride=2, padding=3, "
                                                                                     "bias=False)")
            exec("self.convs1.append(self.conv1"+str(j)+")")
            exec("self.bn1"+str(j)+"= []")
            for i in range(self.num_tasks):
                exec("self.bn1"+str(j)+str(i)+" = self._norm_layer(self.inplanes)")
                exec("self.bn1"+str(j)+".append(self.bn1"+str(j)+str(i)+")")
            exec("self.bns1.append(self.bn1" + str(j) +")")
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for j in range(self.M):
            self.inplanes = 64
            self.dilation = 1
            exec("self.layer1"+str(j)+"= self._make_layer(block, 64, layers[0])")
            exec("self.layers1.append(self.layer1"+str(j)+")")
            exec("self.layer2"+str(j)+" = self._make_layer(block, 128, layers[1], stride=2,"
                                           "dilate=replace_stride_with_dilation[0])")
            exec("self.layers2.append(self.layer2"+str(j)+")")
            exec("self.layer3"+str(j)+" = self._make_layer(block, 256, layers[2], stride=2,"
                                           "dilate=replace_stride_with_dilation[1])")
            exec("self.layers3.append(self.layer3"+str(j)+")")
            exec("self.layer4"+str(j)+" = self._make_layer(block, 512, layers[3], stride=2,"
                                           "dilate=replace_stride_with_dilation[2])")
            exec("self.layers4.append(self.layer4"+str(j)+")")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.fc = nn.Linear(2048, num_classes) #2048


        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                #nn.init.constant_(m.weight, 0)
                nn.init.normal_(m.weight, 0, 0.001)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    for k in range(self.num_tasks):
                        nn.init.constant_(m.bn3[k].weight, 0)
                elif isinstance(m, BasicBlock):
                    for k in range(self.num_tasks):
                        nn.init.constant_(m.bn2[k].weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        downsample2 = []
        result_planes = int(planes * block.expansion)
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != result_planes:
            for i in range(self.num_tasks):
                exec("downsample2.append(norm_layer(result_planes))")
            downsample2 = nn.ModuleList(downsample2)
            downsample = nn.Sequential(
                conv1x1(self.inplanes, result_planes, stride), downsample2)

        layers = []
        layers.append(block(self.inplanes, planes, self.num_tasks, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = result_planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.num_tasks, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x, path, k):
        y = 0
        for j in range(self.M):
            if (path[0][j] == 1):
                y += self.maxpool(self.relu(self.bns1[j][k](self.convs1[j](x))))
        x = y

        y = 0
        for j in range(self.M):
            if (path[1][j] == 1):
                y += self.layers1[j]([x, k])[0]
        x = y

        y = 0
        for j in range(self.M):
            if (path[2][j] == 1):
                y += self.layers2[j]([x, k])[0]
        x = y

        y = 0
        for j in range(self.M):
            if (path[3][j] == 1):
                y += self.layers3[j]([x, k])[0]
        x = y

        y = 0
        for j in range(self.M):
            if (path[4][j] == 1):
                y += self.layers4[j]([x, k])[0]
        x = y

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(M, num_tasks, num_classes=100, dataset_history=[], dataset2num_classes={}, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],  M, num_tasks, num_classes, dataset_history=[], dataset2num_classes={}, **kwargs)

def resnet34(M, num_tasks, num_classes=100, dataset_history=[], dataset2num_classes={}, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], M, num_tasks, num_classes, dataset_history=[], dataset2num_classes={}, **kwargs)

def resnet50(M, num_tasks, num_classes=100, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], M, num_tasks, num_classes, dataset_history=[], dataset2num_classes={}, **kwargs)

def resnet101(M, num_tasks, num_classes=100, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], M, num_tasks, num_classes,  dataset_history=[], dataset2num_classes={}, **kwargs)

def resnet152(M, num_tasks, num_classes=100, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], M, num_tasks, num_classes,  **kwargs)

def resnext50_32x4d(M, num_tasks, num_classes=100, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], M, num_tasks, num_classes,  dataset_history=[], dataset2num_classes={}, groups=4, width_per_group=32, **kwargs)


def resnext101_32x8d(M, num_tasks, num_classes=100, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], M, num_tasks, num_classes, dataset_history=[], dataset2num_classes={}, groups=8, width_per_group=32, **kwargs)
