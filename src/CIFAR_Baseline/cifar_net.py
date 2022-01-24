
from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class cifar_net(nn.Module):

        def __init__(self, args):
            super(cifar_net, self).__init__()
            self.args = args

            """Initialize all parameters"""
            self.conv1 = []
            self.conv2 = []
            self.conv3 = []
            self.conv4 = []
            self.conv5 = []
            self.conv6 = []
            self.conv7 = []
            self.conv8 = []
            self.conv9 = []

            a1 = 64
            a2 = 64
            a3 = 128
            a4 = 256
            a5 = 512

            self.a5 =a5
            # conv1
            for i in range(self.args['M']):
                exec("self.m1" + str(i) + " = nn.Sequential(nn.Conv2d(3, "+str(a1)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a1)+"),nn.ReLU())")
                exec("self.conv1.append(self.m1" + str(i) + ")")

            # conv2
            for i in range(self.args['M']):
                exec("self.m2" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a1)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"),nn.ReLU(), nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a1)+"))")
                exec("self.conv2.append(self.m2" + str(i) + ")")

            # conv3
            for i in range(self.args['M']):
                exec("self.m3" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"),nn.ReLU(), nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"))")
                exec("self.conv3.append(self.m3" + str(i) + ")")

            # conv4
            for i in range(self.args['M']):
                exec("self.m4" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU(), nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"))")
                exec("self.conv4.append(self.m4" + str(i) + ")")

            # conv5
            for i in range(self.args['M']):
                exec("self.m5" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU(), nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"))")
                exec("self.conv5.append(self.m5" + str(i) + ")")
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            # conv6
            for i in range(self.args['M']):
                exec("self.m6" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU(), nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"))")
                exec("self.conv6.append(self.m6" + str(i) + ")")

            # conv7
            for i in range(self.args['M']):
                exec("self.m7" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU(), nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"))")
                exec("self.conv7.append(self.m7" + str(i) + ")")
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # conv8
            for i in range(self.args['M']):
                exec("self.m8" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
                exec("self.conv8.append(self.m8" + str(i) + ")")

            # conv9
            for i in range(self.args['M']):
                exec("self.m9" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
                exec("self.conv9.append(self.m9" + str(i) + ")")
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.final_layer = nn.Linear(a5, self.args['num_classes'])

            if self.args['use_cuda']:
                self.cuda()

        def forward(self, x, path):

            y = 0
            for j in range(self.args['M']):
                if(path[0][j]==1):
                    y += self.conv1[j](x)
            x = y
            x = F.relu(x)

            y = 0
            for j in range(self.args['M']):
                if(path[1][j]==1):
                    y += self.conv2[j](x)
            x = y + x
            x = F.relu(x)

            y = 0
            for j in range(self.args['M']):
                if(path[2][j]==1):
                    y += self.conv3[j](x)
            x = y + x
            x = F.relu(x)

            y = 0
            for j in range(self.args['M']):
                if(path[3][j]==1):
                    y += self.conv4[j](x)
            x = y
            x = F.relu(x)

            y = 0
            for j in range(self.args['M']):
                if(path[4][j]==1):
                    y += self.conv5[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool1(x)

            y = 0
            for j in range(self.args['M']):
                if(path[5][j]==1):
                    y += self.conv6[j](x)
            x = y
            x = F.relu(x)

            y = 0
            for j in range(self.args['M']):
                if(path[6][j]==1):
                    y += self.conv7[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool2(x)

            y = 0
            for j in range(self.args['M']):
                if(path[7][j]==1):
                    y += self.conv8[j](x)
            x = y
            x = F.relu(x)

            y = 0
            for j in range(self.args['M']):
                if(path[8][j]==1):
                    y += self.conv9[j](x)
            x = y + x
            x = F.relu(x)

            x = F.avg_pool2d(x, (8, 8), stride=(1,1))
            x = x.view(-1, self.a5)
            x = self.final_layer(x)

            return x
