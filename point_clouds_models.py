#This file is based on https://github.com/IdanAchituve/DefRec_and_PCM
# args.model == 'dgcnn', args.dropout
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

K = 20

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, args, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    # Run on cpu or gpu
    device = 'cuda'

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                nn.BatchNorm1d(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """
    def __init__(self, args, in_ch, out=3):
        super(transform_net, self).__init__()
        self.K = out
        self.args = args

        activation = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = False if args.model == 'dgcnn' else True

        self.conv2d1 = conv_2d(in_ch, 64, kernel=1, activation=activation, bias=bias)
        self.conv2d2 = conv_2d(64, 128, kernel=1, activation=activation, bias=bias)
        self.conv2d3 = conv_2d(128, 1024, kernel=1, activation=activation, bias=bias)
        self.fc1 = fc_layer(1024, 512, activation=activation, bias=bias, bn=True)
        self.fc2 = fc_layer(512, 256, activation=activation, bn=True)
        self.fc3 = nn.Linear(256, out * out)

    def forward(self, x):
        device = 'cuda'

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        if self.args.model == "dgcnn":
            x = x.max(dim=-1, keepdim=False)[0]
            x = torch.unsqueeze(x, dim=3)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x

class PointNet(nn.Module):
    def __init__(self, args, num_class=10):
        super(PointNet, self).__init__()
        self.args = args

        self.trans_net1 = transform_net(args, 3, 3)
        self.trans_net2 = transform_net(args, 64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        num_f_prev = 64 + 64 + 64 + 128

        self.C = classifier(args, num_class)

    def forward(self, x):
        num_points = x.size(2)
        x = torch.unsqueeze(x, dim=3)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        transform = self.trans_net2(x2)
        x = x2.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x4)
        x5, _ = torch.max(x, dim=2, keepdim=False)
        x = x5.squeeze(dim=2)  # batchsize*1024

        logits = self.C(x)

        return logits


class DGCNN(nn.Module):
    def __init__(self, args, num_class=10):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = K

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64*2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64*2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128*2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.bn5 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)

        self.C = classifier(args, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, self.args, k=self.k)
        transformd_x0 = self.input_transform_net(x0)
        x = torch.matmul(transformd_x0, x)

        x = get_graph_feature(x, self.args, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, self.args, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = F.leaky_relu(self.bn5(self.conv5(x_cat)), negative_slope=0.2)

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        x5 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x = x5

        logits = self.C(x)

        return logits


class classifier(nn.Module):
    def __init__(self, args, num_class=10):
        super(classifier, self).__init__()

        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(1024, 512, bias=bias, activation=activate, bn=True)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = fc_layer(512, 256, bias=True, activation=activate, bn=True)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        return logits


    
# PointNet

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x
        #return F.log_softmax(x, dim=1), trans, trans_feat
    
    
    
class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
        
        
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    