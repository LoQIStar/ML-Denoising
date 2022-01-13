import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import chirp

# tensor of chirp functions moving at random velocities
def random_data(nrows,ncols):
    x = torch.arange(nrows)/nrows
    t = torch.arange(ncols)/ncols
    X = torch.zeros((ncols,nrows))
    for c in torch.randn(5):
        X+=torch.tensor([chirp((ti-x/c),5,1,ncols/10) for ti in t])
    return X[None,:,:]


# add random noise with random variance to a tensor
def add_randn_noise(data):
    return data.clone().detach() + torch.rand(1)*torch.randn(data.shape)


# PyTorch Dataset for random data
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, nrows, ncols, N, transforms):
        self.nrows = nrows
        self.ncols = ncols
        self.N = N
        self.transforms = transforms

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        labels = random_data(self.nrows,self.ncols)

        data = labels.clone().detach()

        for transform in self.transforms:
            data = transform(data)

        return data, labels


# one epoch of a training loop
def train_loop(dataloader, model, criterion, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    running_loss = 0.0
    for batch, (data, labels) in enumerate(dataloader):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # compute prediction and loss
        pred = model(data)
        loss = criterion(pred, labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss/size


# one epoch of a test loop
def test_loop(dataloader, model, criterion, device='cpu'):
    size = len(dataloader.dataset)
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(data)
            running_loss += criterion(pred, labels).item()

    return running_loss/size


# a generic convolitional neural network consisting of conv2d+batchnorm2d
# cnn_channels is the number of channels in each layer, including the input
# kernel sizes is the numer of elements in each dim of the 2D filter kernel
#   in each layer (should be one element less than cnn_channels)
class CNN(nn.Module):
    def __init__(self, cnn_channels=[1,1], kernel_sizes=[3]):
        super().__init__()

        # number of cnn layers
        n_feature_layers = len(cnn_channels)-1

        # create cnn
        feature_layers = []
        for i in range(n_feature_layers-1):
            feature_layers.append(nn.Conv2d(cnn_channels[i], cnn_channels[i+1],
                                    kernel_sizes[i], padding=kernel_sizes[i]//2,bias=False))
            feature_layers.append(nn.BatchNorm2d(cnn_channels[i+1]))
            #feature_layers.append(nn.ReLU(inplace=True))

        # last layer gets no batchnorm
        feature_layers.append(nn.Conv2d(cnn_channels[-2], cnn_channels[-1],
                                kernel_sizes[-1], padding=kernel_sizes[-1]//2,bias=False))

        # create cnn sequential net
        self.seq_cnn = nn.Sequential(*feature_layers)

    def forward(self,x):
        return self.seq_cnn(x)




class ConvBatchRelu(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, double_conv=False):
        super().__init__()

        feature_layers = []
        feature_layers.append(nn.Conv2d(in_channels, out_channels,
            kernel_size, padding=kernel_size//2,bias=False))
        if double_conv:
            feature_layers.append(nn.Conv2d(out_channels, out_channels,
                kernel_size, padding=kernel_size//2,bias=False))
        feature_layers.append(nn.BatchNorm2d(out_channels))
        #feature_layers.append(nn.ReLU(inplace=True))

        self.seq_cnn = nn.Sequential(*feature_layers)

    def forward(self,x):
        return self.seq_cnn(x)


class DownConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, double_conv=False):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBatchRelu(in_channels, out_channels, kernel_size, double_conv)
        )

    def forward(self,x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, bilinear=True, double_conv=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)

        self.conv = ConvBatchRelu(in_channels, out_channels, kernel_size, double_conv)

    def forward(self, x, x2=None):

        x = self.up(x)

        if not x2 == None:
            diffY = x2.size()[2] - x.size()[2]
            diffX = x2.size()[3] - x.size()[3]

            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            x = torch.cat([x, x2], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, cnn_channels=[1,1], kernel_size=[3], bilinear=True, double_conv=False):
        super().__init__()

        n_feature_layers = len(cnn_channels)-1

        down_layers = []
        for i in range(n_feature_layers):
            down_layers.append(DownConv(cnn_channels[i], cnn_channels[i+1], kernel_size[i], double_conv))

        self.down_net = nn.Sequential(*down_layers)

        self.middle_conv = ConvBatchRelu(cnn_channels[-1],cnn_channels[-2], kernel_size[i], double_conv)

        up_layers = []
        for i in range(1,n_feature_layers):
            up_layers.append(UpConv(2*cnn_channels[-1-i],cnn_channels[-2-i],kernel_size[-1-i],bilinear,double_conv))

        up_layers.append(UpConv(2*cnn_channels[0],cnn_channels[0],kernel_size[0],bilinear,double_conv))

        self.up_net = nn.Sequential(*up_layers)

    def forward(self, x):
        down_out = [x]
        #print('input:',down_out[-1].shape)
        for i,layer in enumerate(self.down_net):
            down_out.append(layer(down_out[i]))
            #print('down %i' % i ,down_out[-1].shape)

        up_out = self.middle_conv(down_out[-1])

        #print('up %i' % 0 ,up_out.shape)
        for i,layer in enumerate(self.up_net):
            up_out = layer(up_out, down_out[-2-i])
            #print('up %i' % i ,up_out.shape)

        return up_out
