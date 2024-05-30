import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):    
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        """
        num_maps_in: number of input channels for the conv layer, also number of input and output channels for the BatchNorm2d layer
        num_maps_out: number of output channels for the conv layer/ number of filters of the conv layer
        k: kernel size of the conv layer
        bias: if True, the batch normalization layer will have learnable affine parameters (scale and shift).
        """
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE
        self.append(torch.nn.BatchNorm2d(num_maps_in, affine=bias))
        self.append(torch.nn.ReLU())
        self.append(torch.nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.margin = 1
        self.unit_1 = _BNReluConv(input_channels, emb_size, k=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.unit_2 = _BNReluConv(emb_size, emb_size, k=3)
        self.unit_3 = _BNReluConv(emb_size, emb_size, k=3)
        self.global_avg = nn.AvgPool2d(kernel_size=2)
        

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # img is tensor of dimenstions BATCH_SIZE, C, H, W -> (B, 1, 28, 28) for MNIST
        # YOUR CODE HERE
        x = self.unit_1(img)
        x = self.maxpool(x)
        x = self.unit_2(x)
        x = self.maxpool(x)
        x = self.unit_3(x)
        x = self.global_avg(x)
        shape=x.shape #should be (B, EMB_SIZE, 1, 1) after global averaging
        x = x.reshape((shape[0], shape[1])) #reshaped to (B, EMB_SIZE)
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        loss = ...
        return loss