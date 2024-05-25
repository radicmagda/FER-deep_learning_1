import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):    
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        """
        num_maps_in: number of input channels for the conv layer
        num_maps_out : number of output channels for the conv layer/ number of filters of the conv layer
        k: kernel size of the conv layer
        bias : if True, the batch normalization layer will have learnable affine parameters (scale and shift).
        """
        super(_BNReluConv, self).__init__()
        self.append(torch.nn.BatchNorm2d(num_maps_in, affine=bias))
        self.append(torch.nn.ReLU())
        self.append(torch.nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = ...
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        loss = ...
        return loss