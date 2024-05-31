import torch
import torch.nn as nn

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        """
        obična vektorizacija slike u R^(CxHxW)
        """
        # YOUR CODE HERE
        # iz (B, C, H, W) u (B, C*H*W) tj "peglanje" slike
        feats = torch.flatten(img, start_dim=1) # start_dim=1 jer zelimo ocuvati dimenziju batcha
        return feats