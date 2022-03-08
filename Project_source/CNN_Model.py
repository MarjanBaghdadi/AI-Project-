import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # The general structure of the model is defined here
        self.conv1 = nn.Conv2d(3, 64, (3, 3), padding=(1, 1), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 256, (3, 3), padding=(1, 1), stride=(2, 2))
        self.linear1 = nn.Linear(self.linear_input_neurons(), 1024)
        self.linear2 = nn.Linear(1024, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

    def num_flat_features(self, x):
        dims = x.size()[1:]
        # print(x.size(), dims)
        feature_number = 1
        for dim in dims:
            feature_number *= dim
        return feature_number

# here we apply convolution operations before linear layer, and it returns the 4-dimensional size tensor.
    def size_after_relu(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        return x.size()

    # after obtaining the size in above method, we call it and multiply all elements of the returned size.
    def linear_input_neurons(self):
        size = self.size_after_relu(torch.rand(1, 3, 128, 128))  # image size: 128x128
        m = 1
        for i in size:
            m *= i

        return int(m)
