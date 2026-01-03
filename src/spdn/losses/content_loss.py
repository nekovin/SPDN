import torch.nn as nn


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, denoised, noisy):
        return self.l1(denoised, noisy)
