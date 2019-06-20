from torch import nn
from torch.nn import functional as F


class InnerConvNet(nn.Module):

    def __init__(self, n_classes=10):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.PReLU(), nn.AvgPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5), nn.PReLU(), nn.AvgPool2d(2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256), nn.PReLU(),
            nn.Linear(256, n_classes))

    def forward(self, inputs):
        outputs = self.convnet(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)
        scores = F.log_softmax(outputs, dim=-1)
        return outputs, scores


class DoubleNet(nn.Module):

    def __init__(self, inner_net):
        super().__init__()
        self.inner_net = inner_net

    def forward(self, inputs_1, inputs_2):
        logits_1, scores_1 = self.inner_net(inputs_1)
        logits_2, scores_2 = self.inner_net(inputs_2)
        return (logits_1, logits_2), (scores_1, scores_2)
