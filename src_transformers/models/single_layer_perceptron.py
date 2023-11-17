import torch.nn as nn


class SingleLayerPerceptron(nn.Module):
    """
    A simple model as test class for the train flow.
    """

    def __init__(self, input_dim: int, output_dim: int, layers: int = 1):
        super(SingleLayerPerceptron, self).__init__()
        self.l1 = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        linear = self.l1(input)
        output = nn.functional.relu(linear)
        return output
