import torch.nn as nn


class SingleLayerPerceptron(nn.Module):
    """
    A simple model as test class for the train flow.
    """

    def __init__(self, input_length):
        super(SingleLayerPerceptron, self).__init__()
        self.l1 = nn.Linear(input_length, 1)

    def forward(self, input):
        # input = input['input']

        output = self.l1(input)
        return output
