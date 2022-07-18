import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import json

class CNNConsistency(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config_keys = []
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.LazyLinear(1)

    def forward(self, sentence):
        sentence = torch.reshape(sentence, (1,16,16))
        x = self.pool(F.relu(self.conv1(sentence)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        return x[0]

    def save(self, output_path: str):
        # with open(os.path.join(output_path, 'cnn_config.json'), 'w') as fOut:
        #     json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path: str):
        # with open(os.path.join(input_path, 'cnn_config.json'), 'r') as fIn:
        #     config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model = CNNConsistency()
        model.load_state_dict(weights)
        return model