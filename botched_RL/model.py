# model.py
import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def save_model(self, path="model.pth"):
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_model(cls, path="model.pth", input_size=None, output_size=None):
        model = cls(input_size, output_size)
        model.load_state_dict(torch.load(path))
        model.eval()  # Set to evaluation mode
        return model
