import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dims, code_dims):
        super(Net, self).__init__()
        self.input_dims = input_dims
        self.code_dims = code_dims
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(-1, self.input_dims)
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x