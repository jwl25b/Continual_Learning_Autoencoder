import torch
import numpy as np
from net import Net

class Autoencoder():
    def __init__(self, input_dims, code_dims, lr=0.001):
        super(Autoencoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net(input_dims, code_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        
        #self.criterion = torch.nn.BCELoss()
        self.criterion = torch.nn.MSELoss()
        self.unreduced_criterion = torch.nn.MSELoss(reduction = 'none')
        
        self.input_dims = input_dims
        self.size = 0
        self.mean = 0
        self.var = 0
        self.std = 0
        
    def optimize_params(self, x, label):
        x = x.to(self.device).reshape(-1, self.input_dims)
        label = label.to(self.device).reshape(-1, self.input_dims)
        y = self._forward(x)
        
        self._update_params(y, label)
        self.update_statistics(y, label)

    def _forward(self, x):
        return self.net(x)

    def _backward(self, y, label):
        self.loss = self.criterion(y, label)
        self.loss.backward()
        #self.accelerator.backward(self.loss)

    def _update_params(self, y, label):
        self.optimizer.zero_grad()
        self._backward(y, label)
        self.optimizer.step()
        #self.scheduler.step()  # scheduler step in each iteration
        
    def update_statistics(self, y, label):
        #https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
        #calculate batchwise loss
        new_element_loss = self.unreduced_criterion(y, label)
        new_loss = torch.mean(new_element_loss, axis=1)
            
        #calculate mean, var and batch size
        new_mean = torch.mean(new_loss).item()
        new_var = torch.var(new_loss).item()
        new_size = y.shape[0]
        #print(new_mean, new_var, new_size)
            
        #updating variance
        part1 = ((self.size - 1)*self.var + (new_size - 1)*new_var)/(self.size + new_size - 1)
        part2 = ((self.size*new_size)*np.square((self.mean - new_mean)))/((self.size + new_size)*(self.size+new_size - 1))
        self.var = part1 + part2
            
        #updating std
        self.std = np.sqrt(self.var)
        #print(self.std)
            
        #updating mean
        with torch.no_grad():
            new_sum = torch.sum(new_loss).item()
        self.mean = (self.size*self.mean + new_sum)/(self.size + new_size)
            
        #updating size
        self.size += new_size
            
    def get_reduced_loss(self, y, label):
        with torch.no_grad():
            y = y.to(self.device).reshape(-1, self.input_dims)
            label = label.to(self.device).reshape(-1, self.input_dims)
            return self.criterion(y, label).item()
    
    def get_unreduced_loss(self, y, label):
        with torch.no_grad():
            y = y.to(self.device).reshape(-1, self.input_dims)
            label = label.to(self.device).reshape(-1, self.input_dims)
            return self.unreduced_criterion(y, label)
    
    def get_prediction(self, x):
        with torch.no_grad():
            return self._forward(x.to(self.device))