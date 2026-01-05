import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class RegressorNet(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, name):
        super(RegressorNet, self).__init__()
        self.n_x = n_input # x dims
        self.n_y = n_output # y dims
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)

        self.checkpoint_file = os.path.join('./', name+'_regressor.model')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x1 = T.tanh(self.fc3(x))

        return x1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



#net=RegressorNet(n_inputs=10,n_outputs=3,n_hidden=32,name='test')
#x=T.zeros(10)
#y=net.forward(x)
#print(x)
#print(y)
