import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # Input (1, 224, 224)
        # Output (10, 221, 221)
        self.conv1 = nn.Conv2d(1, 10, 4)
        
        # Output (10, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Output (20, 108, 108)
        self.conv2 = nn.Conv2d(10, 20, 3)
        
        # Output (20, 54, 54)
        # pool
        
        # Output (32, 53, 53)
        self.conv3 = nn.Conv2d(20, 32, 2)
        
        # Output (32, 26, 26)
        # pool

        self.linear1 = nn.Linear(32*26*26, 32*26)
        
        self.drop = nn.Dropout(p=0.1)
        
        self.linear2 = nn.Linear(32*26, 32*10)
        
        # 68*2 - key points shape
        self.linear3 = nn.Linear(32*10, 68*2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop(x)
   
        x = x.view(x.size(0), -1) # flatted

        x = F.relu(self.linear1(x))
        x = self.drop(x)

        x = F.relu(self.linear2(x))
        x = self.drop(x)
        
        return self.linear3(x)