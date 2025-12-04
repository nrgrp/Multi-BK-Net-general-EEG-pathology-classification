import mne
import numpy as np
#import pyedflib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from IPython.display import Image
from IPython.core.display import HTML


    
class InceptionBlock(nn.Module): # Creating the class for our inception block
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels = 32, kernel_size = 2, stride = 2, padding = 0) # Defines 1st Convolution
        self.conv2 = nn.Conv1d(in_channels, out_channels = 32, kernel_size = 4, stride = 2, padding = 1) # Defines 2nd Convolution
        self.conv3 = nn.Conv1d(in_channels, out_channels = 32, kernel_size = 8, stride = 2, padding = 3) # Defines 3rd Convolution
        self.relu=nn.ReLU() # Defines the ReLU Activation Function

    #Here, xn is the output of the nth layer.
    def forward(self,x): #Defining the forward function
        x1 = self.relu(self.conv1(x)) #performing 1st conv and outputting x1
        x2 = self.relu(self.conv2(x)) #performing 2nd conv and outputting x2
        x3 = self.relu(self.conv3(x)) #performing 3rd conv and outputting x3
        x = torch.cat((x1,x2,x3), dim = 1) #taking all outputs of convolutions and concatenating them on 1 Dimension
        
        return x
    
    
class ChronoNet(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.inception_block1=InceptionBlock(channel) # 1st Inception Block
        self.inception_block2=InceptionBlock(96) # 2nd Inception Block
        self.inception_block3=InceptionBlock(96) # 3rd Inception Block
        self.gru1 = nn.GRU(input_size = 96, hidden_size = 32, batch_first = True) # 1st GRU layer
        self.gru2 = nn.GRU(input_size = 32, hidden_size = 32, batch_first = True) # 2nd GRU layer
        self.gru3 = nn.GRU(input_size = 64, hidden_size = 32, batch_first = True) # 3rd GRU layer
        self.gru4 = nn.GRU(input_size = 96, hidden_size = 32, batch_first = True) # 4th GRU layer
        self.relu = nn.ReLU() # ReLU Activation Function
        self.gru_linear=nn.Linear(in_features = 750, out_features = 1) # Linear Layer for the 4th GRU
        self.flatten = nn.Flatten() # Flattening Layer
        self.fc1 = nn.Linear(32,2) # Fully Connected Layer / Output Layer.

    def forward(self,x): # Defining the feed forward function
        x=self.inception_block1(x) # Fed to Inception Block 1
        #print(x.size)
        x=self.inception_block2(x) # Fed to Inception Block 2
        x=self.inception_block3(x) # Fed to Inception Block 3
        x=x.permute(0,2,1) # Permuted for GRU layers
        gru_out1,_=self.gru1(x) # Fed into GRU layer 1
        gru_out2,_=self.gru2(gru_out1) # Fed into GRU layer 2
        gru_out=torch.cat((gru_out1, gru_out2), dim = 2) # Concatenated, defining the skip connection
        gru_out3,_=self.gru3(gru_out)  # Fed into GRU layer 3
        gru_out = torch.cat((gru_out1, gru_out2, gru_out3), dim = 2) #C Concatenated, defining the next 2 skip connections
      #  print(gru_out.size())
        gru_out = gru_out.permute(0,2,1) # Permuted for the linear layer
        linear_out=self.relu(self.gru_linear(gru_out)) # Fed into the linear layer to reduce dimensionality
        linear_out = linear_out.permute(0,2,1) # Permuted for the 4th GRU layer
        gru_out4,_=self.gru4(linear_out) # Fed into the 4th GRU Layer
        x=self.flatten(gru_out4) # Data is Flattened for Fully Connected Layer
        x=self.fc1(x) # Fed into the Fully Connected Layer

        return x  #x # Output
    
class ChronoNetfeat(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.inception_block1=InceptionBlock(channel) # 1st Inception Block
        self.inception_block2=InceptionBlock(96) # 2nd Inception Block
        self.inception_block3=InceptionBlock(96) # 3rd Inception Block
        self.gru1 = nn.GRU(input_size = 96, hidden_size = 32, batch_first = True) # 1st GRU layer
        self.gru2 = nn.GRU(input_size = 32, hidden_size = 32, batch_first = True) # 2nd GRU layer
        self.gru3 = nn.GRU(input_size = 64, hidden_size = 32, batch_first = True) # 3rd GRU layer
        self.gru4 = nn.GRU(input_size = 96, hidden_size = 32, batch_first = True) # 4th GRU layer
        self.relu = nn.ReLU() # ReLU Activation Function
        self.gru_linear=nn.Linear(in_features = 750, out_features = 1) # Linear Layer for the 4th GRU
        self.flatten = nn.Flatten() # Flattening Layer
        self.fc1 = nn.Linear(32,2) # Fully Connected Layer / Output Layer.

    def forward(self,x): # Defining the feed forward function
        x=self.inception_block1(x) # Fed to Inception Block 1
        #print(x.size)
        x=self.inception_block2(x) # Fed to Inception Block 2
        x=self.inception_block3(x) # Fed to Inception Block 3
        x=x.permute(0,2,1) # Permuted for GRU layers
        gru_out1,_=self.gru1(x) # Fed into GRU layer 1
        gru_out2,_=self.gru2(gru_out1) # Fed into GRU layer 2
        gru_out=torch.cat((gru_out1, gru_out2), dim = 2) # Concatenated, defining the skip connection
        gru_out3,_=self.gru3(gru_out)  # Fed into GRU layer 3
        gru_out = torch.cat((gru_out1, gru_out2, gru_out3), dim = 2) #C Concatenated, defining the next 2 skip connections
      #  print(gru_out.size())
        gru_out = gru_out.permute(0,2,1) # Permuted for the linear layer
        linear_out=self.relu(self.gru_linear(gru_out)) # Fed into the linear layer to reduce dimensionality
        linear_out = linear_out.permute(0,2,1) # Permuted for the 4th GRU layer
        gru_out4,_=self.gru4(linear_out) # Fed into the 4th GRU Layer
#         x=self.flatten(gru_out4) # Data is Flattened for Fully Connected Layer
#         x=self.fc1(x) # Fed into the Fully Connected Layer

        return gru_out  #x # Output