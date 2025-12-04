import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu, relu, gelu
from torch.nn.utils import weight_norm
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from braindecode.models.modules import Expression, AvgPool2dWithConv, Ensure4d
from braindecode.models.functions import identity, transpose_time_to_spat, squeeze_final_output

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
   # print(table)
  #  print(f"Total Trainable Params: {total_params}")
    return total_params

class MultiBKNet(nn.Sequential):
    
    
    def __init__(
        self,
        n_outputs=2,
        n_chans=21,
        n_filters_conv=40,

        pool_conv='mean',
        pool_block_conv='mean',
        pool_conv_length=50,
        pool_conv_stride=15,
        norm_conv='batch',
        nonlin_conv='elu',

        drop_prob=0.5,
        filter_length_2=20,
        filter_length_3=20,
        pool_time_length=3,
        pool_time_stride=3,

        fourth_block=False,
        fourth_block_broader=False,


        chs_info=None,

        sfreq=100,

        input_window_samples=6000,
        add_log_softmax=True,
        stride_before_pool=True,
        return_feats=False,
    ):
        
        
        super().__init__( )
        self.n_outputs=n_outputs
        self.n_classes=n_outputs
        self.n_chans=n_chans
        self.chs_info=chs_info
        self.n_times=input_window_samples
        self.input_window_samples=input_window_samples
        self.sfreq=sfreq
        self.add_log_softmax=add_log_softmax
        self.pool_conv = pool_conv
        self.norm_conv=norm_conv
        self.nonlin_conv = nonlin_conv
        self.drop_prob=drop_prob
        self.return_feats=return_feats
     #   self.final_pool_type= final_pool_type
    
        n_filters_conv_branch = int(n_filters_conv /5)
    
        self.n_filters_2= int(n_filters_conv*2)
        self.n_filters_3 = int(self.n_filters_2*2)
        
        self.filter_length_2=filter_length_2
        self.filter_length_3=filter_length_3
        
        self.fourth_block =fourth_block
        self.fourth_block_broader = fourth_block_broader
        

        if stride_before_pool ==True:
            conv_stride =pool_time_stride 
            pool_time_stride = 1
        n_classes=n_outputs
        n_channels=n_chans
        pool_time_length2=pool_time_length
        pool_time_stride2=pool_time_stride #3
        
       
        if nonlin_conv=='elu':
            self.cnn_activation = elu 
            
        if nonlin_conv=='gelu':
            self.cnn_activation = gelu
    
        self.ensuredims = Ensure4d()
        self.dimshuffle = Expression(transpose_time_to_spat)
        
        
        self.conv_time1 = nn.Conv2d(
                in_channels=1,
                out_channels=n_filters_conv_branch,
                kernel_size=(200, 1),
                stride=(1,1),
                padding='same',
                    )
        
        self.conv_time2 = nn.Conv2d(
                in_channels=1,
                out_channels=n_filters_conv_branch,
                kernel_size=(25, 1), 
                stride=(1,1),
                padding='same',
                    )
                
        self.conv_time3 = nn.Conv2d(
                in_channels=1,
                out_channels=n_filters_conv_branch,
                kernel_size=(13, 1),
                stride=(1,1),
                padding='same',
                    )
        
        
        self.conv_time4 = nn.Conv2d(
                in_channels=1,
                out_channels=n_filters_conv_branch,
                kernel_size=(7, 1),
                stride=(1,1),
                padding='same'
                    )

        self.conv_time5 = nn.Conv2d(
                in_channels=1,
                out_channels=n_filters_conv_branch,
                kernel_size=(3, 1),
                stride=(1,1),
                padding='same'
                    )


        
        self.conv_spat1 = nn.Conv2d(
                in_channels=n_filters_conv_branch,
                out_channels=n_filters_conv_branch,
                kernel_size=(1,n_channels),
                stride=(1,1),#
                #padding='same',
                    ) 
        
        self.conv_spat2 = nn.Conv2d(
                in_channels=n_filters_conv_branch,
                out_channels=n_filters_conv_branch,
                kernel_size=(1,n_channels),
                stride=(1,1),#
                #padding='same',
                    ) 

        self.conv_spat3 = nn.Conv2d(
                in_channels=n_filters_conv_branch,
                out_channels=n_filters_conv_branch,
                kernel_size=(1,n_channels),
                stride=(1,1),#
                #padding='same',
                    ) 
        self.conv_spat4 = nn.Conv2d(
                in_channels=n_filters_conv_branch,
                out_channels=n_filters_conv_branch,
                kernel_size=(1,n_channels),
                stride=(1,1),#
                #padding='same',
                    ) 
        self.conv_spat5 = nn.Conv2d(
                in_channels=n_filters_conv_branch,
                out_channels=n_filters_conv_branch,
                kernel_size=(1,n_channels),
                stride=(1,1),#
                #padding='same',
                    ) 
        

        if norm_conv == 'batch':
        
            batch_norm_alpha=0.1


            self.norm1 = nn.BatchNorm2d(
                        n_filters_conv_branch,
                        momentum=batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    )
            self.norm2 = nn.BatchNorm2d(
                        n_filters_conv_branch,
                        momentum=batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    )
            self.norm3 = nn.BatchNorm2d(
                        n_filters_conv_branch,
                        momentum=batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    )
            self.norm4 = nn.BatchNorm2d(
                        n_filters_conv_branch,
                        momentum=batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    )
            self.norm5 = nn.BatchNorm2d(
                        n_filters_conv_branch,
                        momentum=batch_norm_alpha,
                        affine=True,
                        eps=1e-5,
                    )
            
        if norm_conv == 'group':       


            num_groups= int(n_filters_conv_branch/2) if n_filters_conv_branch  % 10 == 0 else int(n_filters_conv_branch/5)
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=n_filters_conv_branch)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=n_filters_conv_branch)            
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=n_filters_conv_branch)
            self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=n_filters_conv_branch)
            self.norm5 = nn.GroupNorm(num_groups=num_groups, num_channels=n_filters_conv_branch)
            
            
        if nonlin_conv=='elu':
            self.nonlin1 =nn.ELU() 
            self.nonlin2 =nn.ELU() 
            self.nonlin3 =nn.ELU() 
            self.nonlin4 =nn.ELU() 
            self.nonlin5 =nn.ELU() 
            
        if nonlin_conv=='gelu':
            self.nonlin1 =nn.GELU() 
            self.nonlin2 =nn.GELU() 
            self.nonlin3 =nn.GELU() 
            self.nonlin4 =nn.GELU() 
            self.nonlin5 =nn.GELU() 
        
        if pool_conv == 'mean':
        
            self.pool1 = nn.AvgPool2d(kernel_size=(50,1), stride=(15,1))#pool_time_length), stride=(1, stride_avg_pool)
            self.pool2 = nn.AvgPool2d(kernel_size=(50,1), stride=(15,1))#
            self.pool3 = nn.AvgPool2d(kernel_size=(50,1), stride=(15,1))#p
            self.pool4 = nn.AvgPool2d(kernel_size=(50,1), stride=(15,1))#
            self.pool5 = nn.AvgPool2d(kernel_size=(50,1), stride=(15,1))#
            
            
            
            #n_patches=396
            
        if pool_conv == 'max':
        
            self.pool1 = nn.MaxPool2d(kernel_size=(50,1), stride=(15,1))
            self.pool2 = nn.MaxPool2d(kernel_size=(50,1), stride=(15,1))
            self.pool3 = nn.MaxPool2d(kernel_size=(50,1), stride=(15,1))
            self.pool4 = nn.MaxPool2d(kernel_size=(50,1), stride=(15,1))
            self.pool5 = nn.MaxPool2d(kernel_size=(50,1), stride=(15,1))
            
            
            
        self.drop_2 = nn.Dropout(p=self.drop_prob, inplace=False)

        self.conv_2 = nn.Conv2d(n_filters_conv, self.n_filters_2, kernel_size=(self.filter_length_2, 1),
                                stride=(conv_stride, 1),#(1,1), 
                               # dilation=(pool_time_stride, 1), 
                                bias=False)

        if norm_conv == 'batch':
            self.bnorm_2 = nn.BatchNorm2d(self.n_filters_2, eps=1e-5,momentum= batch_norm_alpha, affine=True)

        if norm_conv == 'group':       


            num_groups= int(self.n_filters_2/2) #if n_filters_conv_branch!= 15 else int(n_filters_conv_branch/3)
            self.bnorm_2 = nn.GroupNorm(num_groups=num_groups, num_channels=self.n_filters_2)
            
        self.nonlin_2 = Expression(self.cnn_activation)
        
        if  pool_block_conv=='max':
            self.pool_2 = nn.MaxPool2d(kernel_size=(pool_time_length, 1), 
                                       stride=(pool_time_stride,1), 
                                       padding=0, 
                                      # dilation=(int(pool_time_stride*pool_time_stride),1), 
                                       ceil_mode=False)
        if  pool_block_conv=='mean':
                self.pool_2 = nn.AvgPool2d(kernel_size=(pool_time_length, 1), 
                                           stride=(pool_time_stride,1), 
                                           padding=0, 
                                          # dilation=(int(pool_time_stride*pool_time_stride),1), 
                                           ceil_mode=False)


        self.pool_nonlin_2 = Expression(self.cnn_activation)
        
        ## ConvBlock3
        
        self.drop_3 = nn.Dropout(p=self.drop_prob, inplace=False)

        self.conv_3 = nn.Conv2d(self.n_filters_2, self.n_filters_3, kernel_size=(self.filter_length_3, 1),
                                stride=(conv_stride, 1),#(1,1), 
                               # dilation=(pool_time_stride, 1), 
                                bias=False)
        if norm_conv == 'batch':
            self.bnorm_3 = nn.BatchNorm2d(self.n_filters_3, eps=1e-5,momentum= batch_norm_alpha, affine=True)
        if norm_conv == 'group':       


            num_groups= int(self.n_filters_3/2) #if n_filters_conv_branch!= 15 else int(n_filters_conv_branch/3)
            self.bnorm_3 = nn.GroupNorm(num_groups=num_groups, num_channels=self.n_filters_3)

        self.nonlin_3 = Expression(self.cnn_activation)

        if  pool_block_conv=='max':
            self.pool_3 = nn.MaxPool2d(kernel_size=(pool_time_length, 1), 
                                       stride=(pool_time_stride,1), 
                                       padding=0, 
                                      # dilation=(int(pool_time_stride*pool_time_stride),1), 
                                       ceil_mode=False)
        if  pool_block_conv=='mean':
                self.pool_3 = nn.AvgPool2d(kernel_size=(pool_time_length, 1), 
                                           stride=(pool_time_stride,1), 
                                           padding=0, 
                                          # dilation=(int(pool_time_stride*pool_time_stride),1), 
                                           ceil_mode=False)
                
        self.pool_nonlin_3 = Expression(self.cnn_activation)
            
        self.n_filters_final =  self.n_filters_3
            
        if self.fourth_block:
            
            if self.fourth_block_broader:
                self.n_filters_4= int(self.n_filters_3*2)
            else:
                self.n_filters_4= int(self.n_filters_2)
                
            self.filter_length_4 = self.filter_length_3
            
            self.drop_4 = nn.Dropout(p=self.drop_prob, inplace=False)

            self.conv_4 = nn.Conv2d(self.n_filters_3, self.n_filters_4 , kernel_size=(self.filter_length_4, 1),
                                    stride=(conv_stride, 1),#(1,1), 
                                   # dilation=(pool_time_stride, 1), 
                                    bias=False)

            if norm_conv == 'batch':
                self.bnorm_4 = nn.BatchNorm2d(self.n_filters_4, eps=1e-5,momentum= batch_norm_alpha, affine=True)
            if norm_conv == 'group':       


                num_groups= int(self.n_filters_4/2) #if n_filters_conv_branch!= 15 else int(n_filters_conv_branch/3)
                self.bnorm_4 = nn.GroupNorm(num_groups=num_groups, num_channels=self.n_filters_4)
            
            self.nonlin_4 = Expression(self.cnn_activation)

            if  pool_block_conv=='max':
                self.pool_4 = nn.MaxPool2d(kernel_size=(pool_time_length, 1), 
                                           stride=(pool_time_stride,1), 
                                           padding=0, 
                                          # dilation=(int(pool_time_stride*pool_time_stride),1), 
                                           ceil_mode=False)
            if  pool_block_conv=='mean':
                self.pool_4 = nn.AvgPool2d(kernel_size=(pool_time_length, 1), 
                                           stride=(pool_time_stride,1), 
                                           padding=0, 
                                          # dilation=(int(pool_time_stride*pool_time_stride),1), 
                                           ceil_mode=False)
                
                
            self.pool_nonlin_4 = Expression(self.cnn_activation)
            
            self.n_filters_final =  self.n_filters_4
        
        self.final_conv_length =  self.get_seq_length(torch.ones(1,n_channels,input_window_samples))
        
        self.conv_classifier= nn.Conv2d(
                                    self.n_filters_final,
                                    self.n_classes,
                                    (self.final_conv_length, 1),
                                    bias=True,
                                )
        
    
        if self.add_log_softmax:
            self.softmax =  nn.LogSoftmax(dim=1)
        
        self.squeeze = Expression(squeeze_final_output)
        
        init.xavier_uniform_(self.conv_time1.weight, gain=1) 
        init.constant_(self.conv_time1.bias, 0)

        init.xavier_uniform_(self.conv_time2.weight, gain=1) 
        init.constant_(self.conv_time2.bias, 0)

        init.xavier_uniform_(self.conv_time3.weight, gain=1) 
        init.constant_(self.conv_time3.bias, 0)

        init.xavier_uniform_(self.conv_time4.weight, gain=1) 
        init.constant_(self.conv_time4.bias, 0)

        init.xavier_uniform_(self.conv_time5.weight, gain=1) 
        init.constant_(self.conv_time5.bias, 0)
        
        init.xavier_uniform_(self.conv_spat1.weight, gain=1)
        init.xavier_uniform_(self.conv_spat2.weight, gain=1)
        init.xavier_uniform_(self.conv_spat3.weight, gain=1)
        init.xavier_uniform_(self.conv_spat4.weight, gain=1)
        init.xavier_uniform_(self.conv_spat5.weight, gain=1)

#         if self.batch_norm:
        init.constant_(self.norm1.weight, 1)
        init.constant_(self.norm1.bias, 0)


        init.constant_(self.norm2.weight, 1)
        init.constant_(self.norm2.bias, 0)

        init.constant_(self.norm3.weight, 1)
        init.constant_(self.norm3.bias, 0)

        init.constant_(self.norm4.weight, 1)
        init.constant_(self.norm4.bias, 0)
        
        init.constant_(self.norm5.weight, 1)
        init.constant_(self.norm5.bias, 0)
            
        param_dict = dict(list(self.named_parameters()))
        
        if self.fourth_block:
            n_blocks = 5
        else:
            n_blocks = 4
            
        for block_nr in range(2, n_blocks):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)

            bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
            bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
            init.constant_(bnorm_weight, 1)
            init.constant_(bnorm_bias, 0)
        
        
        self.eval()
        
    def get_seq_length(self, x: Tensor) -> Tensor:
       # print(x.shape)
        x = self.ensuredims(x)
#        # Dimension: (batch_size, C, T, 1)
      # 
        x = self.dimshuffle(x)
        conv_time1 = self.conv_time1(x)   
        conv_spat1 = self.conv_spat1(conv_time1)
        conv_norm1 = self.norm1(conv_spat1) 
        conv_nonlin1 = self.nonlin1(conv_norm1)     
        conv_pool1 = self.pool1(conv_nonlin1)
        
        conv_time2 = self.conv_time2(x)   
        conv_spat2 = self.conv_spat2(conv_time2)
        conv_norm2 = self.norm2(conv_spat2) 
        conv_nonlin2 = self.nonlin2(conv_norm2)     
        conv_pool2 = self.pool2(conv_nonlin2)
        
        conv_time3 = self.conv_time3(x)   
        conv_spat3 = self.conv_spat3(conv_time3)
        conv_norm3 = self.norm3(conv_spat3) 
        conv_nonlin3 = self.nonlin3(conv_norm3)     
        conv_pool3 = self.pool3(conv_nonlin3)
        
        conv_time4 = self.conv_time4(x)   
        conv_spat4 = self.conv_spat4(conv_time4)
        conv_norm4 = self.norm4(conv_spat4) 
        conv_nonlin4 = self.nonlin4(conv_norm4)     
        conv_pool4 = self.pool4(conv_nonlin4)
        
        conv_time5 = self.conv_time5(x)   
        conv_spat5 = self.conv_spat5(conv_time5)
        conv_norm5 = self.norm5(conv_spat5) 
        conv_nonlin5 = self.nonlin5(conv_norm5)     
        conv_pool5 = self.pool5(conv_nonlin5)

    
        conv_feats = torch.cat([conv_pool1,conv_pool2,conv_pool3,conv_pool4,conv_pool5],dim=1)
        
     #   print(conv_feats.shape)
        
        out= self.drop_2(conv_feats)
        out= self.conv_2(out)
        out= self.bnorm_2(out)
        out=self.nonlin_2(out)
        out=self.pool_2(out)
        out=self.pool_nonlin_2(out)

       # print(out.shape)
        out= self.drop_3(out)
        out= self.conv_3(out)
        out= self.bnorm_3(out)
        out=self.nonlin_3(out)
        out=self.pool_3(out)
        out=self.pool_nonlin_3(out)
        #print(out.shape)
        if self.fourth_block:
            out= self.drop_4(out)
            out= self.conv_4(out)
            out= self.bnorm_4(out)
            out=self.nonlin_4(out)
            out=self.pool_4(out)
            out=self.pool_nonlin_4(out)
            
        return out.shape[2]
        
       # print('conv_feats',conv_feats.shape)
        
       # return conv_feats.shape[3]

    def forward(self, x: Tensor) -> Tensor:
        x = self.ensuredims(x)
#        # Dimension: (batch_size, C, T, 1)
      # 
        x = self.dimshuffle(x)
       # print(x.shape)
        conv_time1 = self.conv_time1(x)   
        conv_spat1 = self.conv_spat1(conv_time1)
        conv_norm1 = self.norm1(conv_spat1) 
        conv_nonlin1 = self.nonlin1(conv_norm1)     
        conv_pool1 = self.pool1(conv_nonlin1)
        
        conv_time2 = self.conv_time2(x)   
        conv_spat2 = self.conv_spat2(conv_time2)
        conv_norm2 = self.norm2(conv_spat2) 
        conv_nonlin2 = self.nonlin2(conv_norm2)     
        conv_pool2 = self.pool2(conv_nonlin2)
        
        conv_time3 = self.conv_time3(x)   
        conv_spat3 = self.conv_spat3(conv_time3)
        conv_norm3 = self.norm3(conv_spat3) 
        conv_nonlin3 = self.nonlin3(conv_norm3)     
        conv_pool3 = self.pool3(conv_nonlin3)
        
        conv_time4 = self.conv_time4(x)   
        conv_spat4 = self.conv_spat4(conv_time4)
        conv_norm4 = self.norm4(conv_spat4) 
        conv_nonlin4 = self.nonlin4(conv_norm4)     
        conv_pool4 = self.pool4(conv_nonlin4)
        
        conv_time5 = self.conv_time5(x)   
        conv_spat5 = self.conv_spat5(conv_time5)
        conv_norm5 = self.norm5(conv_spat5) 
        conv_nonlin5 = self.nonlin5(conv_norm5)     
        conv_pool5 = self.pool5(conv_nonlin5)

    
        conv_feats = torch.cat([conv_pool1,conv_pool2,conv_pool3,conv_pool4,conv_pool5],dim=1)
        
        
        out= self.drop_2(conv_feats)
        out= self.conv_2(out)
        out= self.bnorm_2(out)
        out=self.nonlin_2(out)
        out=self.pool_2(out)
        out=self.pool_nonlin_2(out)

       
        out= self.drop_3(out)
        out= self.conv_3(out)
        out= self.bnorm_3(out)
        out=self.nonlin_3(out)
        out=self.pool_3(out)
        out=self.pool_nonlin_3(out)
        
        if self.fourth_block:
            out= self.drop_4(out)
            out= self.conv_4(out)
            out= self.bnorm_4(out)
            out=self.nonlin_4(out)
            out=self.pool_4(out)
            out=self.pool_nonlin_4(out)
            
            
        if self.return_feats:
            out = self.squeeze(out)
            return out
        
        else:
                   
            out= self.conv_classifier(out)

            if self.add_log_softmax:
                out=self.softmax(out)

            out = self.squeeze(out)




            return out 