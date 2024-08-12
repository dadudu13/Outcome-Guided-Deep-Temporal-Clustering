import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

class OGDTCModel(nn.Module):
    def __init__(self, input_size_V1, input_size_X, hidden_size, num_clusters, num_layers=2): 
        super(DTCRModel,self).__init__()
        self.input_size_V1=input_size_V1
        self.input_size_X=input_size_X
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.input_size=input_size_V1+input_size_X #+input_size_A #+input_size_V2
        self.num_clusters=num_clusters

        self.bn=nn.BatchNorm1d(self.input_size, eps=1)
        self.gru_encoder=nn.GRU(self.input_size, self.hidden_size, batch_first=True, 
                                bidirectional=False, num_layers=self.num_layers)
        self.gru_decoder=nn.GRU(self.hidden_size, self.input_size, batch_first=True, 
                                bidirectional=False)
    
        self.linear=nn.Linear(self.hidden_size, 2)
        
    def forward(self, XVA, seq_len):
        batch_size=XVA.size(0)
        XVA=self.bn(XVA.reshape(-1, self.input_size)).reshape(-1, max_sequence_length, self.input_size)
        # XVA=F.dropout(XVA, p, training=self.training)
        pack=nn.utils.rnn.pack_padded_sequence(XVA, seq_len.cpu(),batch_first=True,enforce_sorted=False)
        z=self.encoder(pack, batch_size)
       
        # Decoder
        out=self.decoder(z, batch_size)
             
        return out
        
    def encoder(self, x, batch_size):
        # x: batch_size * max_seqeunece_length * input_size
        # Encoder
        h0=self.init_hidden(batch_size, self.hidden_size, self.num_layers) 
        out, h_n =self.gru_encoder(x, h0) #out=batch_size * max_seq_length * hidden_size*self.num_layers
        unpacked, unpacked_len =torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)      
        indices=(unpacked_len.cuda()-1).view(-1,1,1).expand(unpacked.size(0), 1, unpacked.size(2)) 
        z=unpacked.gather(1, indices).squeeze() #batch_size * hidden_size*self.num_layers
        
        return z
        
        
    def decoder(self, z, batch_size):
        # z: batch_size * hidden_size
        z=z.unsqueeze(1).repeat(1, max_sequence_length,1) #batch_size * max_sequence_length * hidden_size*self.num_layers
        h0=self.init_hidden(batch_size, self.input_size, 1)
        out, _ = self.gru_decoder(z,h0) #batch_size * max_sequence_length * input_size
        
        #Decode the hidden state
        out=out.reshape(-1, self.input_size).reshape(-1, max_sequence_length, self.input_size)
        
        return out
    
    
    def get_k_mean_loss(self, XVA, seq_len):
        """
        EM
        """
        batch_size=XVA.size(0)
        XVA=self.bn(XVA.reshape(-1, self.input_size)).reshape(-1, max_sequence_length, self.input_size)
        pack=nn.utils.rnn.pack_padded_sequence(XVA, seq_len.cpu(),batch_first=True,enforce_sorted=False)   
        z=self.encoder(pack, batch_size)
        W=torch.transpose(z,1,0) #W is hidden_dim * num_samples
        
        # Assignment update
        U, sigma, VT=torch.linalg.svd(W)
        topk_evecs=VT[:self.num_clusters,:] #
        assignment=torch.transpose(topk_evecs,1,0) #assignment: matrix of num_samples * num_clusters
        
        # K-means clustering loss
        WTW=torch.matmul(z,W)
        FTWTWF=torch.matmul(torch.matmul(torch.transpose(assignment,1,0),WTW),assignment) #Eq(6) https://papers.nips.cc/paper/2019/file/1359aa933b48b754a2f54adb688bfa77-Paper.pdf
        loss_k_means=torch.trace(WTW)-torch.trace(FTWTWF)
        
        return loss_k_means, assignment

    def linearReg(self, XVA, seq_len):
        batch_size = XVA.size(0)
        XVA = self.bn(XVA.reshape(-1, self.input_size)).reshape(-1, max_sequence_length, self.input_size)
        pack = nn.utils.rnn.pack_padded_sequence(XVA, seq_len.cpu(),batch_first=True,enforce_sorted=False)
        z=self.encoder(pack, batch_size)
        
        return self.linear(z)
        
        
    def init_hidden(self, batch_size, hidden_size, num_layers):
        hidden=torch.randn(num_layers, batch_size, hidden_size).cuda()
        return hidden       