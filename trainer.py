import numpy as np
from numpy import vstack
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pcgrad import PCGrad
import pickle

def train(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
):
    total_rloss = 0
    total_kloss = 0
    total_lrloss = 0
    total_loss = 0
    total_sample = 0
    for batch_id, sample in enumerate(train_loader):
        # load data
        V = Variable(sample['V1']).float().cuda()
        V_repeat = V.unsqueeze(1).repeat(1, max_sequence_length, 1)
        X = Variable(sample['X']).float().permute(0,2,1).cuda()
        y = Variable(sample['outcomes']).cuda()

        XVA = torch.cat((X, V_repeat),axis=2)
        mask = Variable(sample['mask'].float()).cuda()
        mask_repeat = mask.unsqueeze(2).repeat(1,1,input_size)
        seq_len = Variable(sample['seq_len']).long().cuda()
        
        # reconstruction loss
        out = model(XVA, seq_len)
        rloss = criterion_MSE(mask_repeat * out, mask_repeat * XVA)

        # linear regression loss
        outputs = model.linearReg(XVA, seq_len)
        lrloss = criterion_MSE(outputs, y.float())
        lrloss = torch.mul(criterion_MSE(outputs, y.float()),lmbda_lrloss)

        # K-means loss
        kloss, assignment = model.get_k_mean_loss(XVA, seq_len)
        kloss = torch.mul(kloss,lmbda_kloss)
        
        # Total loss
        loss = rloss + kloss + lrloss

        optimizer.zero_grad()
        optimizer.pc_backward([rloss, kloss, lrloss]) 
        optimizer.step()     

        with torch.no_grad():
            total_rloss += rloss.cpu().item() * sample['X'].size(0)
            total_kloss += kloss.cpu().item() * sample['X'].size(0)
            total_lrloss += lrloss.cpu().item() * sample['X'].size(0)
            total_loss += loss.cpu().item() * sample['X'].size(0) 
            total_sample += sample['X'].size(0)
    
    rloss = total_rloss / total_sample
    kloss = total_kloss / total_sample
    lrloss = total_lrloss / total_sample
    loss = total_loss / total_sample

    return loss, rloss, kloss, lrloss

def evaluate(
    model, 
    eval_loader, 
    criterion
):
    total_rloss = 0
    total_kloss = 0
    total_lrloss = 0
    total_loss = 0
    total_sample = 0
    predictions, actuals = torch.empty((0,2)), torch.empty((0,2))
    r2_list = list()
    with torch.no_grad():
        for batch_id, sample in enumerate(eval_loader):
            V = Variable(sample['V1']).float().cuda()
            V_repeat = V.unsqueeze(1).repeat(1, max_sequence_length, 1)
            X = Variable(sample['X']).float().permute(0,2,1).cuda()
            y = Variable(sample['outcomes']).cuda()

            XVA = torch.cat((X, V_repeat),axis=2)
            mask = Variable(sample['mask'].float()).cuda()#.to(device)
            mask_repeat = mask.unsqueeze(2).repeat(1,1,input_size)
            seq_len = Variable(sample['seq_len']).long().cuda()#.to(device)        

            # reconstruction loss
            out = model(XVA, seq_len)
            rloss = criterion_MSE(mask_repeat * out, mask_repeat * XVA)      

            # linear regression loss
            outputs = model.linearReg(XVA, seq_len)
            predictions = torch.cat((predictions,outputs.cpu()), axis=0)
            actuals = torch.cat((actuals,y.cpu()),axis=0)
            lrloss = torch.mul(criterion_MSE(outputs, y.float()),lmbda_lrloss)

            # K-means loss
            kloss, assignment = model.get_k_mean_loss(XVA,seq_len)
            kloss = torch.mul(kloss,lmbda_kloss)

            # Total loss
            loss = rloss + kloss + lrloss
            
            total_rloss += rloss.cpu().item() * sample['X'].size(0)
            total_kloss += kloss.cpu().item() * sample['X'].size(0)
            total_lrloss += lrloss.cpu().item() * sample['X'].size(0)
            total_loss += loss.cpu().item() * sample['X'].size(0) 
            total_sample += sample['X'].size(0)
    rloss = total_rloss / total_sample
    kloss = total_kloss / total_sample
    lrloss = total_lrloss / total_sample
    loss = total_loss / total_sample
    predictions_, actuals_ = vstack(predictions), vstack(actuals)
    r2 = r2_score(actuals_, predictions_)

    # by label
    for i in list(range(2)):
        pred_, actual_ = vstack(predictions[:,i]), vstack(actuals[:,i])
        r2_= r2_score(actual_, pred_)
        r2_list.append(r2_)
    
    return loss, rloss, kloss, lrloss, r2, r2_list