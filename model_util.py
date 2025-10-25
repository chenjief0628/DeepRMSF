import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import time
import random
from tqdm import tqdm
import torch.nn.functional as F
import os
import math
import matplotlib.pyplot as plt


def generate_mask(ana_box_list, mask=0, rmsf=True, tensor=True):
    if tensor:
        if rmsf:
            mask_list = torch.where(ana_box_list != mask, 1, 0)
        else:
            mask_list = torch.where(ana_box_list == mask, 0, 1)
    else:
        if rmsf:
            mask_list = np.where(ana_box_list != mask, 1, 0)
        else:
            mask_list = np.where(ana_box_list == mask, 0, 1)
    return mask_list


def get_loader(data, only_rmsf=False, batch=16, only_sse=False, cuda=True, contour=False, shuff=True):  
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if not only_rmsf:
        intensity = data['intensity'].to(device)
        sse = data['sse'].to(device)
        resid = data['resid'].to(device)
        resname = data['resname'].to(device)
        if not only_sse:
            rmsf = data['rmsf'].to(device)
            if contour:
                con = data['contour'].to(device)
                dataset = Data.TensorDataset(intensity, sse, resid, resname, rmsf, con)  # sse,
            else:
                dataset = Data.TensorDataset(intensity, sse, resid, resname, rmsf)  # sse, 

        else:
            dataset = Data.TensorDataset(intensity, sse, resid, resname)    # sse, 
    else:
        intensity = data['intensity'].to(device)
        rmsf = data['rmsf'].to(device)
        if contour:
            con = data['contour'].to(device)
            dataset = Data.TensorDataset(intensity, rmsf, con)
        else:
            dataset = Data.TensorDataset(intensity, rmsf)

    dataloader = Data.DataLoader(dataset, batch_size=batch, shuffle=shuff,num_workers=0)
    return dataloader


def shuffle_pdb2(ori_list, seed=1):
    np.random.seed(seed)

    selected_indices = set(random.sample(range(len(ori_list)), 60))
    list = [elem for idx, elem in enumerate(ori_list) if idx not in selected_indices]
    print(len(list))

    index_list = np.random.permutation(len(list))
    cut1 = int(len(list) / 5 * 1)
    cut2 = int(len(list) / 5 * 2)
    cut3 = int(len(list) / 5 * 3)
    cut4 = int(len(list) / 5 * 4)
    index1 = index_list[:cut1]
    index2 = index_list[cut1:cut2]
    index3 = index_list[cut2:cut3]
    index4 = index_list[cut3:cut4]
    index5 = index_list[cut4:]
    list1 = np.take(np.array(list), index1)
    list2 = np.take(np.array(list), index2)
    list3 = np.take(np.array(list), index3)
    list4 = np.take(np.array(list), index4)
    list5 = np.take(np.array(list), index5)

    sp_list = []
    te_list1, te_list2, te_list3, te_list4, te_list5 = list1, list2, list3, list4, list5
    tr_list1, tr_list2, tr_list3, tr_list4, tr_list5 = np.concatenate((list2, list3, list4)), np.concatenate((list3, list4, list5)), np.concatenate((list4, list5, list1)), np.concatenate((list5, list1, list2)), np.concatenate((list1, list2, list3))
    val_list1, val_list2, val_list3, val_list4, val_list5 = list5, list1, list2, list3, list4
    sp_list.append({'train': tr_list1, 'val': val_list1, 'test': te_list1})
    sp_list.append({'train': tr_list2, 'val': val_list2, 'test': te_list2})
    sp_list.append({'train': tr_list3, 'val': val_list3, 'test': te_list3})
    sp_list.append({'train': tr_list4, 'val': val_list4, 'test': te_list4})
    sp_list.append({'train': tr_list5, 'val': val_list5, 'test': te_list5})
    
    return sp_list, [ori_list[i] for i in selected_indices]


def rna_concat_data(file_pre, datalist, dsize=(40,40,40,40,10), train=True, only_rmsf=False, only_sse=False, pre=False):
    # c_size = (int((40-dsize[0])/2), int((40-dsize[1])/2), int((40-dsize[2])/2)), int((40-dsize[3])/2), int((40-dsize[4])/2)
    c_size = (
    int((40 - dsize[0]) / 2),
    int((40 - dsize[1]) / 2),
    int((40 - dsize[2]) / 2),
    int((40 - dsize[3]) / 2),
    int((40 - dsize[4]) / 2)
    )                                                
    if train:
        if only_rmsf:
            concat_intensity = torch.full([0, 1, dsize[0], dsize[0], dsize[0]], 0, dtype=torch.float32)
            concat_rmsf = torch.full([0, 1, dsize[2], dsize[2], dsize[2]], 0, dtype=torch.float32)
            for item in datalist:
                data_file = f"{file_pre}/{item}"
                cur_data = torch.load(data_file)
                if c_size[0] == 0:
                    concat_intensity = torch.concat([concat_intensity, cur_data['intensity']], dim=0)
                else:
                    concat_intensity = torch.concat([concat_intensity, cur_data['intensity'][:,:,c_size[0]:-c_size[0],c_size[0]:-c_size[0],c_size[0]:-c_size[0]]], dim=0)
                if c_size[2] == 0:
                    concat_rmsf = torch.concat([concat_rmsf, cur_data['rmsf']], dim=0)
                else:
                    concat_rmsf = torch.concat([concat_rmsf, cur_data['rmsf'][:,:,c_size[2]:-c_size[2],c_size[2]:-c_size[2],c_size[2]:-c_size[2]]], dim=0)
                
            concat_data = {'intensity': concat_intensity, 'rmsf': concat_rmsf}
            return concat_data
        elif only_sse:
            concat_intensity = torch.full([0, 1, dsize[0], dsize[0], dsize[0]], 0, dtype=torch.float32)
            concat_sse = torch.full([0, dsize[1], dsize[1], dsize[1]], 2, dtype=torch.int)
            for item in datalist:
                data_file = f"{file_pre}/{item}"
                cur_data = torch.load(data_file)
                if c_size[0] == 0:
                    concat_intensity = torch.concat([concat_intensity, cur_data['intensity']], dim=0)
                else:
                    concat_intensity = torch.concat([concat_intensity, cur_data['intensity'][:,:,c_size[0]:-c_size[0],c_size[0]:-c_size[0],c_size[0]:-c_size[0]]], dim=0)
                if c_size[1] == 0:
                    concat_sse = torch.concat([concat_sse, cur_data['sse']], dim=0)
                else:
                    concat_sse = torch.concat([concat_sse, cur_data['sse'][:,c_size[1]:-c_size[1],c_size[1]:-c_size[1],c_size[1]:-c_size[1]]], dim=0)
            
            concat_data = {'intensity': concat_intensity, 'sse': concat_sse}
            return concat_data
        else:

            concat_intensity = torch.full([0, 1, dsize[0], dsize[0], dsize[0]], 0, dtype=torch.float32)
            concat_sse = torch.full([0, dsize[1], dsize[1], dsize[1]], 2, dtype=torch.int)
            concat_res = torch.full([0, dsize[2], dsize[2], dsize[2]], 414, dtype=torch.int)
            concat_resname = torch.full([0, dsize[3], dsize[3], dsize[3]], 4, dtype=torch.int)
            concat_rmsf = torch.full([0, 1, dsize[4], dsize[4], dsize[4]], 0, dtype=torch.float32)
            for item in datalist:
                data_file = f"{file_pre}/{item}"
                cur_data = torch.load(data_file)
                if c_size[0] == 0:
                    concat_intensity = torch.concat([concat_intensity, cur_data['intensity']], dim=0)
                else:
                    concat_intensity = torch.concat([concat_intensity, cur_data['intensity'][:,:,c_size[0]:-c_size[0],c_size[0]:-c_size[0],c_size[0]:-c_size[0]]], dim=0)
                if c_size[1] == 0:
                    concat_sse = torch.concat([concat_sse, cur_data['sse']], dim=0)
                else:
                    concat_sse = torch.concat([concat_sse, cur_data['sse'][:,c_size[1]:-c_size[1],c_size[1]:-c_size[1],c_size[1]:-c_size[1]]], dim=0)
                if c_size[2] == 0:
                    concat_res = torch.concat([concat_res, cur_data['resid']], dim=0)
                else:
                    concat_res = torch.concat([concat_res, cur_data['resid'][:,c_size[2]:-c_size[2],c_size[2]:-c_size[2],c_size[2]:-c_size[2]]], dim=0)
                if c_size[3] == 0:
                    concat_resname = torch.concat([concat_resname, cur_data['resname']], dim=0)
                else:
                    concat_resname = torch.concat([concat_resname, cur_data['resname'][:,c_size[3]:-c_size[3],c_size[3]:-c_size[3],c_size[3]:-c_size[3]]], dim=0)
                if c_size[4] == 0:
                    concat_rmsf = torch.concat([concat_rmsf, cur_data['rmsf']], dim=0)
                else:
                    concat_rmsf = torch.concat([concat_rmsf, cur_data['rmsf'][:,:,c_size[4]:-c_size[4],c_size[4]:-c_size[4],c_size[4]:-c_size[4]]], dim=0)
                
            else:
                print(f"no data augment")
            concat_data = {'intensity': concat_intensity, 'sse': concat_sse, 'resid': concat_res, 'resname': concat_resname, 'rmsf': concat_rmsf}  # 'sse': concat_sse, 'resname': concat_resname, 
            return concat_data

    else:
        data_dic = dict()
        if only_rmsf:
            for item in datalist:
                data_file = f"{file_pre}/{item}"
                cur_data = torch.load(data_file)
                cu_data = {}
                if c_size[0] == 0:
                    cu_data['intensity'] = cur_data['intensity']
                else:
                    cu_data['intensity'] = cur_data['intensity'][:,:,c_size[0]:-c_size[0],c_size[0]:-c_size[0],c_size[0]:-c_size[0]]
                if c_size[2] == 0:
                    cu_data['rmsf'] = cur_data['rmsf']
                else:
                    cu_data['rmsf'] = cur_data['rmsf'][:,:,c_size[2]:-c_size[2],c_size[2]:-c_size[2],c_size[2]:-c_size[2]]

                if cur_data.__contains__('keep_list'):
                    cu_data['keep_list'] = cur_data['keep_list']
                if cur_data.__contains__('total_list'):
                    cu_data['total_list'] = cur_data['total_list']
                data_dic[item] = cu_data

        elif only_sse:
            for item in datalist:
                data_file = f"{file_pre}/{item}"
                cur_data = torch.load(data_file)
                cu_data = {}
                if c_size[0] == 0:
                    cu_data['intensity'] = cur_data['intensity']
                else:
                    cu_data['intensity'] = cur_data['intensity'][:,:,c_size[0]:-c_size[0],c_size[0]:-c_size[0],c_size[0]:-c_size[0]]
                if c_size[1] == 0:
                    cu_data['sse'] = cur_data['sse']
                else:
                    cu_data['sse'] = cur_data['sse'][:,c_size[1]:-c_size[1],c_size[1]:-c_size[1],c_size[1]:-c_size[1]]

                if cur_data.__contains__('keep_list'):
                    cu_data['keep_list'] = cur_data['keep_list']
                if cur_data.__contains__('total_list'):
                    cu_data['total_list'] = cur_data['total_list']
                data_dic[item] = cu_data

        else:        
            for item in datalist:
                data_file = f"{file_pre}/{item}"
                cur_data = torch.load(data_file)
                cu_data = {}
                if c_size[0] == 0:
                    cu_data['intensity'] = cur_data['intensity']
                else:
                    cu_data['intensity'] = cur_data['intensity'][:,:,c_size[0]:-c_size[0],c_size[0]:-c_size[0],c_size[0]:-c_size[0]]
                if c_size[1] == 0:
                    cu_data['sse'] = cur_data['sse']
                else:
                    cu_data['sse'] = cur_data['sse'][:,c_size[1]:-c_size[1],c_size[1]:-c_size[1],c_size[1]:-c_size[1]]
                if c_size[2] == 0:
                    cu_data['resid'] = cur_data['resid']
                else:
                    cu_data['resid'] = cur_data['resid'][:,c_size[2]:-c_size[2],c_size[2]:-c_size[2],c_size[2]:-c_size[2]]
                if c_size[3] == 0:
                    cu_data['resname'] = cur_data['resname']
                else:
                    cu_data['resname'] = cur_data['resname'][:,c_size[3]:-c_size[3],c_size[3]:-c_size[3],c_size[3]:-c_size[3]]
                if not pre:
                    if c_size[4] == 0:
                        cu_data['rmsf'] = cur_data['rmsf']
                    else:
                        cu_data['rmsf'] = cur_data['rmsf'][:,:,c_size[4]:-c_size[4],c_size[4]:-c_size[4],c_size[4]:-c_size[4]]

                if cur_data.__contains__('keep_list'):
                    cu_data['keep_list'] = cur_data['keep_list']
                if cur_data.__contains__('total_list'):
                    cu_data['total_list'] = cur_data['total_list']
                data_dic[item] = cu_data
        
        return data_dic


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def onehot(sse):
    sse_ = sse
    sse_ = F.one_hot(sse_).permute(0, 4, 1, 2, 3)
    return sse_


def rmsf_get_average_std(cor_list):
    cor_sum, std_sum = 0, 0
    for i in cor_list:
        cor_sum = cor_sum + cor_list[i]
    cor_average = cor_sum / len(cor_list)
    for j in cor_list:
        std_sum = std_sum + (cor_list[j] - cor_average) ** 2
    std = (std_sum / len(cor_list)) ** 0.5
    return cor_average, std