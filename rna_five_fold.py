import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import sys
sys.path.append('../..')
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn import metrics

from model_util import EarlyStopping, generate_mask, onehot, rmsf_get_average_std
import numpy as np

from rmsf_model import rmsf_model, init_weights
from moleculekit.molecule import Molecule
import time
from rna_util import parse_map2, rna_get_dir
from rna_to_input import rna_save_ana_map


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
                dataset = Data.TensorDataset(intensity, sse, resid, resname, rmsf, con)  # sse, resname, 
            else:
                dataset = Data.TensorDataset(intensity, sse, resid, resname, rmsf)  # sse, resname, 

        else:
            dataset = Data.TensorDataset(intensity, sse, resid, resname)    # sse, , resname
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


def box2map(ana_pre, keep_list, info, box_size=40, core_size=10, pad=None):
    '''anapre: rmsf_pre which is generated during prediction,
    info: info which is generated through parse_map'''
    
    print(ana_pre.shape)
    csize = (ana_pre.shape[-1] - core_size) // 2

    if csize == 0:
        ana_box_list = ana_pre    
    else:
        ana_box_list = ana_pre[:,csize:-csize,csize:-csize,csize:-csize]  
    map_size = info['nxyz']
    edge_size = (box_size - core_size) // 2
    
    if not pad:
        pad = 0      

    pad_map = np.full((map_size[0]+2*box_size,map_size[1]+2*box_size,map_size[2]+2*box_size), pad, dtype=np.float32)
    
    start_point = box_size - int((box_size - core_size) / 2)
    cur_x, cur_y, cur_z = start_point, start_point, start_point

    i = 0  
    cur_pos = 0   
    if True:
        while (cur_z + (box_size - core_size) / 2 < map_size[2] + box_size):
            cur = keep_list[cur_pos]
            if i == cur:
                next_ana_box = ana_box_list[cur_pos]  
                pad_map[cur_x+edge_size:cur_x + box_size-edge_size, cur_y+edge_size:cur_y + box_size-edge_size, cur_z+edge_size:cur_z + box_size-edge_size] = next_ana_box
                cur_pos = cur_pos+1
                if cur_pos >= len(keep_list):
                    break
            i = i + 1

            cur_x += core_size
            if (cur_x + (box_size - core_size) / 2 >= map_size[0] + box_size):
                cur_y += core_size
                cur_x = start_point # Reset
                if (cur_y + (box_size - core_size) / 2  >= map_size[1] + box_size):
                    cur_z += core_size
                    cur_y = start_point # Reset
                    cur_x = start_point # Reset

    ana_map = pad_map[box_size:-box_size,box_size:-box_size,box_size:-box_size]  

    return ana_map


def write_rmsf_pdb(pdb_file, ana_map, info, save_file, r=1.5):
    '''write prediction results to the pdb'''
    mol = Molecule(pdb_file)        
    xyz = mol.get('coords')-info['origin']
    xyz_norm = (xyz / np.array([r,r,r])).round() - info['xyz_start']

    ana_list = []

    for cor in xyz_norm:
        try:
            x, y, z = int(cor[2]), int(cor[1]), int(cor[0])
            ana_list.append(10 ** ana_map[x,y,z] - 1)
        except:
            ana_list.append(0.)
            continue
    ana_list = np.array(ana_list)
    mol.set('beta', np.array(ana_list))
    mol.write(save_file)
    print(f"the pdb visualization file saved at {save_file}")
    return ana_list


def nor_rmsf(pre_pdb, save_pre):
    pre_pdb = Molecule(pre_pdb)
    pre_rmsf = pre_pdb.beta
    res = pre_pdb.resid

    #  fill 0 with predicted values of the same residue
    for i in range(len(res)):
        k = 1
        while k < 50:
            if pre_rmsf[i] != 0:
                break
            elif pre_rmsf[i] == 0 and pre_rmsf[i+k] != 0 and res[i] == res[i+k] and i+k < len(res):
                pre_rmsf[i] = pre_rmsf[i+k]
                break
            elif i + k >= len(res):
                break
            k += 1
    for i in range(len(res)):
        k = 1
        while k < 50:
            if pre_rmsf[len(res)-1-i] != 0:
                break
            elif pre_rmsf[len(res)-1-i] == 0 and pre_rmsf[len(res)-1-i-k] != 0 and res[len(res)-1-i] == res[len(res)-1-i-k] and len(res)-1-i-k >= 0:
                pre_rmsf[len(res)-1-i] = pre_rmsf[len(res)-1-i-k]
                break
            elif len(res) - 1 - i - k < 0:
                break
            k += 1

    avg_pre = sum(pre_rmsf) / len(pre_rmsf)
    std_pre = (sum([(x2-avg_pre)**2 for x2 in pre_rmsf]) / len(pre_rmsf)) ** 0.5
    pre = (pre_rmsf - avg_pre) / std_pre

    pre_pdb.set('beta', np.array(pre))
    pre_pdb.write(save_pre)
    print(f"the pdb visualization file saved at {save_pre}")


def figure(pre_nor):
    pdb = Molecule(pre_nor)
    x = pdb.serial
    y = pdb.beta
    plt.figure(figsize=(30,5))
    plt.plot(x, y, c='#f96ccc')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', width=2, labelsize=20, direction='in')
    plt.tick_params(which='major', length=20)
    plt.tick_params(which='minor', length=5)
    plt.xlabel('Atom ID', fontsize=30)
    plt.ylabel('Dynamics', fontsize=30)
    x_locator = MultipleLocator(1000)
    y_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.xlim(0, len(x))


def train_and_validate_for_rmsf(model, i, log_dir, data_dir, f, lr=4e-3, batch_size=16, early_stoping=30, sse=False, sse_model=None):  
    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    cur_log_dir = f"{log_dir}/{i}"
    if not os.path.exists(cur_log_dir):
        os.mkdir(cur_log_dir)

    model_file = f"{cur_log_dir}/model.pth"
    print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
    if os.path.exists(model_file):
        model.cuda()   
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_file))
        print('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC')
        return model, sse_model
    print('DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')

    train_data_file = f"{data_dir}/{i}/train_data.pth"
    val_data_file = f"{data_dir}/{i}/val_data.pth"
    train_data = torch.load(train_data_file)
    val_data = torch.load(val_data_file)

    print(f"* {i}th train and validate rmsf train * \n \n", file=f)
    print(f"* {i}th train and validate rmsf train * \n \n")
    print(f"model and other files saved at {cur_log_dir}", file=f)
    print(f"model and other files saved at {cur_log_dir}")

    model = model.cuda()
    model = nn.DataParallel(model)
    loss_fn1 = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(optimizer.step)

    num_epoch = 100
    start = time.time()
    early_stopping = EarlyStopping(cur_log_dir, patience=30, verbose=True)

    train_loss_rmsf = []
    eval_loss_rmsf = []
    eval_cor = []
    train_loader = get_loader(train_data, batch=batch_size, cuda=False, only_rmsf=False, contour=False)
    val_loader = get_loader(val_data, batch=batch_size, cuda=False, only_rmsf=False, contour=False)

    device = torch.device("cuda")

    for epoch in range(num_epoch):
        print(epoch,'**********************', file=f)
        print(epoch,'**********************')
        model.train(True)
        for i, (train_data, train_sse, train_resid, train_resname, train_rmsf) in enumerate(train_loader):  # train_sse, 
            train_data = train_data.to(device)
            train_sse = train_sse.to(device)
            train_resid = train_resid.to(device)
            train_resname = train_resname.to(device)
            train_rmsf = train_rmsf.to(device)

            train_rmsf_mask = generate_mask(train_rmsf, mask=0, rmsf=True, tensor=True)
            
            
            loss1_epoch = []
            if not sse:   
                model_rmsf = model(train_data)  
            else:
                train_sse = onehot(train_sse)
                train_resid = onehot(train_resid)
                train_resname = onehot(train_resname)

                model_rmsf = model(train_data, sse=train_sse, resid=train_resid, resname=train_resname)  # sse=train_sse, 
            rmsf_pre = model_rmsf * (train_rmsf_mask)    
            loss1 = loss_fn1(rmsf_pre, train_rmsf) / torch.sum(train_rmsf_mask)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()     
            if i % 100 == 0:
                print(f"epoch:{epoch}|{num_epoch}  batch:{i}   loss1:", loss1.item(), file=f)    
            loss1_epoch.append(loss1.item())
            del train_data, train_sse, train_resid, train_resname, train_rmsf, train_rmsf_mask, model_rmsf, rmsf_pre  # train_sse, 
        loss1_epoch_mean = np.array(loss1_epoch).mean()
        train_loss_rmsf.append(loss1_epoch_mean)
    
        model.eval()
        
        loss = 0
        length_rmsf = 0
        cor_list = []
        with torch.no_grad():
            for i, (val_data, val_sse, val_resid, val_resname, val_rmsf) in enumerate(val_loader):  # val_sse, 
                val_data = val_data.to(device)
                val_sse = val_sse.to(device)
                val_resid = val_resid.to(device)
                val_resname = val_resname.to(device)
                val_rmsf = val_rmsf.to(device)

                val_rmsf_mask = generate_mask(val_rmsf, mask=0, rmsf=True, tensor=True)
                if not sse:
                    model_rmsf = model(val_data)  
                else:
                    val_sse = onehot(val_sse)
                    val_resid = onehot(val_resid)
                    val_resname = onehot(val_resname)
                    model_rmsf = model(val_data, sse=val_sse, resid=val_resid, resname=val_resname)  # sse=val_sse, 
                    
                rmsf_pre = model_rmsf * (val_rmsf_mask)
                loss1 = loss_fn1(rmsf_pre,val_rmsf)
                length_rmsf = length_rmsf+torch.sum(val_rmsf_mask)
                loss = loss + float(loss1)         

                rmsf_pree = torch.take(rmsf_pre.flatten(), torch.where(val_rmsf.flatten()!=0)[0])
                val_rmsff = torch.take(val_rmsf.flatten(), torch.where(val_rmsf.flatten()!=0)[0])
                rmsf_preee = rmsf_pree.cpu().detach().numpy()
                val_rmsfff = val_rmsff.cpu().detach().numpy()
                cor = np.corrcoef(rmsf_preee, val_rmsfff)[0, 1]
                cor_list.append(round(cor, 3))
                del val_data, val_sse, val_resid, val_resname, val_rmsf, val_rmsf_mask, model_rmsf, rmsf_pre, rmsf_pree, val_rmsff, rmsf_preee, val_rmsfff  # val_sse, 
                    
            epoch_loss_rmsf = loss / length_rmsf
            epoch_cor = np.mean(np.array(cor_list))

            eval_loss_rmsf.append(epoch_loss_rmsf)
            eval_cor.append(epoch_cor)

        torch.cuda.empty_cache()   
        early_stopping(epoch_loss_rmsf, model)
        if early_stopping.early_stop:
            print("Early stopping at epoch{epoch}", file=f)
            break
    end = time.time()
    print(f"the total time for training is {end-start}", file=f)
    print(f"the total time for training is {end-start}")
    print('$$$$$$$$$$$$$$$$$$$$$$$ \n \n', file=f)
    print('$$$$$$$$$$$$$$$$$$$$$$$ \n \n')

    print(f"train_loss_rmsf is {train_loss_rmsf} \n \n", file=f)
    print(f"eval_loss_rmsf is {eval_loss_rmsf} \n \n", file=f)
    print(f"eval_cor is {eval_cor} \n \n", file=f)  
    
    
    train_log = {'train_loss_rmsf':np.array(train_loss_rmsf),
            'eval_loss_rmsf':np.array(eval_loss_rmsf),'eval_cor':np.array(eval_cor)}
    train_log_file = f"{cur_log_dir}/train_log.npy"
    np.save(train_log_file,train_log)
    print(f"train_log saved at {train_log_file}", file=f)
    print(f"train_log saved at {train_log_file}")

    model.load_state_dict(torch.load(f"{cur_log_dir}/best_network.pth"))
    model_file = f"{cur_log_dir}/model.pth"
    torch.save(model.state_dict(), model_file)
    print(f"model saved at {model_file}", file=f )
    print(f"model saved at {model_file}")
    return model, sse_model


def test_model_for_rmsf(model, i, log_dir, data_dir, ori_dir, f, sse=False, sse_model=None, pre=False): 
    device = torch.device("cpu")

    if sse_model and sse_model.module:
        sse_model = sse_model.module
        sse_model = sse_model.to(device)

    if not pre:
        if model.module:
            model = model.module
            model = model.to(device)
    else:
        model = model.to(device)

    if pre:
        cur_log_dir = f"{log_dir}"
        test_Data_file = f"{data_dir}/test_data.pth"
    else:
        cur_log_dir = f"{log_dir}/{i}"
        test_Data_file = f"{data_dir}/{i}/test_data.pth"
    test_Data = torch.load(test_Data_file)
    if sse_model:
        sse_model.eval()
    model.eval()
    cor_list = {}
    mae_list = {}
    rmse_list = {}
    mean_loss_list = {}
    loss_fn1 = nn.MSELoss(reduction='sum')
    time_list = {}

    # get info
    total_info = {}
    if pre:
        pdbid_list = rna_get_dir(ori_dir)
        for pdbid in pdbid_list:
            mrc = f"{ori_dir}/{pdbid}/{pdbid}_out.mrc"
            map2, info = parse_map2(mrc)
            total_info[pdbid] = info
    else:
        ori_dir_list = ["/data/sunxw/result", '/data/sunxw/result_add', '/data/sunxw/rna371']
        for ori_dir in ori_dir_list:
            pdbid_list = rna_get_dir(ori_dir)
            for pdbid in pdbid_list:
                mrc = f"{ori_dir}/{pdbid}/{pdbid}_out.mrc"
                map2, info = parse_map2(mrc)
                total_info[pdbid] = info
        # for j in range(1, 21):
        #     ori_dir = f"/data/rna/{j}"
        #     pdbid_list = rna_get_dir(ori_dir)
        #     for pdbid in pdbid_list:
        #         mrc = f"{ori_dir}/{pdbid}/{pdbid}_out.mrc"
        #         map2, info = parse_map2(mrc)
        #         total_info[pdbid] = info

    with torch.no_grad():
        for item in test_Data.keys():
            start = time.time()
            item_name = item[:-4]
            # these PDBs do not have RMSF data
            # if not pre:
                # if item_name == '1KFO':
                #     continue
                # if item_name == 'pdb5new' or item_name == 'pdb5npm' or item_name == 'pdb5ns3' or item_name == 'pdb4q9q' or item_name == 'pdb3egz':
                #     continue
            print(f"* test * {item_name} \n \n",file=f)
            print(f"* test * {item_name} \n \n")
            pdb = test_Data[item]
            if sse:
                test_sse = pdb['sse'].to(device)
            if not pre:
                test_data, test_sse, test_resid, test_resname, test_rmsf = pdb['intensity'].to(device), test_sse.to(device), pdb['resid'].to(device), pdb['resname'].to(device), pdb['rmsf'].to(device)
                test_rmsf_mask = generate_mask(test_rmsf, mask=0, rmsf=True, tensor=True)
            else:
                test_data, test_resid, test_resname = pdb['intensity'].to(device), pdb['resid'].to(device), pdb['resname'].to(device)
            keep_list = pdb['keep_list'].to(device)

            if len(test_resid) == 0 or len(test_resname) == 0:  # len(test_sse) == 0 or 
                continue
            
            if not sse:   
                model_rmsf = model(test_data)  
            else:
                test_sse = onehot(test_sse)
                test_resid = onehot(test_resid)
                test_resname = onehot(test_resname)
                model_rmsf = model(test_data, sse=test_sse, resid=test_resid, resname=test_resname)  # sse=test_sse, 

            if not pre:
                rmsf_pre = model_rmsf * (test_rmsf_mask)
                end = time.time()
                ttime = end - start
                time_list[item_name] = ttime

                loss1 = loss_fn1(rmsf_pre, test_rmsf)
                length_rmsf = torch.sum(test_rmsf_mask)
                loss = loss1

                # PCC
                rmsf_pree = torch.take(rmsf_pre.flatten(), torch.where(test_rmsf.flatten()!=0)[0])
                test_rmsff = torch.take(test_rmsf.flatten(), torch.where(test_rmsf.flatten()!=0)[0])
                if 'cuda' in str(rmsf_pree.device):
                    rmsf_preee = rmsf_pree.cpu().detach().numpy()
                    test_rmsfff = test_rmsff.cpu().detach().numpy()
                else:
                    rmsf_preee = rmsf_pree.detach().numpy()
                    test_rmsfff = test_rmsff.detach().numpy()
                cor = round(np.corrcoef(rmsf_preee, test_rmsfff)[0,1], 3)
                print(f"cor for {item_name}:{cor} \n \n", file=f)
                mae = round(metrics.mean_absolute_error(test_rmsfff, rmsf_preee), 3)
                print(f"MAE for {item_name}:{mae} \n \n", file=f)
                rmse = round(np.sqrt(metrics.mean_squared_error(test_rmsfff, rmsf_preee)), 3)
                print(f"RMSE for {item_name}:{rmse} \n \n", file=f)
                cor_list[item_name] = cor
                mae_list[item_name] = mae
                rmse_list[item_name] = rmse
            else:
                rmsf_pre = model_rmsf
                end = time.time()
                ttime = end - start
                time_list[item_name] = ttime

            # predicted values are written to the PDB
            # if item_name[0:3] != 'pdb':
            if True:
                rmsf_pre = torch.squeeze(rmsf_pre, 1)
                rmsf_pre = np.array(rmsf_pre)
                ana_map = box2map(rmsf_pre, keep_list, total_info[item_name])
                if pre:
                    for root, dirs, files in os.walk(ori_dir):
                        for dir in dirs:
                            if dir == item_name:
                                org_pdb = f"{ori_dir}/{dir}/{dir}.pdb"
                    save_file = f"{log_dir}/{item_name}_pre.pdb"
                    ana_list = write_rmsf_pdb(org_pdb, ana_map, total_info[item_name], save_file)
                    # normalized
                    save_pre = f"{log_dir}/{item_name}_pre_nor.pdb"
                    nor_rmsf(save_file, save_pre)
                    os.chdir(log_dir)
                    os.unlink(save_file)
                    # output figure
                    figure(save_pre)
                    image = f"{log_dir}/{item_name}_pre_nor.png"
                    plt.savefig(fname=image, bbox_inches='tight')
                    plt.show()
                else:
                    add_pdb = ['3DHS', '3E5C', '3F2Q', '3GX2', '5FKD', '5HBY']
                    fun = lambda x: 'rna371' if x.startswith('pdb') else ('result_add' if x in add_pdb else 'result')
                    fun2 = lambda x: f'{x}.pdb' if x.startswith('pdb') else 'RNA_with_rmsf.pdb'
                    
                    # ori_dir = "/data/sunxw/result"
                    ori_dir = f"/data/sunxw/{fun(item_name)}"
                    org_pdb = f"{ori_dir}/{item_name}/{fun2(item_name)}"
                    # for root, dirs, files in os.walk(ori_dir):
                    #     for dir in dirs:
                    #         if dir == item_name:
                    #             org_pdb = f"{ori_dir}/{dir}/RNA_with_rmsf.pdb"
                    # save_file = f"{ori_dir}/{item_name}/{item_name}_pre_id_resname.pdb"
                    save_file = f"{log_dir}/visualize/{fun(item_name)}/{item_name}_pre_id_resname.pdb"
                    os.makedirs(f"{log_dir}/visualize/{fun(item_name)}", exist_ok=True)
                    # if item_name != '7LYG' and item_name != '1YFG' and item_name != '4GMA' and item_name != '7EEM' and item_name != '7TZU':
                    #     ana_list = write_rmsf_pdb(org_pdb, ana_map, total_info[item_name], save_file)
                    ana_list = write_rmsf_pdb(org_pdb, ana_map, total_info[item_name], save_file)
                # for j in range(1, 21):
                #     ori_dir = f"/data/rna/{j}"
                #     for root, dirs, files in os.walk(ori_dir):
                #         for dir in dirs:
                #             if dir == item_name:
                #                 org_pdb = f"{ori_dir}/{dir}/{dir}.pdb"
                #     save_file = f"{ori_dir}/{item_name}_pre.pdb"
                #     ana_list = write_rmsf_pdb(org_pdb, ana_map, total_info[item_name], save_file)
            
            if sse:
                del test_resid, test_sse, test_resname  # test_sse, 
            if not pre:
                del test_data, test_rmsf, test_rmsf_mask, rmsf_pre, rmsf_pree, test_rmsff
                mean_loss = loss / length_rmsf
                mean_loss_list[item_name] = float(mean_loss.item())
                print(f"{item_name} loss for rmsf is {mean_loss} \n ", file=f)
            else:
                del test_data, rmsf_pre
    
        if not pre:
            print(f"mean_loss of rmsf for every pdb is {mean_loss_list} \n \n", file=f)
            print(f"cor of rmsf for every pdb is {cor_list} \n \n", file=f)
            print(f"cor of rmsf for every pdb is {cor_list} \n \n")
            print(f"MAE of rmsf for every pdb is {mae_list} \n \n", file=f)
            print(f"MAE of rmsf for every pdb is {mae_list} \n \n")
            print(f"RMSE of rmsf for every pdb is {rmse_list} \n \n", file=f)
            print(f"RMSE of rmsf for every pdb is {rmse_list} \n \n")

            test_log = {'cor_list': cor_list, 'mae_list': mae_list, 'rmse_list': rmse_list, 'mean_loss': mean_loss}
            test_log_file = f"{cur_log_dir}/test_log.npy"  # test 60 rna
            np.save(test_log_file, test_log)
            print(f"test log saved at {test_log_file}", file=f)
            print(f"test log saved at {test_log_file}")

    return cor_list, mae_list, rmse_list, time_list


def train_and_test_rmsf(record, log_dir, data_dir, ori_dir, sse=True, sse_model=None, sse_model_dir=None):
    total_cor_list = {}
    total_mae_list = {}
    total_rmse_list = {}
    total_time_list = {}
    if sse:
        input_chan = 424
    else:
        input_chan = 1
    with open(record,'w') as f:
        for i in [1,2,3,4,5]:
            model = rmsf_model(in_channels=input_chan)
            init_weights(model)
            if sse:
                if sse_model and sse_model_dir:
                    sse_model_file = f"{sse_model_dir}/{i}/sse_model.pth"
                    trained_model, sse_model = train_and_validate_for_rmsf(model, i, log_dir, data_dir, f, sse=True, sse_model=sse_model, sse_model_file=sse_model_file)
                    cor_list, mae_list, rmse_list, time_list = test_model_for_rmsf(trained_model, i, log_dir, data_dir, ori_dir, f, sse=True, sse_model=sse_model)
                else:
                    trained_model, sse_model = train_and_validate_for_rmsf(model, i, log_dir, data_dir, f, sse=True,)
                    cor_list, mae_list, rmse_list, time_list = test_model_for_rmsf(trained_model, i, log_dir, data_dir, ori_dir, f, sse=True)
            else:
                trained_model,sse_model = train_and_validate_for_rmsf(model, i, log_dir, data_dir, f, sse=False)
                cor_list, mae_list, rmse_list, time_list = test_model_for_rmsf(trained_model, i, log_dir, data_dir, ori_dir, f, sse=False)

            total_cor_list.update(cor_list)
            total_mae_list.update(mae_list)
            total_rmse_list.update(rmse_list)
            total_time_list.update(time_list)
            print(f"cor_list is {cor_list}")
            print(f"mae_list is {mae_list}")
            print(f"rmse_list is {rmse_list}")
            
        print(f"total_time_list is {total_time_list}", file=f)
        np.save(f"{log_dir}/total_time_list.npy", total_time_list)
        print(f"total_time_list is {total_time_list} and saved ")
        print(f"total_cor_list is {total_cor_list}", file=f)
        np.save(f"{log_dir}/total_cor_list.npy", total_cor_list)
        print(f"total_cor_list is {total_cor_list} and saved ")
        print(f"total_mae_list is {total_mae_list}", file=f)
        np.save(f"{log_dir}/total_mae_list.npy", total_mae_list)
        print(f"total_mae_list is {total_mae_list} and saved ")
        print(f"total_rmse_list is {total_rmse_list}", file=f)
        np.save(f"{log_dir}/total_rmse_list.npy", total_rmse_list)
        print(f"total_rmse_list is {total_rmse_list} and saved ")
        cor_average, std = rmsf_get_average_std(total_cor_list)
        print(f"average of correlation is {cor_average} \n \n", file=f)
        print(f"average of correlation is {cor_average} \n \n")
        print(f"std of correlation is {std} \n \n", file=f)
        print(f"std of correlation is {std} \n \n")


def count_param(model):
    param_count = 0
    for param in model().parameters():
        param_count += param.view(-1).size()[0]
    return param_count
        

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
    seed = 57
    #Please use the same seed as in the rna_to_input.py file
    exp_dir = "PATH_TO_SAVE_TRANING_RELATED_FILES"
    #Please assign the same path as in the rna_to_input.py file
    
    
    exp_name = f'exp_{seed}'
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    data_dir = f"{exp_dir}/{exp_name}/rna_nw_all_input/ad_4040404010"
    log_dir = f"{exp_dir}/{exp_name}/rna_log/all_nosse_0407"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    record = f"{log_dir}/record.txt"    
    
    train_and_test_rmsf(record, log_dir, data_dir, ori_dir=None, sse=True)


if  __name__=='__main__':
    main()