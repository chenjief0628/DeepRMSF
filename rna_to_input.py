import numpy as np
from tqdm import tqdm
import torch
import sys
sys.path.append('../..')
from rna_util import rna_get_ana_pdb1, rna_split_map_and_select_back, rna_get_dir, rna_get_smi_map
from model_util import rna_concat_data, shuffle_pdb2
import random
import mrcfile
import os


def rna_save_ana_map(pdbid_list, data_dir, ori_dir, saved=False, sim=True, pre=False, res=4):
    if not saved:
        emd_info = dict()
        assererr_id = []
        for pdbid in tqdm(pdbid_list,desc='save anontated map', total=len(pdbid_list)):
            try:
                os.makedirs(f"{data_dir}/{pdbid}", exist_ok=True)
                print(f"{pdbid}******************")
                if not pre:
                    map, res_map, rmsf_map, sse_map, resname_map, info = rna_get_ana_pdb1(pdbid, ori_dir, sim=sim, pre=pre, res=res)  # sse_map, resname_map, 
                else:
                    map, sse_map, res_map, resname_map, info = rna_get_ana_pdb1(pdbid, ori_dir, sim=sim, pre=pre, res=res)
                sse_file = f"{data_dir}/{pdbid}/{pdbid}sse_anamap.npy"
                res_file = f"{data_dir}/{pdbid}/{pdbid}res_anamap.npy"
                resname_file = f"{data_dir}/{pdbid}/{pdbid}resname_anamap.npy"
                map_file = f"{data_dir}/{pdbid}/{pdbid}_map.npy"
                np.save(sse_file, sse_map)
                np.save(res_file, res_map)
                np.save(resname_file, resname_map)
                np.save(map_file, map)
                if not pre:
                    rmsf_file = f"{data_dir}/{pdbid}/{pdbid}rmsf_anamap.npy"
                    np.save(rmsf_file, rmsf_map)
                emd_info[pdbid] = info
            except AssertionError:
                print(f"{pdbid}:Assertionerror")
                assererr_id.append(pdbid)
                continue
        info_file = f"{data_dir}/emd_info.npy"
        np.save(info_file, emd_info)
        print(f"{assererr_id} AssertionError")
    return emd_info


def rna_get_box_list(data_dir, pdbid_list, out_dir, sim=True, pre=False, res=4):
    Box_norm_list = np.empty((0,40,40,40), dtype=np.float32)
    Sse_boxn_list = np.empty((0,40,40,40), dtype=np.int32)
    Res_boxn_list = np.empty((0,40,40,40), dtype=np.int32)
    Resname_boxn_list = np.empty((0,40,40,40), dtype=np.int32)
    if not pre:
        Rmsf_boxn_list = np.empty((0,40,40,40), dtype=np.float32)
    for pdbid in tqdm(pdbid_list,desc='loop get box list'):
        print('...................................................')
        if not pre:
            box_norm_list, sse_boxn_list, res_boxn_list, resname_boxn_list, rmsf_boxn_list, keep_list, total_list = rna_split_map_and_select_back(pdbid,data_dir, sim=sim, pre=pre, res=res)
        else:
            box_norm_list, sse_boxn_list, res_boxn_list, resname_boxn_list, keep_list, total_list = rna_split_map_and_select_back(pdbid,data_dir, sim=sim, pre=pre, res=res)
        print(f"box_norm_list is {box_norm_list.shape}")
        if (len(box_norm_list) != 0):
            Box_norm_list = np.concatenate((Box_norm_list,box_norm_list), axis=0)
            Sse_boxn_list = np.concatenate((Sse_boxn_list,sse_boxn_list), axis=0)
            Res_boxn_list = np.concatenate((Res_boxn_list,res_boxn_list), axis=0)
            Resname_boxn_list = np.concatenate((Resname_boxn_list,resname_boxn_list), axis=0)
            if not pre:
                Rmsf_boxn_list = np.concatenate((Rmsf_boxn_list,rmsf_boxn_list), axis=0)
        print('..................................................')
    if not pre:
        torch.save({'intensity':torch.from_numpy(Box_norm_list).unsqueeze_(1),
                'sse':torch.from_numpy(Sse_boxn_list), 'resid':torch.from_numpy(Res_boxn_list), 'resname':torch.from_numpy(Resname_boxn_list), 'rmsf':torch.from_numpy(Rmsf_boxn_list).unsqueeze_(1), 'keep_list':torch.from_numpy(keep_list), 'total_list':torch.from_numpy(total_list)},out_dir)
    else:
        torch.save({'intensity':torch.from_numpy(Box_norm_list).unsqueeze_(1),
            'sse':torch.from_numpy(Sse_boxn_list), 'resid':torch.from_numpy(Res_boxn_list), 'resname':torch.from_numpy(Resname_boxn_list), 'keep_list':torch.from_numpy(keep_list), 'total_list':torch.from_numpy(total_list)},out_dir) 
    print(f"tensor datafile saved at {out_dir}")


def rna_get_sp_data(sp_data_dir, d_size=(40,40,40,40,10), only_rmsf=False, only_sse=False, file_folder="/data/datafile", sp_list_file="/data/input/rna_sp_list.npy"):
    '''5-fold cross-validation'''
    '''d_size: size to be obtained'''
    '''sp_data_dir: data splitting'''
    '''file_folder: pdbid.pth obtained by rna_get_box_list'''

    spl_list = np.load(sp_list_file, allow_pickle=True)
    # print(spl_list)
    print(f"got spl_list")

    for i, spl in enumerate(spl_list):
        sp_dir = f"{sp_data_dir}/{i+1}"
        print(f"data will be saved at {sp_dir}")
        if not os.path.exists(sp_dir):
            os.mkdir(sp_dir)
        train_list, val_list, test_list = spl['train'], spl['val'], spl['test']
        train_data = rna_concat_data(file_folder, train_list, dsize=d_size, train=True, only_rmsf=only_rmsf, only_sse=only_sse)
        print(f"got train_data, len{len(train_list)}")
        val_data = rna_concat_data(file_folder, val_list, dsize=d_size, train=True, only_rmsf=only_rmsf, only_sse=only_sse)
        print(f"got val_data, len{len(val_list)}")
        test_data = rna_concat_data(file_folder, test_list, dsize=d_size, train=False, only_rmsf=only_rmsf, only_sse=only_sse)
        print(f"got test_data, len{len(test_list)}")
        print(f"saving data")
        torch.save(train_data, f"{sp_dir}/train_data.pth")
        torch.save(val_data, f"{sp_dir}/val_data.pth")
        torch.save(test_data, f"{sp_dir}/test_data.pth")


def main():
    ori_dir = "PATH_TO_YOUR_DATA"
    label_map_dir = "PATH_TO_SAVE_LABEL_MAP"
    box_dir = 'PATH_TO_SAVE_BOXES'
    exp_dir = "PATH_TO_SAVE_TRAINING_RELATED_FILES"
    #Please assign three different directories for label_map, box and training
    seed = 66
    
    
    
    pdbid_list = rna_get_dir(ori_dir)
    os.makedirs(label_map_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    # get labeled maps
    rna_save_ana_map(pdbid_list, label_map_dir, ori_dir)
    # get density boxes
    for pdbid in pdbid_list:
        out_file = f"{box_dir}/{pdbid}.pth"
        rna_get_box_list(label_map_dir, [pdbid], out_file, sim=True, res=4)
    # get sp_list_file
    
    exp_name = f'exp_{seed}'
    
    sp_list_file = f"{exp_dir}/{exp_name}/rna_nw_all_input/rna_sp_list_60.npy"
    list_record = f"{exp_dir}/{exp_name}/rna_nw_all_input/list_record.txt"
    
    os.makedirs(os.path.dirname(list_record), exist_ok=True)
    pdbid_list = []
    box_dir = "/data/sunxw/wenbc_workdir/result_map_all"
    for root, dirs, files in os.walk(box_dir):
        for f in files:
            pdbid_list.append(f)
    sp_list, select_list = shuffle_pdb2(pdbid_list, seed=seed)

    np.save(sp_list_file, sp_list)
    with open(list_record,'w') as f:
        print(f"sp_list is {sp_list} \n \n", file=f)
        print(f"select_list is {select_list}", file=f)

    sp_data_dir = f"{exp_dir}/{exp_name}/rna_nw_all_input/ad_4040404010"
    os.makedirs(sp_data_dir, exist_ok=True)
    rna_get_sp_data(sp_data_dir, d_size=(40,40,40,40,10), file_folder=box_dir, sp_list_file=sp_list_file)


if __name__ == '__main__':
    main()