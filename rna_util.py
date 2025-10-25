import numpy as np
from moleculekit.molecule import Molecule
import os
import subprocess
import sys

sys.path.append('..')
import mrcfile
import string
import shutil
from Bio import PDB


def get_dssr(pdbid, ori_dir):
    """use x3dna-dssr"""
    path = f"{ori_dir}/{pdbid}"
    print(path)
    # pdb = f"{ori_dir}/{pdbid}/{pdbid}.pdb"
    pdb = f"{ori_dir}/{pdbid}/RNA_with_rmsf.pdb"
    os.chdir(path)
    # os.system('/home/sunxw/DeepRMSF/program/dssr-basic-linux/x3dna-dssr -i='f"{pdbid}.pdb"' -o='f"{pdbid}.out"'')
    os.system('/home/sunxw/DeepRMSF/program/dssr-basic-linux/x3dna-dssr -i='"RNA_with_rmsf.pdb"' -o='f"{pdbid}.out"'')


def get_file(pdbid, ori_dir):
    path = f"{ori_dir}/{pdbid}"
    dssr_pairs = None
    for root, dirs, files in os.walk(path):
        for f in files:
            if f == "dssr-pairs.pdb":
                dssr_pairs = f"{ori_dir}/{pdbid}/dssr-pairs.pdb"
    return dssr_pairs


def rna_get_dir(ori_dir):
    pdbid_list = []
    for root, dirs, files in os.walk(ori_dir):
        for dir in dirs:
            pdbid_list.append(dir)
    return pdbid_list


# just consider the altloc A
def select_a(pdb_file, new_file):
    f = open(pdb_file)
    lines = f.readlines()
    new_data = []
    for i in range(len(lines)):
        if lines[i].split()[0] == 'ATOM':
            a = lines[i][16]
            if a == 'A' or a == ' ':
                new_data.append(lines[i])
    f.close()
    with open(new_file, 'w') as new:
        new.writelines(new_data)
        new.close


def rna_get_smi_map(pdb_file, out_file, res=4, number=0.1, r=1.5):
    chimera_script = open('./measure.cmd', 'w')

    chimera_script.write('open ' + pdb_file + '\n'
                                              'molmap #0 ' + str(res) + ' gridSpacing ' + str(r) + '\n'
                                                                                                   'volume #' + str(
        number) + ' save ' + str(out_file) + '\n'
                                             'close all')
    chimera_script.close()
    output = subprocess.check_output(["/home/sunxw/DeepRMSF/chimera/bin/chimera", '--nogui', chimera_script.name])
    return output


def parse_map2(map):
    
    mrc=mrcfile.open(map,'r')

    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    cella = (mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)

    start_xyz=np.asarray([mrc.header.nxstart,mrc.header.nystart,mrc.header.nzstart],dtype=np.float32)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)
    map2=np.asfarray(mrc.data.copy(),dtype=np.float32)


    assert(angle[0] == angle[1] == angle[2] == 90.0)
    mapcrs = np.subtract([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)
    
    sort = np.asarray([0, 1, 2], dtype=np.int)
    
    if (mapcrs==sort).all():
        changed=False
        xyz_start = np.asarray([start_xyz[i] for i in sort])
        nxyz = np.asarray([ncrs[i] for i in sort])
        mrc.close()
    else:
        changed=True
        for i in range(3):
            sort[mapcrs[i]] = i
        xyz_start = np.asarray([start_xyz[i] for i in sort])
        nxyz = np.asarray([ncrs[i] for i in sort])
        map2 = np.transpose(map2, axes=2-sort[::-1])
        mrc.close()

    nxyz = np.asarray([map2.shape[0], map2.shape[1], map2.shape[2]], dtype=np.int)
    info = dict()
    info['cella'] = cella
    info['xyz_start'] = xyz_start
    info['voxel_size'] = voxel_size
    info['nxyz'] = nxyz
    info['origin'] = origin
    info['changed'] = changed

    return map2, info


def rna_pdb2sse(org_pdb, dssr_pairs):
    """use the secondary structure files obtained by x3dna-dssr software to label the secondary structure
    input: secondary structure file
    output: xx_sse.list"""
    pdb = Molecule(org_pdb)
    serials = pdb.serial
    # chains = pdb.chain
    resnames = pdb.resname
    num = 0

    with open(org_pdb) as o_pdb:
        lines = o_pdb.readlines()
        resids = []
    for line in lines:
        if line.split()[0] == 'ATOM' and line.split()[1] == str(serials[num]):
            resid = line[20:26]
            resids.append(resid)
            num += 1
        if num >= len(serials):
            break
    r = resids
    list_serial = []
    list_sse = [''] * len(serials)
    for i in range(len(serials)):
        list_serial.append(str(serials[i]))

    # label the base pairs
    if dssr_pairs != None:
        with open(dssr_pairs) as pairs:
            lines = pairs.readlines()
            i_pair = 0
            list_resid = []
            # list_chain = []
        for line in lines:
            if line.split()[0] == 'ATOM':
                resid_pair = line[20:26]
                # chain_pair = line[21]
                # list_chain.append(chain_pair)
                list_resid.append(resid_pair)
        # res_pair = zip(list_chain, list_resid)
        # r_p = list(res_pair)
        r_p = list_resid
        while i_pair < len(r_p):
            for i in range(len(r)):
                if r[i] == r_p[i_pair]:
                    if list_sse[i] == '':
                        list_sse[i] = 'pair'
            i_pair = i_pair + 1

    # label others
    for i in range(len(list_sse)):
        if list_sse[i] == '':
            list_sse[i] = 'nopair'

    list_resid = []
    # list_c = []
    list_r = []
    for i in range(len(resids)):
        list_resid.append(str(resids[i]))
        # list_c.append(str(chains[i]))
        list_r.append(str(resnames[i]))

    sselist_file = f"{os.path.splitext(org_pdb)[0]}_sse.list"
    with open(sselist_file, 'w') as p:
        for serid, resid, SSE, rname in zip(list_serial, list_resid, list_sse, list_r):
            p.write(f"{serid}            {resid}            {SSE}            {rname} \n ")


def rna_sse2map3(mrc, pdb, pdbsselist, r=1.5, sim=True):
    map2, info = parse_map2(mrc)
    pdb = Molecule(pdb)
    xyz = pdb.get('coords') - info['origin']
    if not sim:
        delta = (info['nxyz'] / 2 + info['xyz_start']) * (info['voxel_size'] - np.array([r, r, r]))
    else:
        delta = 0
    xyz_5A = xyz - delta
    xyz_norm = (xyz_5A / np.array([r, r, r])).round() - info['xyz_start']

    sse = []
    resn = []  # nucletide AGCU
    serial = []
    resid = []  # nucletide index
    with open(pdbsselist, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.isspace():
                serial.append(int(line.split()[0]))
                resid.append(line.split()[1])
                c = line.split()[2]
                if c == 'pair':
                    sse.append(0)
                else:
                    sse.append(1)
                r = line.split()[3]
                if r[0] == 'A':
                    resn.append(0)
                elif r[0] == 'U':
                    resn.append(1)
                elif r[0] == 'C':
                    resn.append(2)
                else:
                    resn.append(3)
    base = resid[0]
    count = 0
    rid = []
    for r in resid:
        if r != base:
            base = r
            count += 1
        rid.append(count)  # nucletide indexes renumber
    print(rid[len(rid) - 1])
    
    sse_map = np.full(map2.shape, 2)
    res_map = np.full(map2.shape, 414)
    resname_map = np.full(map2.shape, 4)
    for i in range(len(serial)):
        x = xyz_norm[i]
        sse_map[int(x[2]) - 1:int(x[2]) + 1, int(x[1]) - 1:int(x[1]) + 1, int(x[0]) - 1:int(x[0]) + 1] = sse[i]
        res_map[int(x[2]) - 1:int(x[2]) + 1, int(x[1]) - 1:int(x[1]) + 1, int(x[0]) - 1:int(x[0]) + 1] = rid[i]
        resname_map[int(x[2]) - 1:int(x[2]) + 1, int(x[1]) - 1:int(x[1]) + 1, int(x[0]) - 1:int(x[0]) + 1] = resn[i]
    return map2, res_map, sse_map, resname_map, info  # sse_map, resname_map, 


def rna_get_rmsf_list(pdb, refpdb, rmsfxvg):
    """refpdb: _ref.pdb rmsfxvg: .dat
    similar to rna_pdb2sse"""

    mol = Molecule(pdb)
    mol_md = Molecule(refpdb)
    mol_record = [f'{ci}{ri}{ni}'
                  for ci, ri, ni
                  in zip(mol.chain, mol.resid, mol.name)]
    mol_md_record = {f'{ci}{ri}{ni}': i
                     for i, (ci, ri, ni)
                     in enumerate(zip(mol_md.chain, mol_md.resid, mol_md.name))}
    serial = mol.serial
    mol_serial = []
    mol_md_serial = []
    for i in range(len(mol_record)):
        get_nm = mol_md_record.get(mol_record[i])
        if get_nm is not None:
            mol_serial.append(i + 1)
            mol_md_serial.append(get_nm + 1)
    rmsf = []
    with open(rmsfxvg, 'r') as f:
        lines = f.readlines()
        for line in lines:
            rmsf.append(float(line.split()[1]))
    ff = os.path.splitext(pdb)[0]
    file = f"{ff}_rmsf.list"
    with open(file, 'w') as w:
        for pdbnm, refnm in zip(mol_serial, mol_md_serial):
            w.write(f"{pdbnm}      {rmsf[refnm - 1]}\r")


def read_rmsf_file(rmsf_file):
    rmsf_dict = {}
    with open(rmsf_file, 'r') as f:
        title = f.readline()
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    atom_id = int(float(parts[0]))
                    rmsf_value = float(parts[1])
                    rmsf_dict[atom_id] = rmsf_value
                except ValueError:
                    continue
    return rmsf_dict


def rna_get_rmsf_list_new(pdb, rmsfxvg):
    """refpdb: _ref.pdb rmsfxvg: .dat
    similar to rna_pdb2sse"""

    mol = Molecule(pdb)
    serials = mol.serial
    rmsf_dict = read_rmsf_file(rmsfxvg)

    mol_serial = []
    for i in range(len(serials)):
        get_rf = rmsf_dict.get(serials[i])
        if get_rf is not None:
            mol_serial.append(i + 1)
    
    ff = os.path.splitext(pdb)[0]
    ff = ff[:-14]
    name = ff[-4:]
    ff = f"{ff}/{name}"
    file = f"{ff}_rmsf.list"
    with open(file, 'w') as w:
        for pdbnm in mol_serial:
            w.write(f"{pdbnm}      {rmsf_dict.get(serials[pdbnm - 1])}\r")


def rna_rmsf2map3(mrc, pdb, pdbrmsflist, r=1.5, sim=True, pre=False):
    map2, info = parse_map2(mrc)
    pdb = Molecule(pdb)
    xyz = pdb.get('coords') - info['origin']
    if not sim:
        delta = (info['nxyz'] / 2 + info['xyz_start']) * (info['voxel_size'] - np.array([r, r, r]))
    else:
        delta = 0
    xyz_5A = xyz - delta
    xyz_norm = (xyz_5A / np.array([r, r, r])).round() - info['xyz_start']

    rmsf = []
    serial_r = []
    if not pre:
        with open(pdbrmsflist, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.isspace():
                    serial_r.append(int(line.split()[0]))
                    rmsf.append(float(line.split()[1]) + 1)
        rmsf_np = np.log10(np.array(rmsf))
        rmsf_map = np.full((map2.shape[0], map2.shape[1], map2.shape[2]), 0, dtype=np.float)
        for i in range(len(serial_r)):
            x = xyz_norm[serial_r[i] - 1]
            rmsf_map[int(x[2]), int(x[1]), int(x[0])] = rmsf_np[i]
    else:
        rmsf_map = np.full((map2.shape[0], map2.shape[1], map2.shape[2]), 0, dtype=np.float)
        for i in range(len(xyz_norm)):
            x = xyz_norm[i]
            rmsf_map[int(x[2]), int(x[1]), int(x[0])] = 1  # In application, RMSF is labeled as 1, which does not affect the prediction result
    print(f"rmsf annotated voxels:{np.sum(np.where(rmsf_map == 0, 0, 1))}")
    return map2, rmsf_map, info


def rna_get_ana_pdb1(pdbid, ori_dir, sim=True, pre=False, res=4):
    """get rmsf_map and sse_map"""

    # pdb = f"{ori_dir}/{pdbid}/RNA_with_rmsf.pdb"
    pdb = f"{ori_dir}/{pdbid}/{pdbid}.pdb"
    mrc = f"{ori_dir}/{pdbid}/{pdbid}_out.mrc"
    # sse_list = f"{ori_dir}/{pdbid}/RNA_with_rmsf_sse.list"
    sse_list = f"{ori_dir}/{pdbid}/{pdbid}_sse.list"
    if not pre:
        ref_pdb = f"{ori_dir}/{pdbid}/{pdbid}_ref.pdb"
        rmsfxvg = f"{ori_dir}/{pdbid}/rmsf.dat"
        rmsf_list = f"{ori_dir}/{pdbid}/{pdbid}_rmsf.list"
        if not os.path.exists(rmsf_list):
            rna_get_rmsf_list(pdb, ref_pdb, rmsfxvg)
            # rna_get_rmsf_list_new(pdb, rmsfxvg)
            rmsf_list = f"{ori_dir}/{pdbid}/{pdbid}_rmsf.list"
    # if not os.path.exists(sse_list):
    #     get_dssr(pdbid, ori_dir)
    #     dssr_pairs = get_file(pdbid, ori_dir)
    #     rna_pdb2sse(pdb, dssr_pairs)
    #     sse_list = f"{ori_dir}/{pdbid}/{pdbid}_sse.list"

    map, res_map, sse_map, resname_map, info = rna_sse2map3(mrc, pdb, sse_list, sim=sim)  # sse_map, resname_map, 
    if not pre:
        map2, rmsf_map, info_ = rna_rmsf2map3(mrc, pdb, rmsf_list, sim=sim, pre=pre)
        assert (map == map2).all()
        return map, res_map, rmsf_map, sse_map, resname_map, info  # sse_map, resname_map, 
    return map, res_map, info  # sse_map, resname_map, 


def rna_select_map_for_split2(box, res_box, rmsf_box, sse_box, resname_box, contour_level, pre=False, th1=0.05, th2=0.005):  # sse_box, resname_box, 
    threshold = float(contour_level)
    max = np.max(box)
    if max <= threshold:
        return False
    elif pre == False:
        sum_s = np.sum(np.where(sse_box == 2, 0, 1))
        sum_c = np.sum(np.where(res_box == 414, 0, 1))  # nucletide index
        sum_res = np.sum(np.where(resname_box == 4, 0, 1))  # nucletide AGCU
        sum_r = np.sum(np.where(rmsf_box > 0, 1, 0))
        length_s = 20
        length_c = 20
        length_r = 10
        if sum_c / (length_c * length_c * length_c) < th1 or \
            sum_r / (length_r * length_r * length_r) < th2 or \
            sum_s / (length_s * length_s * length_s) < th1 or \
            sum_res / (length_c * length_c * length_c) < th1:  # sum_s / (length_s * length_s * length_s) < th1 or sum_res / (length_c * length_c * length_c) < th1 or 
            return False
    return True


def rna_split_map_and_select_back(pdbid, data_dir, pre=False, sim=True, res=4):
    map_file = f"{data_dir}/{pdbid}/{pdbid}_map.npy"
    sse_file = f"{data_dir}/{pdbid}/{pdbid}sse_anamap.npy"
    res_file = f"{data_dir}/{pdbid}/{pdbid}res_anamap.npy"
    resname_file = f"{data_dir}/{pdbid}/{pdbid}resname_anamap.npy"
    if not pre:
        rmsf_file = f"{data_dir}/{pdbid}/{pdbid}rmsf_anamap.npy"
        
        if os.path.exists(res_file) and os.path.exists(
                rmsf_file) and os.path.exists(map_file):  # os.path.exists(sse_file) and os.path.exists(resname_file) and 
            map = np.load(map_file)
            sse_map = np.load(sse_file)
            res_map = np.load(res_file)
            resname_map = np.load(resname_file)
            rmsf_map = np.load(rmsf_file)
        else:
            return np.empty((0, 40, 40, 40), dtype=np.float32), np.empty(
                (0, 40, 40, 40), dtype=np.int32), np.empty((0, 40, 40, 40), dtype=np.float32), np.nan, np.nan  # np.empty((0, 40, 40, 40), dtype=np.int32), np.empty((0, 40, 40, 40), dtype=np.int32), 
    else:
        if os.path.exists(res_file) and os.path.exists(map_file):  # os.path.exists(sse_file) and os.path.exists(resname_file) and 
            map = np.load(map_file)
            sse_map = np.load(sse_file)
            res_map = np.load(res_file)
            resname_map = np.load(resname_file)
        else:
            return np.empty((0, 40, 40, 40), dtype=np.float32), np.empty((0, 40, 40, 40), dtype=np.int32), np.empty(
                (0, 40, 40, 40), dtype=np.int32), np.empty((0, 40, 40, 40), dtype=np.int32), np.nan, np.nan

    if not sim:
        contour_level = 3 * np.std(map)
    else:
        contour_level = 2 * np.std(map)
    print(pdbid, contour_level)

    box_size = 40
    core_size = 10
    map_size = np.shape(map)
    # original map
    pad_map = np.full((map_size[0] + 2 * box_size, map_size[1] + 2 * box_size, map_size[2] + 2 * box_size), 0,
                      dtype=np.float32)
    pad_map[box_size:-box_size, box_size:-box_size, box_size:-box_size] = map

    # sse_map
    pad_sse_map = np.full((map_size[0] + 2 * box_size, map_size[1] + 2 * box_size, map_size[2] + 2 * box_size), 2,
                          dtype=np.int)
    pad_sse_map[box_size:-box_size, box_size:-box_size, box_size:-box_size] = sse_map

    # res_map
    pad_res_map = np.full((map_size[0] + 2 * box_size, map_size[1] + 2 * box_size, map_size[2] + 2 * box_size), 414,
                          dtype=np.int)
    pad_res_map[box_size:-box_size, box_size:-box_size, box_size:-box_size] = res_map

    # resname_map
    pad_resname_map = np.full((map_size[0] + 2 * box_size, map_size[1] + 2 * box_size, map_size[2] + 2 * box_size), 4,
                              dtype=np.int)
    pad_resname_map[box_size:-box_size, box_size:-box_size, box_size:-box_size] = resname_map

    # rmsf_map
    if not pre:
        pad_rmsf_map = np.full((map_size[0] + 2 * box_size, map_size[1] + 2 * box_size, map_size[2] + 2 * box_size), 0,
                            dtype=np.float32)
        pad_rmsf_map[box_size:-box_size, box_size:-box_size, box_size:-box_size] = rmsf_map

    start_point = box_size - int((box_size - core_size) / 2)

    cur_x, cur_y, cur_z = start_point, start_point, start_point

    box_list = list()
    sse_box_list = list()
    res_box_list = list()
    resname_box_list = list()
    rmsf_box_list = list()

    length = [int(np.ceil(map_size[0] / core_size)), int(np.ceil(map_size[1] / core_size)),
              int(np.ceil(map_size[2] / core_size))]
    print(f"the total box of this map is {length[0]}*{length[1]}*{length[2]}={length[0] * length[1] * length[2]}")
    keep_list = []
    total_list = []
    i = 0
    while (cur_z + (box_size - core_size) / 2 < map_size[2] + box_size):
        next_box = pad_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        next_sse_box = pad_sse_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size,
                       cur_z:cur_z + box_size]
        next_res_box = pad_res_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        next_resname_box = pad_resname_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        if not pre:
            next_rmsf_box = pad_rmsf_map[cur_x:cur_x + box_size, cur_y:cur_y + box_size,
                            cur_z:cur_z + box_size]
        cur_x += core_size
        if (cur_x + (box_size - core_size) / 2 >= map_size[0] + box_size):
            cur_y += core_size
            cur_x = start_point  # Reset
            if (cur_y + (box_size - core_size) / 2 >= map_size[1] + box_size):
                cur_z += core_size
                cur_y = start_point  # Reset
                cur_x = start_point  # Reset

        if not pre:
            if (rna_select_map_for_split2(next_box, next_res_box[10:-10, 10:-10, 10:-10],
                    next_rmsf_box[15:-15, 15:-15, 15:-15], next_sse_box[10:-10, 10:-10, 10:-10], next_resname_box[10:-10, 10:-10, 10:-10], contour_level, pre=pre)):  # next_sse_box[10:-10, 10:-10, 10:-10], next_resname_box[10:-10, 10:-10, 10:-10], 
                box_list.append(next_box)
                sse_box_list.append(next_sse_box)
                res_box_list.append(next_res_box)
                resname_box_list.append(next_resname_box)
                rmsf_box_list.append(next_rmsf_box)
                keep_list.append(i)
        else:
            if (rna_select_map_for_split2(next_box, next_res_box[10:-10, 10:-10, 10:-10],
                    None, next_sse_box[10:-10, 10:-10, 10:-10], next_resname_box[10:-10, 10:-10, 10:-10], contour_level, pre=pre)):  # next_sse_box[10:-10, 10:-10, 10:-10], next_resname_box[10:-10, 10:-10, 10:-10], 
                box_list.append(next_box)
                # sse_box_list.append(next_sse_box)
                res_box_list.append(next_res_box)
                # resname_box_list.append(next_resname_box)
                keep_list.append(i)
        total_list.append(i)
        i = i + 1
    print(f"the selected maps: {len(keep_list)}")
    print(f"the total maps: {len(total_list)}")
    if not pre:
        return np.asarray(box_list), np.asarray(sse_box_list), np.asarray(res_box_list), np.asarray(resname_box_list), np.asarray(rmsf_box_list), np.asarray(keep_list), np.asarray(total_list)  # np.asarray(sse_box_list), np.asarray(resname_box_list), 
    else:
        return np.asarray(box_list), np.asarray(res_box_list), np.asarray(keep_list), np.asarray(total_list)  # np.asarray(sse_box_list), np.asarray(resname_box_list), 
    

def copy_rna_files_recursive(src_dir, dst_dir):
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename == "RNA.pdb":
                src_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, src_dir)
                relative_path = relative_path[0:4]
                dest_folder = os.path.join(dst_dir, relative_path)
                dst_path = os.path.join(dest_folder, filename)
                shutil.copy(src_path, dst_path)
                print(f"copy: {src_path} -> {dst_path}")


def update_pdb_with_rmsf(pdb_file, pdbrmsflist, output_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', pdb_file)

    rmsf = {}

    with open(pdbrmsflist, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.isspace():
                    rmsf[int(line.split()[0])] = float(line.split()[1])
    
    i = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    i = i + 1
                    if i in rmsf:
                        rmsf_value = rmsf[i]
                        atom.set_bfactor(rmsf_value)
    
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)
    print(f"Updated PDB file with RMSF values saved to {output_file}")



def main():
    # source_folder = "/hdd_data/zt_data/result"
    # destination_folder = "/data/sunxw/pdbs/rna157+6"
    # copy_rna_files_recursive(source_folder, destination_folder)
    ori_dir = "/data/sunxw/pdbs/result"
    for root, dirs, files in os.walk(ori_dir):
        for dir in dirs:
            # if dir == "result": # 7TD7
            #     continue
            directory = f"/data/sunxw/pdbs/result/{dir}"

            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                if os.path.isfile(file_path) and filename != f"{dir}_pre_id.pdb" and filename != f"{dir}_pre_id_resname.pdb":
                    try:
                        os.remove(file_path)
                        print(f"remove: {filename}")
                    except Exception as e:
                        print(f"remove {filename} file, reason: {str(e)}")

    # print("success!")

    # step one: use x3dna-dssr
    # for i in range(1, 21):
    #     ori_dir = f"/data/rna/{i}"
    #     for root, dirs, files in os.walk(ori_dir):
    #         for dir in dirs:
    #             get_dssr(dir, ori_dir)
    # ori_dir = "/data/sunxw/result_add"
    # for root, dirs, files in os.walk(ori_dir):
    #     for dir in dirs:
    #         get_dssr(dir, ori_dir)

    # step two：determine the secondary structure of each residue and generate a file
    # for i in range(1, 21):
    #     ori_dir = f"/data/rna/{i}"
    #     for root, dirs, files in os.walk(ori_dir):
    #         for dir in dirs:
    #             org_pdb = f"{ori_dir}/{dir}/{dir}.pdb"
    #             dssr_pairs = get_file(dir, ori_dir)
    #             rna_pdb2sse(org_pdb, dssr_pairs)
    # ori_dir = "/data/sunxw/result_add"
    # for root, dirs, files in os.walk(ori_dir):
    #     for dir in dirs:
    #         org_pdb = f"{ori_dir}/{dir}/RNA_with_rmsf.pdb"
    #         dssr_pairs = get_file(dir, ori_dir)
    #         rna_pdb2sse(org_pdb, dssr_pairs)

    # step three：simulated map
    # for i in range(1, 21):
    #     ori_dir = f"/data/rna/{i}"
    #     for root, dirs, files in os.walk(ori_dir):
    #         for dir in dirs:
    #             org_pdb = f"{ori_dir}/{dir}/{dir}_a.pdb"  # just consider the altloc A
    #             out_file = f"{ori_dir}/{dir}/{dir}_out.mrc"
    #             output = rna_get_smi_map(org_pdb, out_file)
    #             print(output)
    # ori_dir = "/data/sunxw/result_add"
    # for root, dirs, files in os.walk(ori_dir):
    #     for dir in dirs:
    #         org_pdb = f"{ori_dir}/{dir}/RNA_with_rmsf.pdb"  # just consider the altloc A
    #         out_file = f"{ori_dir}/{dir}/{dir}_out.mrc"
    #         output = rna_get_smi_map(org_pdb, out_file)
    #         print(output)
    # ori_dir = "/data/sunxw/pdbs/rna371"
    # for root, dirs, files in os.walk(ori_dir):
    #     for dir in dirs:
    #         org_pdb = f"{ori_dir}/{dir}/{dir}.pdb"
    #         rmsf = f"{ori_dir}/{dir}/{dir}_rmsf.list"
    #         out_file = f"{ori_dir}/{dir}/RNA_with_rmsf.pdb"
    #         update_pdb_with_rmsf(org_pdb, rmsf, out_file)
    # pdb = "/data/sunxw/rna371/pdb1fir/pdb1fir.pdb"
    # refpdb = "/data/sunxw/rna371/pdb1fir/pdb1fir_ref.pdb"
    # dat = "/data/sunxw/rna371/pdb1fir/pdb1fir.dat"
    # rmsf = "/data/sunxw/rna371/pdb1fir/pdb1fir_rmsf.list"
    # output_file = "/data/sunxw/rna371/pdb1fir/RNA_with_rmsf.pdb"
    # update_pdb_with_rmsf(pdb, rmsf, output_file)
    # pdb = "/data/sunxw/result/1DUH/RNA_with_rmsf.pdb" # 5NQI
    # rmsf = "/data/sunxw/result/1DUH/rmsf.dat"
    # mrc = "/data/sunxw/result/1DUH/1DUH_out.mrc"
    # sse_list = "/data/sunxw/result/1DUH/RNA_with_rmsf_sse.list"
    # # rna_get_rmsf_list_new(pdb, rmsf)
    # map, sse_map, res_map, resname_map, info = rna_sse2map3(mrc, pdb, sse_list, sim=True)


if __name__ == '__main__':
    main()
    