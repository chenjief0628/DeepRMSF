import numpy as np
import os

def check_condition(sse_list):
    resid = []
    length = 0
    
    with open(sse_list, 'r') as sse:
        sse_lines = sse.readlines()
        for sse_line in sse_lines:
            if not sse_line.isspace():
                resid.append(sse_line.split()[1])
    base = resid[0]
    count = 0
    rid = []
    for r in resid:
        if r != base:
            base = r
            count += 1
        rid.append(count)  # nucletide indexes renumber
    length = rid[len(rid) - 1] + 1
    return length


file_path = "/data/sunxw/result_train/rna_log/all_gtresall_0227/total_time_list.npy"
data = np.load(file_path, allow_pickle=True).item()
record = "/data/sunxw/result_train/rna_log/all_gtresall_0227/time.txt"
ori_dir = "/data/sunxw/result"
ori_dir_2 = "/data/sunxw/result_add"
ori_dir_3 = "/data/sunxw/rna371"

total = []

# with open(record, 'r', encoding='utf-8') as f:
    # for key, value in data.items():
    #     sse_list = f"{ori_dir}/{key}/RNA_with_rmsf_sse.list"
    #     if os.path.exists(sse_list):
    #         length = check_condition(sse_list)
    #         line = f"{str(key)}\t{str(value)}\t{length}\n"
    #     else:
    #         line = f"{str(key)}\t{str(value)}\t{0}\n"
    #     f.write(line)
for key, value in data.items():
    rna = []
    rna.append(str(key))
    rna.append(str(value))
    sse_list = f"{ori_dir}/{key}/RNA_with_rmsf_sse.list"
    sse_list_2 = f"{ori_dir_2}/{key}/RNA_with_rmsf_sse.list"
    sse_list_3 = f"{ori_dir_3}/{key}/{key}_sse.list"
    if os.path.exists(sse_list):
        length = check_condition(sse_list)
        rna.append(str(length))
    elif os.path.exists(sse_list_2):
        length = check_condition(sse_list_2)
        rna.append(str(length))
    elif os.path.exists(sse_list_3):
        length = check_condition(sse_list_3)
        rna.append(str(length))
    total.append(rna)
print(total)

with open(record, 'w', encoding='utf-8') as f:
    for rna in total:
        line = f"{rna[0]}\t{rna[1]}\t{rna[2]}\n"
        f.write(line)