# Five-Fold Cross-Validation for DeepRMSF

## Overview
This pipeline performs five-fold cross-validation on RNA data through a two-step process.

## Step 1: Data Preparation

### Configure Parameters
Edit `rna_to_input.py` and modify the following parameters:
- ori_dir
- label_map_dir  
- box_dir
- exp_dir
- seed

### Execute Data Preparation
```bash
python rna_to_input.py
```

## Step 2: Five-Fold-Validation

### Configure Parameters
Edit `rna_five_fold.py` and modify the following parameters:
- seed
- exp_dir

### Execute Five-Fold-Validation
```bash
python five_fold_validation.py
```
## Results of 10 Experiments of Five-Fold Validation

We conducted **10 rounds of five-fold validation** using different random seeds. The results are organized in the `result_train` directory. Each experiment has its own subdirectory named `exp_$seed$`.  

For each experiment directory:

| Directory / File | Description |
|-----------------|-------------|
| `rna_nw_all_input` | Records the dataset splits for the five-fold validation. |
| `rna_log/all_nosse_0407/visualize` | Contains the inference results from the trained models. |
| `rna_log/all_nosse_0407/1-5` | Stores the models obtained from the 1st to 5th fold validation. |
| `rna_log/all_nosse_0407/record.txt` | Summary of the experiment. |


