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

### Step 2: Five-Fold-Validation

### Configure Parameters
Edit `rna_five_fold.py` and modify the following parameters:
- seed
- exp_dir

### Execute Five-Fold-Validation
```bash
python five_fold_validation.py
```
