# Code for the Paper Experiments

This repository provides Python code for the experiments conducted in the paper.

Some of the datasets used in the experiments are included in this repository.  
However, the **PolyInfo dataset** cannot be redistributed and therefore is **not included** due to redistribution restrictions.

---

## Dataset Availability

### Included datasets
The repository contains datasets that allow users to run the code and verify the implementation **without the PolyInfo dataset**.

The experimental pipeline is designed to work even when the PolyInfo dataset is not available.

### PolyInfo dataset (for full reproducibility)
To fully reproduce the experimental results reported in the paper, users need to collect the dataset of experimental values of heat capacities from **PolyInfo**:

- **PolyInfo**: https://polymer.nims.go.jp/en/  
- Reference: Otsuka et al., 2011

Due to redistribution restrictions, the PolyInfo dataset must be obtained directly from the official distributor.

---

## Reproducing the Paper Results

The notebook  
[`0_CreateDataset.ipynb`](https://github.com/ryoji-himeno/paper_LCP_code/blob/main/0_CreateDataset.ipynb)  
contains **two alternative implementations**:

- **Default code (active)**:  
  Runs without the PolyInfo dataset.

- **Commented-out code**:  
  When the commented sections are restored and the PolyInfo dataset is provided,  
  the dataset construction used in the paper can be reproduced.

By replacing the default implementation with the commented-out code in  
`0_CreateDataset.ipynb` and supplying the PolyInfo dataset, users can fully reproduce the experimental setup and results described in the paper.

---

## Requirements
- Python 3.x
- Required Python packages are listed in `MyPackages.py`

---

## Usage
Please refer to the provided notebooks and scripts for detailed instructions on running the experiments.

---

## Reproducibility
This repository contains the code used in the manuscript:
"Design and Phase Control of Liquid Crystal Polymers Using Machine Learning"

## License
- Code: MIT License  
- Datasets: MIT License
