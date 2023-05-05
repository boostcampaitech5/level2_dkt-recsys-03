# Tabular Approach

## Setup
```bash
cd /opt/ml/level2_dkt_recsys-03/tabular/
conda init
(base) . ~/.bashrc
(base) conda create -n tabular python=3.10 -y
(base) conda activate tabular
(tabular) pip install -r requirements.txt
(tabular) python train.py
(tabular) python inference.py
```