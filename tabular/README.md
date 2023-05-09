# Tabular approach

## Structure
```
│
├── configs                   <- Hydra configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── wandb                    <- W&B configs
│   │
│   └──config.yaml           <- Main config
│
├── tabular                    <- Source code
│   ├── data.py                     <- Data scripts
│   ├── metric.py                   <- Metric scripts
│   ├── trainer.py                  <- Trainer scripts
│   └── utils.py                    <- Utility scripts
│
├── main.py                  <- Run training and inference
├── requirements.txt         <- File for installing python dependencies
└── README.md
```
## Setup
```bash
cd /opt/ml/level2_dkt_recsys-03/tabular/
conda init
(base) . ~/.bashrc
(base) conda create -n tabular python=3.10 -y
(base) conda activate tabular
(tabular) pip install -r requirements.txt
```
## 실행 명령어

```bash
python main.py
```
## TODO
- [ ] sweep
- [ ] features -> config
