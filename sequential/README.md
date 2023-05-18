# Sequential approach

## Structure
```
│
├── configs                   <- Hydra configs
│   ├── data                     <- Data configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── wandb                    <- W&B configs
│   │
│   └──config.yaml               <- Main config
│
├── sequential                <- Source code
│   ├── dataloader.py              <- DataLoader scripts
│   ├── metrics.py                 <- Metric scripts
│   ├── models.py                  <- Models scripts
│   ├── trainer.py                 <- Trainer scripts
│   └── utils.py                   <- Utility scripts
│
├── main.py                  <- Run training and inference
├── requirements.txt         <- File for installing python dependencies
└── README.md
```

## Setup
```bash
cd /opt/ml/level2_dkt_recsys-03/sequential/
conda init
(base) . ~/.bashrc
(base) conda create -n sequential python=3.10 -y
(base) conda activate sequential
(sequential) pip install -r requirements.txt
```

## How to run
```bash
python main.py
```

## Sweep
```bash
wandb sweep configs/sweep/default.yaml
wandb agent recsys01/DKT_Sequential/{sweepid}
```
