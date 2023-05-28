# Graph approach

## Structure
```
│
├── configs                   <- Hydra configs
│   ├── data                     <- Data configs
│   ├── model.test               <- Model pytest configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── sweep                    <- Sweep configs
│   ├── trainer                  <- Trainer configs
│   ├── wandb                    <- W&B configs
│   │
│   ├──config.yaml               <- Main config
│   └──test.yaml                 <- Pytest config
│
├── graph                     <- Source code
│   ├── models                      <- Graph model scripts
│   │
│   ├── dataloader.py               <- Dataloader scripts
│   ├── preprocess.py               <- Data preprocessing scripts
│   ├── trainer.py                  <- Trainer scripts
│   └── utils.py                    <- Utility scripts
│
├── main.py                  <- Run training and inference
├── requirements.txt         <- File for installing python dependencies
└── README.md
```


## Setup
```bash
cd ~/level2_dkt-recsys-03/graph
conda init
(base) . ~/.bashrc
(base) conda create -n graph python=3.10 -y
(base) conda activate graph
(graph) pip install -r requirements.txt
```

## How to run
```bash
python main.py
```

## Sweep
```bash
wandb sweep {configs/sweep/sweep.yaml}
wandb agent {entity}/{project}/{sweepid}
```
