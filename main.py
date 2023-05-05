import torch

from sequential.dataloader import Preprocess
from sequential.args import parse_args
from sequential.utils import set_seeds

def main():
    args = parse_args()
    set_seeds(args.seed)
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = Preprocess(args)

if __name__=="__main__":
    main()