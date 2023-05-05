import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument("--path", default="/opt/ml/input/data/", type=str, help="data path",
    )

    args = parser.parse_args()

    return args