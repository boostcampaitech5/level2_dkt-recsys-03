import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed number")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument("--data_path", default="/opt/ml/input/data/", type=str, help="csvdata path")
    parser.add_argument("--asset_path", default="asset/", type=str, help="asset data path")
    parser.add_argument("--output_path", default="outputs/", type=str, help="output directory")

    parser.add_argument("--train_file", default="train_data.csv", type=str, help="train file name")
    parser.add_argument("--test_file", default="test_data.csv", type=str, help="test file name")

    parser.add_argument("--epoch", default=10, type=int, help="epochs to run")
    parser.add_argument("--batch_size", default=64, type=int, help="mini batch szie")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")

    parser.add_argument("--max_seq_len", default=20, type=int, help="max sequence of length")

    args = parser.parse_args()

    return args