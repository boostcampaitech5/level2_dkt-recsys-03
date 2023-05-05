import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        print(os.listdir(self.args.path))