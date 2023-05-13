import lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from .preprocess import load_data, indexing_data, process_data


class GraphDataset(Dataset):
    def __init__(self, data):
        self.edge_list = data['edge']
        self.label_list = data['label']

    def __getitem__(self, index):
        edge = self.edge_list[:,index]
        label = self.label_list[index]
        return edge, label

    def __len__(self):
        return len(self.label_list)

class GraphDataModule(L.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.train_data = None
        self.test_data = None
        
    def prepare_data(self):
        print("+++++++preparing data++++++++")
        # 데이터를 부르고 train&test concat하고 중복 처리
        self.data = load_data(self.config.paths.data_path)
        # len을 구해 노드의 개수로 활용
        self.id2index:dict = indexing_data(self.data)
        self.data = process_data(data=self.data, id2index=self.id2index)

    # train vs test split
    def setup(self, stage):
        print("+++++++setting up data++++++++")
        if stage == 'fit' or stage is None:
            tr_idx = [self.data['label']!=-1]
            self.train_data = {
                'edge' : torch.stack([self.data['edge'][0][tr_idx],
                                      self.data['edge'][1][tr_idx]]),
                'label' : self.data['label'][tr_idx]
            }

        if stage == 'predict' or stage is None:
            te_idx = [self.data['label']==-1]
            self.test_data = {
                'edge' : torch.stack([self.data['edge'][0][te_idx],
                                      self.data['edge'][1][te_idx]]),
                'label' : self.data['label'][te_idx]
            }

    def train_dataloader(self):
        trainset = GraphDataset(self.train_data)
        train_loader = DataLoader(trainset, 
                            num_workers = self.config.data.num_workers, 
                            batch_size=self.config.data.train_batch_size)
        return train_loader
        
    def predict_dataloader(self):
        testset = GraphDataset(self.test_data)
        pred_loader = DataLoader(testset, 
                            num_workers = self.config.data.num_workers, 
                            batch_size = self.config.data.test_batch_size)
        return pred_loader

