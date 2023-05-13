import lightning as L
import torch
from torch_geometric.nn.models import LightGCN
from .preprocess import *
from sklearn.metrics import accuracy_score, roc_auc_score


class LightGCNNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Define Graph-based model
        self.data = load_data("/opt/ml/input/data")
        self.n_nodes = len(indexing_data(data=self.data))
        self.embedding_dim = 64
        self.num_layers = 2
        self.model = LightGCN(num_nodes=self.n_nodes, embedding_dim=self.embedding_dim, num_layers=self.num_layers)
    
    def forward(self, edge_index):
        print("+++++++forward++++++++")
        pred = self.model.predict_link(edge_index=edge_index, prob=True)
        return pred

    def training_step(self, batch, batch_idx):
        print("+++++++training step++++++++")
        edge_index, label = batch
        edge_index = edge_index.T
        pred = self(edge_index)

        print("+++++++calculating loss++++++++++")
        loss = self.model.link_pred_loss(pred, label)

        label = label.cpu().numpy()
        pred = pred.detach().cpu().numpy()

        acc = accuracy_score(y_true=label, y_pred=(pred > 0.5))
        auc = roc_auc_score(y_true=label, y_score=pred)

        return {'auc':auc, 'acc':acc, 'loss':loss}
            
    def predict_step(self, batch, batch_idx):
        print("+++++++predict step++++++++")
        edge_index, _ = batch
        edge_index = edge_index.T
        pred = self.model.predict_link(edge_index=edge_index, prob=True)
        pred = pred.detach().cpu().numpy()
        return pred

    def configure_optimizers(self):
        print("+++++++config opt++++++++")
        return torch.optim.Adam(params=self.model.parameters(), lr=0.001)