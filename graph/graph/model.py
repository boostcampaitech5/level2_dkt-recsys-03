import torch
import wandb
import lightning as L
from omegaconf import DictConfig
from torch_geometric.nn.models import LightGCN
from .preprocess import load_data, indexing_data
from sklearn.metrics import accuracy_score, roc_auc_score

class LightGCNNet(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.data = load_data(self.config.paths.data_path)
        self.n_nodes = len(indexing_data(data=self.data))
        self.embedding_dim = self.config.model.emb_dim
        self.num_layers = self.config.model.n_layers
        self.model = LightGCN(num_nodes=self.n_nodes, 
                              embedding_dim=self.embedding_dim, 
                              num_layers=self.num_layers)
        
        wandb.log({"num_nodes" : self.n_nodes, "embedding_dim" : self.embedding_dim, "num_layers" : self.num_layers})
        self.training_step_outputs = []
    
    def forward(self, edge_index):
        pred = self.model.predict_link(edge_index=edge_index, prob=True)
        return pred

    def training_step(self, batch, batch_idx):
        edge_index, label = batch
        edge_index = edge_index.T
        pred = self(edge_index)

        loss = self.model.link_pred_loss(pred, label)

        label = label.cpu().numpy()
        pred = pred.detach().cpu().numpy()

        acc = accuracy_score(y_true=label, y_pred=(pred > 0.5))
        auc = roc_auc_score(y_true=label, y_score=pred)
        training_metrics = {"tr_loss" : loss, "tr_auc" : torch.tensor(auc), "tr_acc" : torch.tensor(acc)}
        self.training_step_outputs.append(training_metrics)

        return {'auc':auc, 'acc':acc, 'loss':loss}
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['tr_loss'] for x in self.training_step_outputs]).mean()
        avg_auc = torch.stack([x['tr_auc'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x['tr_acc'] for x in self.training_step_outputs]).mean()

        wandb.log({"tr_loss" : avg_loss, "tr_auc" : avg_auc, "tr_acc" : avg_acc})

        self.training_step_outputs.clear()
            
    def predict_step(self, batch, batch_idx):
        edge_index, _ = batch
        edge_index = edge_index.T
        pred = self.model.predict_link(edge_index=edge_index, prob=True)
        pred = pred.detach().cpu().numpy()
        return pred

    def configure_optimizers(self):
        if self.config.trainer.optimizer == "adam":
            optimizer = torch.optim.Adam(params = self.model.parameters(), 
                                         lr = self.config.trainer.lr)
        return [optimizer]