import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from .metrics import get_metric
from .utils import get_logger, logging_conf


logger = get_logger(logging_conf)


class ModelBase(pl.LightningModule):
    def __init__(self,
        hidden_dim: int = 64,
        n_layers: int=2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913
    ):
        super(ModelBase, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags

        # Embedding
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)

        # Concat embedding projection
        self.comb_proj = nn.Linear(intd * 4, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs= []

    def forward(self, test, question, tag, correct, mask, interaction):
        batch_size = interaction.size(0)

        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())
        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            dim=2,
        )
        X = self.comb_proj(embed)
        return X, batch_size

    # Compute loss using Binary Cross Entropy loss
    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor):
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss_val = loss(preds, targets.float())

        # using only last sequence
        loss_val = loss_val[:, -1]
        loss_val = torch.mean(loss_val)
        return loss_val

    # Set optimizer, scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        output = self(**batch) # predict
        target = batch["correct"]
        loss = self.compute_loss(output, target) # loss

        pred = F.sigmoid(output[:, -1])
        target = target[:, -1]

        auc, acc = get_metric(targets=target, preds=pred)
        metrics = {"tr_loss" : loss, "tr_auc" : torch.tensor(auc), "tr_acc" : torch.tensor(acc)}
        self.training_step_outputs.append(metrics)
        
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['tr_loss'] for x in self.training_step_outputs]).mean()
        avg_auc = torch.stack([x['tr_auc'] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x['tr_acc'] for x in self.training_step_outputs]).mean()

        logger.info(f"[Train] avg_loss: {avg_loss}, avg_auc: {avg_auc}, avg_acc: {avg_acc}")

        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        output = self(**batch) # predict
        target = batch["correct"]
        loss = self.compute_loss(output, target) # loss

        pred = F.sigmoid(output[:, -1])
        target = target[:, -1]

        auc, acc = get_metric(targets=target, preds=pred)
        metrics = {"val_loss" : loss, "val_auc" : torch.tensor(auc), "val_acc" : torch.tensor(acc)}
        self.validation_step_outputs.append(metrics)

        return metrics

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_auc = torch.stack([x['val_auc'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.validation_step_outputs]).mean()

        logger.info(f"[Valid] avg_loss: {avg_loss}, avg_auc: {avg_auc}, avg_acc: {avg_acc}")

        self.validation_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(**batch) # predict
        pred = F.sigmoid(output[:, -1])
        pred = pred.cpu().detach().numpy()
        return pred


class LSTM(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs,
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags,
        )
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out
