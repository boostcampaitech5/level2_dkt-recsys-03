import torch
import wandb
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

from .metrics import get_metric
from .utils import get_logger, logging_conf
from .scheduler import get_scheduler
from .optimizer import get_optimizer


logger = get_logger(logging_conf)


class ModelBase(pl.LightningModule):
    def __init__(self, config):
        super(ModelBase, self).__init__()

        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.n_layers = self.config.model.n_layers
        self.n_tests = self.config.model.n_tests
        self.n_questions = self.config.model.n_questions
        self.n_tags = self.config.model.n_tags

        # Embedding
        hd, intd = self.hidden_dim, self.hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd)
        self.embedding_test = nn.Embedding(self.n_tests + 1, intd)
        self.embedding_question = nn.Embedding(self.n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(self.n_tags + 1, intd)

        # Concat embedding projection
        self.comb_proj = nn.Linear(intd * 4, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.tr_result = []
        self.val_result = []

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
        optimizer = get_optimizer(param=self.parameters(), config=self.config)
        scheduler = get_scheduler(optimizer=optimizer, config=self.config)

        if self.config.trainer.scheduler == "plateau":
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_auc",
                    "name": "seq_lr_scheduler",
                }
            ]
        else:
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "seq_lr_scheduler",
                }
            ]

    def training_step(self, batch, batch_idx):
        output = self(**batch)  # predict
        target = batch["correct"]
        loss = self.compute_loss(output, target)  # loss

        pred = F.sigmoid(output[:, -1])
        target = target[:, -1]

        auc, acc = get_metric(targets=target, preds=pred)
        metrics = {
            "tr_loss": loss,
            "tr_auc": torch.tensor(auc),
            "tr_acc": torch.tensor(acc),
        }
        self.training_step_outputs.append(metrics)

        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(
            [x["tr_loss"] for x in self.training_step_outputs]
        ).mean()
        avg_auc = torch.stack([x["tr_auc"] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x["tr_acc"] for x in self.training_step_outputs]).mean()

        logger.info(
            f"[Train] avg_loss: {avg_loss}, avg_auc: {avg_auc}, avg_acc: {avg_acc}"
        )
        wandb.log({"tr_loss": avg_loss, "tr_auc": avg_auc, "tr_acc": avg_acc})

        self.tr_result.append({"tr_avg_auc": avg_auc, "tr_avg_acc": avg_acc})
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        output = self(**batch)  # predict
        target = batch["correct"]
        loss = self.compute_loss(output, target)  # loss

        pred = F.sigmoid(output[:, -1])
        target = target[:, -1]

        auc, acc = get_metric(targets=target, preds=pred)
        metrics = {
            "val_loss": loss,
            "val_auc": torch.tensor(auc),
            "val_acc": torch.tensor(acc),
        }
        self.validation_step_outputs.append(metrics)

        return metrics

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_auc = torch.stack(
            [x["val_auc"] for x in self.validation_step_outputs]
        ).mean()
        avg_acc = torch.stack(
            [x["val_acc"] for x in self.validation_step_outputs]
        ).mean()

        logger.info(
            f"[Valid] avg_loss: {avg_loss}, avg_auc: {avg_auc}, avg_acc: {avg_acc}"
        )
        wandb.log({"val_loss": avg_loss, "val_auc": avg_auc, "val_acc": avg_acc})
        self.log("val_auc", avg_auc)

        self.val_result.append({"val_avg_auc": avg_auc, "val_avg_acc": avg_acc})
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(**batch)  # predict
        pred = F.sigmoid(output[:, -1])
        pred = pred.cpu().detach().numpy()
        return pred


class LSTM(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(
            test=test,
            question=question,
            tag=tag,
            correct=correct,
            mask=mask,
            interaction=interaction,
        )
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.n_heads = self.config.model.n_heads
        self.drop_out = self.config.model.drop_out
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.bert_config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.bert_config)

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(
            test=test,
            question=question,
            tag=tag,
            correct=correct,
            mask=mask,
            interaction=interaction,
        )

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        # Adding Attention
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.n_heads = self.config.model.n_heads
        self.drop_out = self.config.model.drop_out
        self.max_seq_len = self.config.data.max_seq_len
        # Bert config
        self.bert_config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=self.max_seq_len,
        )
        self.encoder = BertModel(self.bert_config)  # Transformer Encoder

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(
            test=test,
            question=question,
            tag=tag,
            correct=correct,
            mask=mask,
            interaction=interaction,
        )

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class EncoderEmbedding(nn.Module):
    def __init__(self, n_questions, n_tags, n_dims, seq_len):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.embedding_question = nn.Embedding(n_questions, n_dims)
        self.embedding_tag = nn.Embedding(n_tags, n_dims)
        self.embedding_pos = nn.Embedding(seq_len, n_dims)

    def forward(self, questions, tags):
        embed_quest = self.embedding_question(questions)
        embed_tag = self.embedding_tag(tags)

        seq = torch.arange(self.seq_len).unsqueeze(0)
        embed_pos = self.embedding_pos(seq)
        return embed_quest + embed_tag + embed_pos


class DecoderEmbedding(nn.Module):
    def __init__(self, n_responses, n_dims, seq_len):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.embedding_response = nn.Embedding(n_responses, n_dims)
        self.embedding_solving_time = nn.Linear(1, n_dims, bias=False)
        self.embedding_pos = nn.Embedding(seq_len, n_dims)

    def forward(self, responses, solving_times):
        embed_response = self.embedding_response(responses)
        embed_solving_time = self.solving_time_embed(solving_times)

        seq = torch.arange(self.seq_len).unsqueeze(0)
        embed_pos = self.embedding_pos(seq)
        return embed_response + embed_solving_time + embed_pos


class SAINTPLUS(pl.LightningModule):
    def __init__(self, config):
        super(SAINTPLUS, self).__init__()

        self.config = config
        self.seq_len = self.config.data.max_seq_len
        self.n_questions = self.config.model.n_questions
        self.n_tags = self.config.model.n_tags

        self.n_heads = self.config.model.n_heads
        self.n_encoder = self.config.model.n_decoder
        self.n_decoder = self.config.model.n_decoder
        self.embed_dims = self.config.model.embed_dims
        self.ffn_dim = self.embed_dims * 4

        # Embedding
        self.encoder_embedding = EncoderEmbedding(
            n_questions=self.n_questions,
            n_tags=self.n_tags,
            n_dims=self.embed_dims,
            seq_len=self.seq_len,
        )

        self.decoder_embedding = DecoderEmbedding(
            n_responses=3, n_dims=self.embed_dims, seq_len=self.seq_len
        )

        # mask
        self.mask = torch.from_numpy(
            np.triu(np.ones((self.seq_len, self.seq_len)), k=1).astype("bool")
        )

        # transformer
        self.transformer = nn.Transformer(
            d_model=self.embed_dims,
            nhead=self.n_heads,
            num_encoder_layers=self.n_decoder,
            num_decoder_layers=self.n_decoder,
            dim_feedforward=self.ffn_dim,
        )

        # fully connected layer
        self.fc = nn.Linear(self.ffn_dim, 1)

        # logs
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(
        self, question, tag, correct, solving_time, **kwargs
    ):  # kwargs is not used
        enc = self.encoder_embedding(questions=question, tags=tag)
        dec = self.decoder_embedding(responses=correct, solving_time=solving_time)

        decoder_output = self.transformer(
            src=enc,  # encoder seq
            tgt=dec,  # decoder seq
            src_mask=self.mask,
            tgt_mask=self.mask,
            memory_mask=self.mask,
        )

        # fully connected layer
        out = self.fc(decoder_output)
        return out.squeeze()

    # Compute loss
    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor):
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss_val = loss(preds, targets.float())

        # using only last seq
        loss_val = loss_val[:, -1]
        loss_val = torch.mean(loss_val)
        return loss_val

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_ids):
        output = self(**batch)  # predict
        target = batch["correct"]
        loss = self.compute_loss(output, target)  # loss

        pred = F.sigmoid(output[:, -1])
        target = target[:, -1]

        auc, acc = get_metric(targets=target, preds=pred)
        metrics = {
            "tr_loss": loss,
            "tr_auc": torch.tensor(auc),
            "tr_acc": torch.tensor(acc),
        }
        self.training_step_outputs.append(metrics)

        return loss

    def training_epoch_end(self, training_ouput):
        avg_loss = torch.stack(
            [x["tr_loss"] for x in self.training_step_outputs]
        ).mean()
        avg_auc = torch.stack([x["tr_auc"] for x in self.training_step_outputs]).mean()
        avg_acc = torch.stack([x["tr_acc"] for x in self.training_step_outputs]).mean()

        logger.info(
            f"[Train] avg_loss: {avg_loss}, avg_auc: {avg_auc}, avg_acc: {avg_acc}"
        )
        wandb.log({"tr_loss": avg_loss, "tr_auc": avg_auc, "tr_acc": avg_acc})

        self.tr_result.append({"tr_avg_auc": avg_auc, "tr_avg_acc": avg_acc})
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        output = self(**batch)  # predict
        target = batch["correct"]
        loss = self.compute_loss(output, target)  # loss

        pred = F.sigmoid(output[:, -1])
        target = target[:, -1]

        auc, acc = get_metric(targets=target, preds=pred)
        metrics = {
            "val_loss": loss,
            "val_auc": torch.tensor(auc),
            "val_acc": torch.tensor(acc),
        }
        self.validation_step_outputs.append(metrics)

        return metrics

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        avg_auc = torch.stack(
            [x["val_auc"] for x in self.validation_step_outputs]
        ).mean()
        avg_acc = torch.stack(
            [x["val_acc"] for x in self.validation_step_outputs]
        ).mean()

        logger.info(
            f"[Valid] avg_loss: {avg_loss}, avg_auc: {avg_auc}, avg_acc: {avg_acc}"
        )
        wandb.log({"val_loss": avg_loss, "val_auc": avg_auc, "val_acc": avg_acc})
        self.log("val_auc", avg_auc)

        self.val_result.append({"val_avg_auc": avg_auc, "val_avg_acc": avg_acc})
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(**batch)  # predict
        pred = F.sigmoid(output[:, -1])
        pred = pred.cpu().detach().numpy()
        return pred
