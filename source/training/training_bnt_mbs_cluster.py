"""
Training Class for BNT with Cluster Supervision
================================================

"""

from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging

from source.models.BNT.BNT_MBS_Cluster import (
    BrainNetworkTransformer_MBS_Cluster,
    compute_cluster_supervision_loss
)


class TrainBNT_MBS_Cluster:
    """Training for BNT with cluster-guided supervision."""

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        # Cluster supervision
        self.aux_weight = cfg.training.get('aux_weight', 0.03)
        self.logger.info(f'Cluster-guided supervision enabled')
        self.logger.info(f'Auxiliary loss weight: {self.aux_weight}')

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]
        
        self.train_aux_loss = TotalMeter()
        self.train_main_loss = TotalMeter()

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss, 
                      self.train_aux_loss, self.train_main_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()

            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)

            # Forward with supervision
            predict, supervision_data = self.model(
                time_series, node_feature, return_supervised=True
            )

            # Main loss
            loss_main = self.loss_fn(predict, label)

            # Cluster supervision loss
            if self.aux_weight > 0:
                loss_aux = compute_cluster_supervision_loss_kl(supervision_data)
            else:
                loss_aux = torch.tensor(0.0).cuda()

            # Total loss
            loss = loss_main + self.aux_weight * loss_aux

            # Track losses
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            self.train_main_loss.update_with_weight(loss_main.item(), label.shape[0])
            self.train_aux_loss.update_with_weight(loss_aux.item(), label.shape[0])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accuracy
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            
            output = self.model(time_series, node_feature, return_supervised=False)

            label = label.float()
            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(loss.item(), label.shape[0])
            
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(labels, result, average='micro')

        report = classification_report(labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def generate_save_learnable_matrix(self):
        learable_matrixs = []
        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            
            try:
                _, learable_matrix, _ = self.model(time_series, node_feature)
                learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            except:
                self.logger.warning("Model doesn't return learnable matrix, skipping...")
                return
            
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", 
                {'matrix': np.vstack(learable_matrixs), "label": np.array(labels)}, 
                allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy", results, allow_pickle=True)
        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            
            val_result = self.test_per_epoch(
                self.val_dataloader, self.val_loss, self.val_accuracy
            )

            test_result = self.test_per_epoch(
                self.test_dataloader, self.test_loss, self.test_accuracy
            )

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'(Main:{self.train_main_loss.avg: .3f}',
                f'Cluster:{self.train_aux_loss.avg: .3f})',
                f'Train Acc:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Acc:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Train Main Loss": self.train_main_loss.avg,
                "Train Cluster Loss": self.train_aux_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Val AUC": val_result[0],
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
            })

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Main Loss": self.train_main_loss.avg,
                "Train Cluster Loss": self.train_aux_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'Test Sensitivity': test_result[-1],
                'Test Specificity': test_result[-2],
                'micro F1': test_result[-4],
                'micro recall': test_result[-5],
                'micro precision': test_result[-6],
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
            })

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        
        self.save_result(training_process)