"""
BNT with Multi-Block Supervision using OCRead Cluster Centers
==============================================================

Novel approach: Supervise intermediate layers with orthogonal cluster centers.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ptdec import DEC
from typing import List, Dict, Optional, Tuple
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel


class TransPoolingEncoder(nn.Module):
    """Same as original BNT."""

    def __init__(self, input_feature_size, input_node_num, hidden_size, 
                 output_node_num, pooling=True, orthogonal=True, 
                 freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size, 
            nhead=4,
            dim_feedforward=hidden_size,
            batch_first=True
        )

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(
                cluster_number=output_node_num, 
                hidden_dimension=input_feature_size, 
                encoder=self.encoder,
                orthogonal=orthogonal, 
                freeze_center=freeze_center, 
                project_assignment=project_assignment
            )

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer_MBS_Cluster(BaseModel):
    """
    BNT with Cluster-Guided Multi-Block Supervision.
    
    Key Innovation: Supervise layer 0 nodes with layer 1 cluster centers.
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        # Positional encoding
        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(
                torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim), 
                requires_grad=True
            )
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        # Transformer stack
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    project_assignment=config.model.project_assignment
                )
            )

        # Classification head
        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        # Supervision config
        if hasattr(config.model, 'supervised_layers'):
            self.supervised_layers = set(config.model.supervised_layers)
        else:
            self.supervised_layers = {0} if len(sizes) > 1 else set()

    def forward(
        self, 
        time_series: torch.Tensor,
        node_feature: torch.Tensor,
        return_supervised: bool = False
    ):
        bz, _, _ = node_feature.shape

        # Positional encoding
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        layer_features = {}
        cluster_centers = {}
        assignments = []

        # Pass through layers
        for i, atten in enumerate(self.attention_list):
            # Store features BEFORE processing for supervision
            if return_supervised and i in self.supervised_layers:
                layer_features[i] = node_feature.clone()
            
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)
            
            # Extract cluster centers
            if return_supervised and atten.is_pooling_enabled():
                centers = atten.dec.assignment.get_cluster_centers()
                cluster_centers[i] = centers

        # Classification
        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))
        logits = self.fc(node_feature)

        if return_supervised:
            supervision_data = {
                'features': layer_features,
                'clusters': cluster_centers,
                'assignments': assignments
            }
            return logits, supervision_data
        
        return logits

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def clustering_loss(self, assignments):
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all


def compute_cluster_supervision_loss(supervision_data: Dict) -> torch.Tensor:
    """
    Supervise intermediate nodes with cluster centers from next layer.
    """
    layer_features = supervision_data['features']
    cluster_centers = supervision_data['clusters']
    
    if not layer_features or not cluster_centers:
        return torch.tensor(0.0)
    
    total_loss = 0.0
    count = 0
    
    for layer_idx in layer_features.keys():
        # Find next cluster layer
        next_cluster_idx = None
        for cluster_idx in sorted(cluster_centers.keys()):
            if cluster_idx > layer_idx:
                next_cluster_idx = cluster_idx
                break
        
        if next_cluster_idx is None:
            continue
        
        features = layer_features[layer_idx]  # [B, N, D]
        centers = cluster_centers[next_cluster_idx]  # [K, D]
        
        loss = _node_cluster_loss(features, centers)
        total_loss += loss
        count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0)

def compute_cluster_supervision_loss_kl(
    supervision_data: dict
) -> torch.Tensor:
    """
    Compute cluster supervision loss using KL divergence.
    
  BNT's existing DEC (Deep Embedding Clustering) loss!
    
    Args:
        supervision_data: Dict with:
            - 'features': {layer_idx: node_features [B, N, D]}
            - 'clusters': {layer_idx: cluster_centers [K, D]}
    
    Returns:
        loss: Scalar KL divergence loss
    
    Key Idea:
    ---------
    BNT already uses KL divergence for clustering:
        KL(P || Q) where P is target distribution, Q is soft assignment
    
    We use the same principle for supervision:
        Layer 0 nodes should have similar cluster assignments to Layer 1
    """
    layer_features = supervision_data['features']
    cluster_centers = supervision_data['clusters']
    
    if not layer_features or not cluster_centers:
        return torch.tensor(0.0)
    
    total_loss = 0.0
    count = 0
    
    for layer_idx in layer_features.keys():
        # Find next cluster layer
        next_cluster_idx = None
        for cluster_idx in sorted(cluster_centers.keys()):
            if cluster_idx > layer_idx:
                next_cluster_idx = cluster_idx
                break
        
        if next_cluster_idx is None:
            continue
        
        features = layer_features[layer_idx]  # [B, N, D]
        centers = cluster_centers[next_cluster_idx]  # [K, D]
        
        loss = _kl_cluster_loss(features, centers)
        total_loss += loss
        count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0)


def _kl_cluster_loss(
    node_features: torch.Tensor,     # [B, N, D]
    cluster_centers: torch.Tensor    # [K, D]
) -> torch.Tensor:
    """
    KL divergence loss between node assignments and target distribution.
    
    This follows DEC's approach exactly!
    
    DEC Loss Formula:
    -----------------
    L = KL(P || Q) = Σ p_ij * log(p_ij / q_ij)
    
    where:
        q_ij = soft assignment of node i to cluster j
        p_ij = target distribution (makes assignments more confident)
    
    Args:
        node_features: [B, N, D] features from intermediate layer
        cluster_centers: [K, D] orthogonal cluster centers
    
    Returns:
        loss: Scalar KL divergence
    """
    B, N, D = node_features.shape
    K = cluster_centers.shape[0]
    
    # Flatten batch dimension
    nodes = node_features.reshape(B * N, D)  # [B*N, D]
    
    # Step 1: Compute soft assignment Q (same as DEC)
    # ================================================
    Q = _compute_soft_assignment(nodes, cluster_centers)  # [B*N, K]
    
    # Step 2: Compute target distribution P (same as DEC)
    # ====================================================
    P = _compute_target_distribution(Q)  # [B*N, K]
    
    # Step 3: KL divergence KL(P || Q)
    # =================================
    # KL(P || Q) = Σ P * log(P / Q)
    kl_loss = (P * torch.log(P / (Q + 1e-10))).sum(dim=1).mean()
    
    return kl_loss


def _compute_soft_assignment(
    nodes: torch.Tensor,              # [B*N, D]
    cluster_centers: torch.Tensor     # [K, D]
) -> torch.Tensor:
    """
    Compute soft assignment Q (Student's t-distribution).
    
    
    Formula (from DEC paper):
    -------------------------
    q_ij = (1 + ||z_i - c_j||^2)^(-1) / Σ_j (1 + ||z_i - c_j||^2)^(-1)
    
    where:
        z_i = node i features
        c_j = cluster j center
        ||·|| = Euclidean distance
    """
    # Compute pairwise distances
    # nodes: [B*N, D], centers: [K, D]
    # distances: [B*N, K]
    distances_squared = torch.cdist(
        nodes.unsqueeze(0), 
        cluster_centers.unsqueeze(0), 
        p=2
    ).squeeze(0) ** 2  # [B*N, K]
    
    # Student's t-distribution (alpha=1, as in DEC)
    q = 1.0 / (1.0 + distances_squared)  # [B*N, K]
    
    # Normalize (soft assignment probabilities)
    q = q / q.sum(dim=1, keepdim=True)  # [B*N, K]
    
    return q


def _compute_target_distribution(Q: torch.Tensor) -> torch.Tensor:
    """
    Compute target distribution P (auxiliary distribution).
    
    
    Formula (from DEC paper):
    -------------------------
    p_ij = (q_ij^2 / Σ_i q_ij) / Σ_j (q_ij^2 / Σ_i q_ij)
    
    Effect: Sharpens the soft assignment
        - Emphasizes high-confidence assignments
        - De-emphasizes low-confidence assignments
    
    Args:
        Q: [B*N, K] soft assignment
    
    Returns:
        P: [B*N, K] target distribution
    """
    # Square and normalize by cluster frequency
    weight = Q ** 2 / Q.sum(dim=0, keepdim=True)  # [B*N, K]
    
    # Normalize to get probability distribution
    P = weight / weight.sum(dim=1, keepdim=True)  # [B*N, K]
    
    return P

def _node_cluster_loss(node_features: torch.Tensor, cluster_centers: torch.Tensor) -> torch.Tensor:
    """
    Loss: nodes should be similar to cluster centers.
    """
    B, N, D = node_features.shape
    K = cluster_centers.shape[0]
    
    nodes = node_features.reshape(B * N, D)  # [B*N, D]
    
    # Similarity to clusters
    similarity = nodes @ cluster_centers.T  # [B*N, K]
    
    # Normalize
    cluster_norms = torch.norm(cluster_centers, p=2, dim=1, keepdim=True).T
    similarity = similarity / (cluster_norms + 1e-8)
    
    # Square (like DEC)
    similarity_sq = torch.pow(similarity, 2)
    
    # Soft assignment
    soft_assign = F.softmax(similarity_sq, dim=1)
    
    # Maximize weighted similarity
    weighted_sim = (soft_assign * similarity_sq).sum(dim=1)
    loss = -weighted_sim.mean()
    
    return loss
