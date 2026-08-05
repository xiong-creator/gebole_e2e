"""模型：冻结的 DINOv2 编码器 + 可训练 Attention Probe → sigmoid 输出速度百分比

设计:
  1. Backbone: 冻结的 DINOv2 ViT，`forward_features` 输出 CLS + patch tokens (B, 1+N, D)
  2. Attention Probe: 一个小的可训练模块
       - 距离桶 embedding → 生成一个 learnable query
       - 该 query 与冻结 tokens 做 multi-head cross-attention，池化得到条件特征
       - MLP head → sigmoid 输出百分比

DINOv2 patch=14，输入边长需要是 14 的倍数。
当前配置使用接近 512 的固定尺寸，因此 patch token 数会比 224 输入更多。
"""
from pathlib import Path

import torch
import torch.nn as nn

from config import BACKBONE, DISTANCE_BUCKETS


_DINOV2_FEAT_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vitg14_reg": 1536,
}


class FrozenDinoV2(nn.Module):
    """冻结的 DINOv2 编码器，输出 (B, 1+N, D) 的 tokens (CLS 在前)"""

    def __init__(self, name: str):
        super().__init__()
        assert name in _DINOV2_FEAT_DIM, f"不支持的 DINOv2 变体: {name}"
        local_repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
        if local_repo.exists():
            self.model = torch.hub.load(str(local_repo), name, source="local")
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.feat_dim = _DINOV2_FEAT_DIM[name]

        # 冻结所有参数
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        # 无论外层怎么调，backbone 始终保持 eval
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        # DINOv2 提供 forward_features(x) -> dict:
        #   x_norm_clstoken:    (B, D)
        #   x_norm_patchtokens: (B, N, D)
        #   (reg 版本还会有 x_norm_regtokens)
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"].unsqueeze(1)      # (B, 1, D)
        patches = out["x_norm_patchtokens"]            # (B, N, D)
        tokens = torch.cat([cls, patches], dim=1)      # (B, 1+N, D)
        return tokens


class AttentionProbe(nn.Module):
    """可训练的小型 attention probe:
       - bucket embedding → query (B, 1, D)
       - tokens (B, T, D) 作 key/value
       - MultiHeadAttention → 聚合 (B, 1, D)
       - MLP → 1 维 logit → sigmoid
    """

    def __init__(
        self,
        feat_dim: int,
        num_buckets: int,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 距离桶 embedding，维度直接对齐 feat_dim，方便作为 query 参与 attention
        self.bucket_embed = nn.Embedding(num_buckets, feat_dim)
        # 也给 query 一个可学习的基向量，避免完全依赖距离
        self.query_base = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.query_base, std=0.02)

        self.token_norm = nn.LayerNorm(feat_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "q_norm": nn.LayerNorm(feat_dim),
                        "kv_norm": nn.LayerNorm(feat_dim),
                        "attn": nn.MultiheadAttention(
                            feat_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "ffn_norm": nn.LayerNorm(feat_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(feat_dim, int(feat_dim * mlp_ratio)),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(int(feat_dim * mlp_ratio), feat_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )

        self.out_norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1),
        )

    def forward(self, tokens, bucket_idx):
        """
        tokens:     (B, T, D)  来自冻结 DINOv2
        bucket_idx: (B,)       long
        return:     pct (B,) 0~1
        """
        b = tokens.shape[0]
        bucket_emb = self.bucket_embed(bucket_idx).unsqueeze(1)      # (B, 1, D)
        query = self.query_base.expand(b, -1, -1) + bucket_emb       # (B, 1, D)

        kv = self.token_norm(tokens)                                 # (B, T, D)

        for layer in self.layers:
            q_norm = layer["q_norm"](query)
            kv_norm = layer["kv_norm"](kv)
            attn_out, _ = layer["attn"](q_norm, kv_norm, kv_norm, need_weights=False)
            query = query + attn_out                                 # residual
            query = query + layer["ffn"](layer["ffn_norm"](query))   # residual FFN

        feat = self.out_norm(query.squeeze(1))                       # (B, D)
        logit = self.head(feat).squeeze(-1)                          # (B,)
        pct = torch.sigmoid(logit)
        return pct


class EyeSpeedNet(nn.Module):
    """冻结 DINOv2 + 可训练 Attention Probe。

    forward(image, bucket_idx) -> pct in (0,1)
    """

    def __init__(
        self,
        backbone: str = BACKBONE,
        num_buckets: int = len(DISTANCE_BUCKETS),
        probe_heads: int = 8,
        probe_layers: int = 2,
        probe_mlp_ratio: float = 2.0,
        probe_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.encoder = FrozenDinoV2(backbone)
        self.probe = AttentionProbe(
            feat_dim=self.encoder.feat_dim,
            num_buckets=num_buckets,
            num_heads=probe_heads,
            num_layers=probe_layers,
            mlp_ratio=probe_mlp_ratio,
            dropout=probe_dropout,
        )

    def forward(self, image, bucket_idx):
        with torch.no_grad():
            tokens = self.encoder(image)          # (B, T, D)
        return self.probe(tokens, bucket_idx)     # (B,)

    def trainable_parameters(self):
        return [p for p in self.probe.parameters() if p.requires_grad]
