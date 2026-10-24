import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSASRec(nn.Module):
    def __init__(
        self,
        semantic_dim=1024,
        d_model=128,
        locale_dim=6,
        max_seq_len=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.semantic_head = nn.Sequential(
            nn.Linear(
                semantic_dim,
                d_model
            ),
            nn.GELU(),
            nn.LayerNorm(
                d_model
            )
        )

        self.locale_head = nn.Linear(
            locale_dim,
            d_model
        )

        self.position_embedding = (
            nn.Embedding(
                max_seq_len,
                d_model
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=(
                    d_model * 4
                ),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers
            )
        )

        self.output_norm = (
            nn.LayerNorm(
                d_model
            )
        )

    def project_items(
        self,
        semantic_vectors
    ):
        vectors = (
            self.semantic_head(
                semantic_vectors
            )
        )

        return F.normalize(
            vectors,
            dim=-1
        )

    def forward(
        self,
        semantic_sequences,
        locales,
        padding_mask
    ):
        batch_size, seq_len, _ = (
            semantic_sequences.shape
        )

        x = self.semantic_head(
            semantic_sequences
        )

        positions = torch.arange(
            seq_len,
            device=x.device
        )

        positions = (
            positions
            .unsqueeze(0)
            .expand(
                batch_size,
                seq_len
            )
        )

        x = (
            x
            + self.position_embedding(
                positions
            )
            + self.locale_head(
                locales
            ).unsqueeze(1)
        )

        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0
        )

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=(
                padding_mask
            )
        )

        lengths = (
            (~padding_mask)
            .sum(dim=1)
        )

        last_indexes = (
            lengths - 1
        )

        batch_indexes = (
            torch.arange(
                batch_size,
                device=x.device
            )
        )

        session_vectors = x[
            batch_indexes,
            last_indexes
        ]

        session_vectors = (
            self.output_norm(
                session_vectors
            )
        )

        return F.normalize(
            session_vectors,
            dim=-1
        )
