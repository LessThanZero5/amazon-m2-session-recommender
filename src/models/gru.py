import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import (
    pack_padded_sequence
)


class SemanticGRURec(nn.Module):
    def __init__(
        self,
        semantic_dim=1024,
        d_model=128,
        locale_dim=6,
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

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            dropout=(
                dropout
                if n_layers > 1
                else 0.0
            ),
            batch_first=True
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
        x = (
            self.semantic_head(
                semantic_sequences
            )
            + self.locale_head(
                locales
            ).unsqueeze(1)
        )

        lengths = (
            (~padding_mask)
            .sum(dim=1)
            .clamp(min=1)
            .cpu()
        )

        packed = (
            pack_padded_sequence(
                x,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
        )

        _, hidden = self.gru(
            packed
        )

        session_vectors = (
            hidden[-1]
        )

        session_vectors = (
            self.output_norm(
                session_vectors
            )
        )

        return F.normalize(
            session_vectors,
            dim=-1
        )