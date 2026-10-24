import numpy as np
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SessionDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        sequence_column='remainder__prev_items_list',
        locale_columns=None
    ):
        self.X = (
            X
            .reset_index(drop=True)
            .copy()
        )

        self.y = (
            y
            .reset_index(drop=True)
            .copy()
        )

        self.sequence_column = (
            sequence_column
        )

        if (
            sequence_column
            not in self.X.columns
        ):
            raise ValueError(
                f'Sequence column not found: '
                f'{sequence_column}'
            )

        if locale_columns is None:
            locale_columns = [
                column
                for column
                in self.X.columns
                if 'locale_' in
                column.lower()
            ]

        if len(locale_columns) == 0:
            raise ValueError(
                'Locale columns not found.'
            )

        self.locale_columns = (
            locale_columns
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sequence = torch.tensor(
            self.X.at[
                index,
                self.sequence_column
            ],
            dtype=torch.long
        )

        locale = torch.tensor(
            self.X.loc[
                index,
                self.locale_columns
            ].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32
        )

        target = torch.tensor(
            int(
                self.y.iloc[index]
            ),
            dtype=torch.long
        )

        return (
            sequence,
            locale,
            target
        )


def collate_fn(
    batch,
    padding_value=0,
    max_seq_len=None
):
    sequences, locales, targets = (
        zip(*batch)
    )

    if max_seq_len is not None:
        sequences = [
            sequence[
                -max_seq_len:
            ]
            for sequence
            in sequences
        ]

    sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=padding_value
    )

    locales = torch.stack(
        locales,
        dim=0
    )

    targets = torch.stack(
        targets,
        dim=0
    )

    return (
        sequences,
        locales,
        targets
    )