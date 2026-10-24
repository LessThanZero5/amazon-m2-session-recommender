from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def aggregate_embeddings(
    embeddings,
    embedding_num_ids,
    n_items=None,
    output_path=None,
    dtype=np.float16,
    chunk_size=100_000
):
    embeddings = np.asarray(
        embeddings
    )

    embedding_num_ids = np.asarray(
        embedding_num_ids,
        dtype=np.int64
    )

    if (
        embeddings.shape[0]
        != embedding_num_ids.shape[0]
    ):
        raise ValueError(
            'Embeddings and IDs '
            'must have the same number of rows.'
        )

    if n_items is None:
        n_items = int(
            embedding_num_ids.max()
        )

    embedding_dim = (
        embeddings.shape[1]
    )

    sums = np.zeros(
        (
            n_items + 1,
            embedding_dim
        ),
        dtype=np.float32
    )

    counts = np.zeros(
        n_items + 1,
        dtype=np.int32
    )

    for start in tqdm(
        range(
            0,
            len(embeddings),
            chunk_size
        ),
        desc='Aggregating embeddings'
    ):
        end = min(
            start + chunk_size,
            len(embeddings)
        )

        chunk_embeddings = (
            embeddings[
                start:end
            ]
            .astype(
                np.float32,
                copy=False
            )
        )

        chunk_ids = (
            embedding_num_ids[
                start:end
            ]
        )

        np.add.at(
            sums,
            chunk_ids,
            chunk_embeddings
        )

        np.add.at(
            counts,
            chunk_ids,
            1
        )

    valid_ids = (
        counts > 0
    )

    global_embeddings = np.zeros(
        (
            n_items + 1,
            embedding_dim
        ),
        dtype=np.float32
    )

    global_embeddings[
        valid_ids
    ] = (
        sums[
            valid_ids
        ]
        / counts[
            valid_ids,
            None
        ]
    )

    global_embeddings[0] = 0.0

    global_embeddings = (
        global_embeddings
        .astype(dtype)
    )

    if output_path is not None:
        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            output_path,
            global_embeddings
        )

    return (
        global_embeddings,
        counts
    )


def load_global_embeddings(
    path,
    mmap_mode='r'
):
    return np.load(
        path,
        mmap_mode=mmap_mode
    )