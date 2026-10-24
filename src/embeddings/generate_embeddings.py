from pathlib import Path
import json

import numpy as np
from tqdm.auto import tqdm


def generate_embeddings(
    client,
    texts,
    model='voyage-4-large',
    embedding_dim=1024,
    batch_size=128,
    output_path=None,
    dtype=np.float16,
    input_type='document'
):
    total = len(texts)

    if output_path is None:
        embeddings = np.empty(
            (
                total,
                embedding_dim
            ),
            dtype=dtype
        )

    else:
        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        embeddings = np.lib.format.open_memmap(
            output_path,
            mode='w+',
            dtype=dtype,
            shape=(
                total,
                embedding_dim
            )
        )

    for start in tqdm(
        range(
            0,
            total,
            batch_size
        ),
        desc='Generating embeddings'
    ):
        end = min(
            start + batch_size,
            total
        )

        batch_texts = texts[
            start:end
        ]

        response = client.embed(
            batch_texts,
            model=model,
            input_type=input_type,
            output_dimension=embedding_dim
        )

        batch_embeddings = np.asarray(
            response.embeddings,
            dtype=dtype
        )

        embeddings[
            start:end
        ] = batch_embeddings

        if isinstance(
            embeddings,
            np.memmap
        ):
            embeddings.flush()

    return embeddings


def save_embedding_config(
    output_path,
    model,
    embedding_dim,
    batch_size,
    dtype,
    rows
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    config = {
        'model': model,
        'embedding_dim': embedding_dim,
        'batch_size': batch_size,
        'dtype': str(dtype),
        'rows': rows
    }

    with open(
        output_path,
        'w',
        encoding='utf-8'
    ) as file:
        json.dump(
            config,
            file,
            indent=2
        )