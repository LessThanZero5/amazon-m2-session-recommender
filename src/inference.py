import numpy as np
import pandas as pd
import torch


def build_locale_vector(
    locale,
    locale_order,
    device
):
    locale = str(
        locale
    ).upper()

    if (
        locale
        not in locale_order
    ):
        raise ValueError(
            f'Unsupported locale: '
            f'{locale}'
        )

    vector = torch.zeros(
        (
            1,
            len(locale_order)
        ),
        dtype=torch.float32,
        device=device
    )

    vector[
        0,
        locale_order.index(
            locale
        )
    ] = 1.0

    return vector


def get_product_metadata(
    product_lookup,
    num_ids,
    locale=None
):
    rows = []

    num_ids = [
        int(item)
        for item
        in num_ids
    ]

    subset = product_lookup[
        product_lookup[
            'num_id'
        ].isin(num_ids)
    ]

    for num_id in num_ids:
        item_rows = subset[
            subset['num_id']
            .eq(num_id)
        ]

        selected = None

        if locale is not None:
            locale_rows = (
                item_rows[
                    item_rows[
                        'locale'
                    ]
                    .astype(str)
                    .eq(
                        str(locale)
                    )
                ]
            )

            if len(
                locale_rows
            ) > 0:
                selected = (
                    locale_rows
                    .iloc[0]
                )

        if (
            selected is None
            and len(item_rows) > 0
        ):
            selected = (
                item_rows.iloc[0]
            )

        if selected is not None:
            rows.append(
                selected.to_dict()
            )

    return pd.DataFrame(
        rows
    )


def recommend_next_items(
    model,
    session_asins,
    locale,
    asin_to_num_id,
    product_lookup,
    embedding_matrix,
    catalog_vectors,
    locale_order,
    device,
    max_seq_len=64,
    top_k=10,
    exclude_seen=True
):
    session_ids = [
        int(
            asin_to_num_id[
                asin
            ]
        )
        for asin
        in session_asins
        if asin
        in asin_to_num_id
    ]

    if len(session_ids) == 0:
        raise ValueError(
            'No valid products '
            'found in session.'
        )

    session_ids = session_ids[
        -max_seq_len:
    ]

    sequence = np.asarray(
        session_ids,
        dtype=np.int64
    )

    semantic = (
        torch.from_numpy(
            np.asarray(
                embedding_matrix[
                    sequence
                ],
                dtype=np.float32
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    locales = (
        build_locale_vector(
            locale,
            locale_order,
            device
        )
    )

    padding_mask = (
        torch.zeros(
            (
                1,
                len(session_ids)
            ),
            dtype=torch.bool,
            device=device
        )
    )

    model.eval()

    with torch.inference_mode():
        session_vector = model(
            semantic,
            locales,
            padding_mask
        )

        catalog_vectors = (
            catalog_vectors
            .to(device)
        )

        scores = (
            session_vector
            @ catalog_vectors.T
        ).squeeze(0)

        if exclude_seen:
            for item_id in session_ids:
                scores[
                    item_id - 1
                ] = -torch.inf

        (
            top_scores,
            top_positions
        ) = torch.topk(
            scores,
            k=top_k
        )

    top_ids = (
        top_positions
        .cpu()
        .numpy()
        + 1
    )

    result = (
        get_product_metadata(
            product_lookup,
            top_ids,
            locale=locale
        )
    )

    result.insert(
        0,
        'rank',
        np.arange(
            1,
            len(result) + 1
        )
    )

    result['score'] = (
        top_scores
        .cpu()
        .numpy()[
            :len(result)
        ]
    )

    return result