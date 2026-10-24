from collections import Counter

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_empty_string(df):
    df = df.copy()

    string_columns = df.select_dtypes(
        include=['object', 'string']
    ).columns

    for column in string_columns:
        mask = (
            df[column]
            .astype('string')
            .str.strip()
            .eq('')
        )

        df.loc[mask, column] = pd.NA

    return df


def parse_sessions(sessions, products):
    sessions_hist = sessions.copy()
    catalog = products.copy()

    sessions_hist['prev_items_list'] = (
        sessions_hist['prev_items']
        .astype('string')
        .str.findall(r"'([^']+)'")
    )

    history_items = {
        item
        for row in sessions_hist['prev_items_list']
        for item in row
    }

    target_items = set(
        sessions_hist['next_item']
        .dropna()
        .astype(str)
    )

    behavioral_items = (
        history_items
        | target_items
    )

    behavioral_asins = (
        catalog.loc[
            catalog['id'].astype(str).isin(
                behavioral_items
            ),
            'id'
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    items_dict = {
        asin: num_id
        for num_id, asin
        in enumerate(
            behavioral_asins,
            start=1
        )
    }

    missing_prev = (
        history_items
        - set(items_dict)
    )

    missing_targets = (
        target_items
        - set(items_dict)
    )

    if missing_prev or missing_targets:
        raise ValueError(
            f'Unknown items detected: '
            f'{len(missing_prev)} history items, '
            f'{len(missing_targets)} targets'
        )

    actual_catalog = catalog[
        catalog['id']
        .astype(str)
        .isin(items_dict)
    ].copy()

    cold_catalog = catalog[
        ~catalog['id']
        .astype(str)
        .isin(items_dict)
    ].copy()

    actual_catalog['num_id'] = (
        actual_catalog['id']
        .astype(str)
        .map(items_dict)
    )

    sessions_hist['prev_items_list'] = [
        [
            items_dict[item]
            for item in row
        ]
        for row
        in sessions_hist['prev_items_list']
    ]

    sessions_hist['next_item'] = (
        sessions_hist['next_item']
        .astype(str)
        .map(items_dict)
        .astype(np.int64)
    )

    sessions_hist = sessions_hist[
        [
            'prev_items_list',
            'next_item',
            'locale'
        ]
    ]

    return (
        sessions_hist,
        actual_catalog,
        cold_catalog,
        items_dict
    )


def split_sessions(
    sessions,
    train_ratio=0.8,
    random_state=42
):
    sessions_hist = (
        sessions
        .sample(
            frac=1,
            random_state=random_state
        )
        .reset_index(drop=True)
        .copy()
    )

    sessions_hist['session_len'] = (
        sessions_hist['prev_items_list']
        .str.len()
    )

    sessions_hist['items_list_str'] = (
        sessions_hist['prev_items_list']
        .astype(str)
    )

    sessions_dict = dict(
        Counter(
            sessions_hist[
                'items_list_str'
            ]
        )
    )

    sessions_hist['num_repeat'] = (
        sessions_hist[
            'items_list_str'
        ]
        .map(sessions_dict)
    )

    repeated = sessions_hist[
        sessions_hist['num_repeat'] > 1
    ]

    unique = sessions_hist[
        sessions_hist['num_repeat'] == 1
    ]

    train_size = round(
        len(sessions_hist)
        * train_ratio
    )

    unique_to_take = (
        train_size
        - len(repeated)
    )

    train = pd.concat(
        [
            repeated,
            unique.iloc[
                :unique_to_take
            ]
        ],
        ignore_index=True
    )

    val_test = unique.iloc[
        unique_to_take:
    ].reset_index(drop=True)

    middle = len(val_test) // 2

    val = (
        val_test
        .iloc[:middle]
        .copy()
    )

    test = (
        val_test
        .iloc[middle:]
        .copy()
    )

    assert (
        len(train)
        + len(val)
        + len(test)
        == len(sessions_hist)
    )

    return train, val, test


def encode_session_locales(
    train,
    val,
    test
):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'ohe_locale',
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore'
                ),
                ['locale']
            )
        ],
        remainder='passthrough'
    )

    preprocessor.set_output(
        transform='pandas'
    )

    train = preprocessor.fit_transform(
        train
    )

    val = preprocessor.transform(
        val
    )

    test = preprocessor.transform(
        test
    )

    drop_columns = [
        'remainder__session_len',
        'remainder__items_list_str',
        'remainder__num_repeat'
    ]

    train = train.drop(
        columns=drop_columns,
        errors='ignore'
    )

    val = val.drop(
        columns=drop_columns,
        errors='ignore'
    )

    test = test.drop(
        columns=drop_columns,
        errors='ignore'
    )

    return (
        train,
        val,
        test,
        preprocessor
    )


def process_catalog(catalog):
    catalog = catalog.copy()

    anomaly_value = (
        catalog['price']
        .max()
    )

    for locale in (
        catalog['locale']
        .dropna()
        .unique()
    ):
        locale_mask = (
            catalog['locale']
            == locale
        )

        anomaly_mask = (
            locale_mask
            & catalog['price']
            .eq(anomaly_value)
        )

        if anomaly_mask.any():
            median_price = (
                catalog.loc[
                    locale_mask
                    & ~anomaly_mask,
                    'price'
                ]
                .median()
            )

            catalog.loc[
                anomaly_mask,
                'price'
            ] = median_price

        indexes = catalog.index[
            locale_mask
        ]

        scaler = StandardScaler()

        prices = (
            catalog.loc[
                indexes,
                'price'
            ]
            .to_numpy()
            .reshape(-1, 1)
        )

        catalog.loc[
            indexes,
            'price'
        ] = (
            scaler
            .fit_transform(prices)
            .reshape(-1)
        )

    catalog_preprocessor = (
        ColumnTransformer(
            transformers=[
                (
                    'Catalog_OHE',
                    OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False
                    ),
                    ['locale']
                )
            ],
            remainder='passthrough',
            sparse_threshold=0
        )
    )

    catalog_preprocessor.set_output(
        transform='pandas'
    )

    catalog = (
        catalog_preprocessor
        .fit_transform(catalog)
    )

    return (
        catalog,
        catalog_preprocessor
    )


def split_xy(
    df,
    target_column='remainder__next_item'
):
    X = df.drop(
        columns=[target_column]
    )

    y = (
        df[target_column]
        .astype(np.int64)
    )

    return X, y