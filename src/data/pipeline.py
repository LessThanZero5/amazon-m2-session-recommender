from collections import Counter

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

    repeat_counts = dict(
        Counter(
            sessions_hist['items_list_str']
        )
    )

    sessions_hist['num_repeat'] = (
        sessions_hist['items_list_str']
        .map(repeat_counts)
    )

    repeated = sessions_hist[
        sessions_hist['num_repeat'] > 1
    ]

    unique = sessions_hist[
        sessions_hist['num_repeat'] == 1
    ]

    train_size = round(
        len(sessions_hist) * train_ratio
    )

    unique_to_take = (
        train_size - len(repeated)
    )

    if unique_to_take < 0:
        raise ValueError(
            'Repeated sessions exceed requested train size.'
        )

    train = pd.concat(
        [
            repeated,
            unique.iloc[:unique_to_take]
        ],
        ignore_index=True
    )

    val_test = (
        unique
        .iloc[unique_to_take:]
        .reset_index(drop=True)
    )

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

    train_encoded = (
        preprocessor
        .fit_transform(train)
    )

    val_encoded = (
        preprocessor
        .transform(val)
    )

    test_encoded = (
        preprocessor
        .transform(test)
    )

    helper_columns = [
        'remainder__session_len',
        'remainder__items_list_str',
        'remainder__num_repeat'
    ]

    train_encoded = train_encoded.drop(
        columns=helper_columns,
        errors='ignore'
    )

    val_encoded = val_encoded.drop(
        columns=helper_columns,
        errors='ignore'
    )

    test_encoded = test_encoded.drop(
        columns=helper_columns,
        errors='ignore'
    )

    return (
        train_encoded,
        val_encoded,
        test_encoded,
        preprocessor
    )


def process_catalog(
    catalog,
    anomaly_value=40_000_000.07
):
    catalog = catalog.copy()

    for locale in (
        catalog['locale']
        .dropna()
        .unique()
    ):
        locale_mask = (
            catalog['locale'] == locale
        )

        anomaly_mask = (
            locale_mask
            & np.isclose(
                catalog['price'],
                anomaly_value
            )
        )

        if anomaly_mask.any():
            region_median = (
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
            ] = region_median

        region_indexes = (
            catalog.index[
                locale_mask
            ]
        )

        scaler = StandardScaler()

        region_prices = (
            catalog.loc[
                region_indexes,
                'price'
            ]
            .to_numpy(
                dtype=np.float32
            )
            .reshape(-1, 1)
        )

        catalog.loc[
            region_indexes,
            'price'
        ] = (
            scaler
            .fit_transform(
                region_prices
            )
            .reshape(-1)
        )

        standardized = (
            catalog.loc[
                region_indexes,
                'price'
            ]
            .to_numpy()
        )

        assert np.isclose(
            standardized.mean(),
            0,
            atol=1e-6
        )

        assert np.isclose(
            standardized.std(ddof=0),
            1,
            atol=1e-6
        )

    catalog_preprocessor = ColumnTransformer(
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

    catalog_preprocessor.set_output(
        transform='pandas'
    )

    encoded_catalog = (
        catalog_preprocessor
        .fit_transform(catalog)
    )

    return (
        encoded_catalog,
        catalog_preprocessor
    )


def split_xy(
    dataset,
    target_column='remainder__next_item'
):
    X = dataset.drop(
        columns=[target_column]
    )

    y = (
        dataset[target_column]
        .astype(np.int64)
    )

    return X, y


def run_session_pipeline(
    sessions,
    train_ratio=0.8,
    random_state=42
):
    train, val, test = split_sessions(
        sessions=sessions,
        train_ratio=train_ratio,
        random_state=random_state
    )

    (
        train,
        val,
        test,
        session_preprocessor
    ) = encode_session_locales(
        train,
        val,
        test
    )

    train_X, train_y = split_xy(train)
    val_X, val_y = split_xy(val)
    test_X, test_y = split_xy(test)

    return {
        'train_X': train_X,
        'train_y': train_y,
        'val_X': val_X,
        'val_y': val_y,
        'test_X': test_X,
        'test_y': test_y,
        'session_preprocessor': session_preprocessor
    }
