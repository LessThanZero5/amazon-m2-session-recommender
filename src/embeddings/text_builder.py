import pandas as pd


PRODUCT_TEXT_FIELDS = {
    'title': 'Title',
    'brand': 'Brand',
    'color': 'Color',
    'size': 'Size',
    'model': 'Model',
    'material': 'Material',
    'author': 'Author',
    'desc': 'Product Description'
}


def build_product_text(
    row,
    mapping=None
):
    if mapping is None:
        mapping = PRODUCT_TEXT_FIELDS

    parts = []

    for column, label in mapping.items():
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value == '':
            continue

        parts.append(
            f'{label} | {value}'
        )

    if len(parts) == 0:
        return None

    return ' '.join(parts)


def build_product_texts(
    catalog,
    mapping=None
):
    catalog = catalog.copy()

    catalog['product_info'] = (
        catalog.apply(
            lambda row: build_product_text(
                row,
                mapping=mapping
            ),
            axis=1
        )
    )

    valid_catalog = (
        catalog
        .dropna(
            subset=['product_info']
        )
        .copy()
    )

    texts = (
        valid_catalog[
            'product_info'
        ]
        .tolist()
    )

    return valid_catalog, texts