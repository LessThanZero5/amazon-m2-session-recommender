import numpy as np
import torch

from tqdm.auto import tqdm


def get_ranks(
    top_ids,
    targets
):
    matches = top_ids.eq(
        targets.unsqueeze(1)
    )

    found = (
        matches.any(dim=1)
    )

    ranks = torch.zeros(
        len(targets),
        dtype=torch.long,
        device=targets.device
    )

    ranks[found] = (
        matches[found]
        .float()
        .argmax(dim=1)
        + 1
    )

    return ranks


def build_catalog_vectors(
    model,
    embedding_matrix,
    device,
    batch_size=20000
):
    n_items = (
        embedding_matrix.shape[0]
        - 1
    )

    vectors = []

    model.eval()

    with torch.inference_mode():
        for start in tqdm(
            range(
                1,
                n_items + 1,
                batch_size
            ),
            desc='Building catalog vectors'
        ):
            end = min(
                start + batch_size,
                n_items + 1
            )

            semantic = (
                torch.from_numpy(
                    np.asarray(
                        embedding_matrix[
                            start:end
                        ],
                        dtype=np.float32
                    )
                )
                .to(device)
            )

            projected = (
                model.project_items(
                    semantic
                )
            )

            vectors.append(
                projected.cpu()
            )

    return torch.cat(
        vectors,
        dim=0
    )


def evaluate_full_catalog(
    model,
    loader,
    embedding_matrix,
    device,
    top_k=20,
    catalog_batch_size=20000
):
    catalog_vectors = (
        build_catalog_vectors(
            model,
            embedding_matrix,
            device,
            catalog_batch_size
        )
        .to(device)
    )

    totals = {
        'Recall@1': 0.0,
        'Recall@5': 0.0,
        'Recall@10': 0.0,
        'Recall@20': 0.0,
        'NDCG@10': 0.0,
        'NDCG@20': 0.0,
        'MRR@20': 0.0
    }

    total_samples = 0

    model.eval()

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=(
                'Full-catalog evaluation'
            )
        )

        for (
            sequences,
            locales,
            targets
        ) in progress:

            sequences = sequences.to(
                device
            )

            locales = locales.to(
                device
            )

            targets = targets.to(
                device
            )

            padding_mask = (
                sequences.eq(0)
            )

            semantic_sequences = (
                torch.from_numpy(
                    np.asarray(
                        embedding_matrix[
                            sequences
                            .cpu()
                            .numpy()
                        ],
                        dtype=np.float32
                    )
                )
                .to(device)
            )

            session_vectors = model(
                semantic_sequences,
                locales,
                padding_mask
            )

            scores = (
                session_vectors
                @ catalog_vectors.T
            )

            top_ids = (
                torch.topk(
                    scores,
                    k=top_k,
                    dim=1
                )
                .indices
                + 1
            )

            ranks = get_ranks(
                top_ids,
                targets
            )

            batch_size = (
                len(targets)
            )

            totals['Recall@1'] += (
                (
                    (
                        (ranks > 0)
                        & (ranks <= 1)
                    )
                    .float()
                    .sum()
                    .item()
                )
            )

            totals['Recall@5'] += (
                (
                    (
                        (ranks > 0)
                        & (ranks <= 5)
                    )
                    .float()
                    .sum()
                    .item()
                )
            )

            totals['Recall@10'] += (
                (
                    (
                        (ranks > 0)
                        & (ranks <= 10)
                    )
                    .float()
                    .sum()
                    .item()
                )
            )

            totals['Recall@20'] += (
                (
                    (
                        (ranks > 0)
                        & (ranks <= 20)
                    )
                    .float()
                    .sum()
                    .item()
                )
            )

            ndcg10 = torch.where(
                (
                    (ranks > 0)
                    & (ranks <= 10)
                ),
                1.0 / torch.log2(
                    ranks.float()
                    + 1.0
                ),
                torch.zeros_like(
                    ranks,
                    dtype=torch.float32
                )
            )

            ndcg20 = torch.where(
                (
                    (ranks > 0)
                    & (ranks <= 20)
                ),
                1.0 / torch.log2(
                    ranks.float()
                    + 1.0
                ),
                torch.zeros_like(
                    ranks,
                    dtype=torch.float32
                )
            )

            mrr20 = torch.where(
                (
                    (ranks > 0)
                    & (ranks <= 20)
                ),
                1.0 / ranks.float(),
                torch.zeros_like(
                    ranks,
                    dtype=torch.float32
                )
            )

            totals['NDCG@10'] += (
                ndcg10
                .sum()
                .item()
            )

            totals['NDCG@20'] += (
                ndcg20
                .sum()
                .item()
            )

            totals['MRR@20'] += (
                mrr20
                .sum()
                .item()
            )

            total_samples += (
                batch_size
            )

    return {
        metric: value
        / total_samples
        for metric, value
        in totals.items()
    }