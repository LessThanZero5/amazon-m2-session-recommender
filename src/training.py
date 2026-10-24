from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


def lookup_embeddings(
    embedding_matrix,
    ids,
    device
):
    ids_numpy = (
        ids
        .detach()
        .cpu()
        .numpy()
    )

    values = np.asarray(
        embedding_matrix[
            ids_numpy
        ],
        dtype=np.float32
    )

    return (
        torch
        .from_numpy(values)
        .to(
            device,
            non_blocking=True
        )
    )


def sample_negative_ids(
    targets,
    n_items,
    n_negatives=128
):
    negatives = torch.randint(
        1,
        n_items + 1,
        (
            targets.size(0),
            n_negatives
        ),
        device=targets.device
    )

    collision = negatives.eq(
        targets.unsqueeze(1)
    )

    while collision.any():
        negatives[
            collision
        ] = torch.randint(
            1,
            n_items + 1,
            (
                int(
                    collision.sum()
                ),
            ),
            device=targets.device
        )

        collision = (
            negatives.eq(
                targets.unsqueeze(1)
            )
        )

    return negatives


def calculate_logits(
    model,
    sequences,
    locales,
    targets,
    embedding_matrix,
    n_items,
    device,
    n_negatives=128
):
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
        lookup_embeddings(
            embedding_matrix,
            sequences,
            device
        )
    )

    session_vectors = model(
        semantic_sequences,
        locales,
        padding_mask
    )

    negative_ids = (
        sample_negative_ids(
            targets,
            n_items=n_items,
            n_negatives=n_negatives
        )
    )

    candidate_ids = torch.cat(
        [
            targets.unsqueeze(1),
            negative_ids
        ],
        dim=1
    )

    candidate_semantics = (
        lookup_embeddings(
            embedding_matrix,
            candidate_ids,
            device
        )
    )

    candidate_vectors = (
        model.project_items(
            candidate_semantics
        )
    )

    logits = torch.einsum(
        'bd,bkd->bk',
        session_vectors,
        candidate_vectors
    )

    labels = torch.zeros(
        targets.size(0),
        dtype=torch.long,
        device=device
    )

    return (
        logits,
        labels
    )


def sampled_metrics(
    logits,
    labels,
    k=10
):
    top_indices = torch.topk(
        logits,
        k=min(
            k,
            logits.size(1)
        ),
        dim=1
    ).indices

    matches = (
        top_indices.eq(
            labels.unsqueeze(1)
        )
    )

    recall = (
        matches
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

    found = (
        matches.any(dim=1)
    )

    ranks = torch.zeros(
        len(labels),
        dtype=torch.float32,
        device=logits.device
    )

    if found.any():
        ranks[found] = (
            matches[found]
            .float()
            .argmax(dim=1)
            + 1
        )

    ndcg = torch.where(
        ranks > 0,
        1.0 / torch.log2(
            ranks + 1.0
        ),
        torch.zeros_like(
            ranks
        )
    ).mean().item()

    return recall, ndcg


def train_model(
    model,
    train_loader,
    val_loader,
    embedding_matrix,
    n_items,
    device,
    experiment_name,
    checkpoint_dir,
    epochs=3,
    learning_rate=1e-3,
    weight_decay=1e-4,
    n_negatives=128,
    validation_batches=500,
    grad_clip=1.0,
    log_dir='runs'
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    checkpoint_dir = Path(
        checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = SummaryWriter(
        log_dir=str(
            Path(log_dir)
            / experiment_name
        )
    )

    best_val_loss = float(
        'inf'
    )

    global_step = 0

    for epoch in range(
        1,
        epochs + 1
    ):
        model.train()

        train_loss_sum = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f'{experiment_name} | '
                f'Epoch {epoch}/{epochs}'
            )
        )

        for (
            sequences,
            locales,
            targets
        ) in progress:

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, labels = (
                calculate_logits(
                    model,
                    sequences,
                    locales,
                    targets,
                    embedding_matrix,
                    n_items,
                    device,
                    n_negatives
                )
            )

            loss = (
                F.cross_entropy(
                    logits,
                    labels
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )

            optimizer.step()

            train_loss_sum += (
                loss.item()
            )

            global_step += 1

            writer.add_scalar(
                'Train/Loss_step',
                loss.item(),
                global_step
            )

            progress.set_postfix(
                loss=(
                    f'{loss.item():.4f}'
                )
            )

        train_loss = (
            train_loss_sum
            / len(train_loader)
        )

        model.eval()

        val_loss_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0.0
        val_steps = 0

        with torch.inference_mode():
            progress = tqdm(
                val_loader,
                desc='Validation'
            )

            for (
                sequences,
                locales,
                targets
            ) in progress:

                logits, labels = (
                    calculate_logits(
                        model,
                        sequences,
                        locales,
                        targets,
                        embedding_matrix,
                        n_items,
                        device,
                        n_negatives
                    )
                )

                loss = (
                    F.cross_entropy(
                        logits,
                        labels
                    )
                )

                recall, ndcg = (
                    sampled_metrics(
                        logits,
                        labels,
                        k=10
                    )
                )

                val_loss_sum += (
                    loss.item()
                )

                recall_sum += recall
                ndcg_sum += ndcg

                val_steps += 1

                if (
                    validation_batches
                    is not None
                    and val_steps
                    >= validation_batches
                ):
                    break

        val_loss = (
            val_loss_sum
            / val_steps
        )

        recall_10 = (
            recall_sum
            / val_steps
        )

        ndcg_10 = (
            ndcg_sum
            / val_steps
        )

        writer.add_scalar(
            'Train/Loss_epoch',
            train_loss,
            epoch
        )

        writer.add_scalar(
            'Validation/Loss',
            val_loss,
            epoch
        )

        writer.add_scalar(
            'Validation/Sampled_Recall@10',
            recall_10,
            epoch
        )

        writer.add_scalar(
            'Validation/Sampled_NDCG@10',
            ndcg_10,
            epoch
        )

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': (
                model.state_dict()
            ),
            'optimizer_state_dict': (
                optimizer.state_dict()
            ),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'recall_10': recall_10,
            'ndcg_10': ndcg_10
        }

        torch.save(
            checkpoint,
            checkpoint_dir
            / (
                f'{experiment_name}'
                f'_last.pt'
            )
        )

        if (
            val_loss
            < best_val_loss
        ):
            best_val_loss = (
                val_loss
            )

            torch.save(
                checkpoint,
                checkpoint_dir
                / (
                    f'{experiment_name}'
                    f'_best.pt'
                )
            )

    writer.close()

    return model