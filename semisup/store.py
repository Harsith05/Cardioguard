"""Tables for pseudo-labels (Phase 5).

Pseudo-labels live in their OWN table, not in predictions.label:
  * a human diagnosis can never be overwritten by a machine guess,
  * the drift job's accuracy check keeps using human labels only,
  * no migration of the existing predictions table is needed.

pseudo_labels       one row per pseudo-labelled prediction
pseudo_label_runs   one row per job run (acceptance rate, disagreement, ...) -> Grafana
"""
from __future__ import annotations

from sqlalchemy import (Column, DateTime, Float, Index, Integer, MetaData, String, Table, Text,
                        UniqueConstraint)

metadata = MetaData()

pseudo_labels = Table(
    "pseudo_labels", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(36), unique=True, nullable=False, index=True),
    Column("label", Integer, nullable=False),
    Column("confidence", Float, nullable=False),       # mean of the two methods' confidence
    Column("self_training_prob", Float),
    Column("label_spreading_prob", Float),
    Column("run_id", Integer, index=True),
    Column("model_version", String(32)),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# Every unlabelled prediction a model version has already scored. Without this, rows that
# were rejected would be re-scored every hour, and the pool would fill with hard cases,
# dragging the acceptance rate down by itself. A NEW model version looks at them again.
pseudo_label_seen = Table(
    "pseudo_label_seen", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(36), nullable=False),
    Column("model_version", String(32), nullable=False),
    Column("run_id", Integer),
    UniqueConstraint("request_id", "model_version", name="uq_seen_request_model"),
    Index("ix_seen_model_request", "model_version", "request_id"),
)

pseudo_label_runs = Table(
    "pseudo_label_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_at", DateTime(timezone=True), nullable=False, index=True),
    Column("model_version", String(32)),
    Column("status", String(24)),                      # ok / skipped_drift / insufficient_data
    Column("reason", Text),
    Column("n_unlabelled", Integer),
    Column("n_human_labels", Integer),
    Column("n_confident_self_training", Integer),
    Column("n_confident_spreading", Integer),
    Column("n_agree_confident", Integer),
    Column("n_accepted", Integer),
    Column("acceptance_rate", Float),                  # accepted / unlabelled
    Column("disagreement_rate", Float),                # the two methods disagree
    Column("mean_confidence", Float),                  # self-training confidence on all unlabelled
    Column("positive_share", Float),                   # share of accepted labels that are "disease"
    Column("total_pseudo_labels", Integer),
    Column("seconds", Float),
)
