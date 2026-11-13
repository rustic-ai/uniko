# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Pydantic model of the full engine configuration.

:class:`UnikoConfigModel` mirrors the Rust ``UnikoConfig``: the shape
``Uniko.config()`` returns and ``UnikoBuilder.config()`` accepts. Every field
has the engine's default, so a model built with a few fields set is a partial
config, and ``UnikoBuilder.config()`` sends only the fields that were set::

    from uniko import Uniko
    from uniko.models import UnikoConfigModel

    cfg = UnikoConfigModel(recall_limit=10, reranker={"top_n": 40})
    memory = Uniko.builder().config(cfg).build_sync()

Unknown keys are rejected (``extra="forbid"``), matching the native check, so
a typo fails here rather than being dropped. ``test_parity.py`` checks that the
fields and defaults stay in step with the engine.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

_CFG = ConfigDict(extra="forbid")


class EmbeddingConfigModel(BaseModel):
    """The embedding model (``embedding``)."""

    model_config = _CFG

    model_id: str = "BGESmallENV15"
    dimensions: int = 384
    batch_size: int = 32
    document_prefix: Optional[str] = None
    query_prefix: Optional[str] = (
        "Represent this sentence for searching relevant passages: "
    )
    execution_providers: Optional[list[str]] = None
    provider: str = "local/onnx"
    #: Provider-specific options, such as a remote endpoint and key. Shown as
    #: ``"<redacted>"`` by ``Uniko.config()`` when set.
    provider_options: Optional[Any] = None
    sparse_dimensions: Optional[int] = None
    multivector_dimensions: Optional[int] = None


class RerankerConfigModel(BaseModel):
    """The cross-encoder reranker (``reranker``)."""

    model_config = _CFG

    enabled: bool = True
    model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 50
    apply_sigmoid: bool = True
    execution_providers: Optional[list[str]] = None
    style: str = "cross-encoder"


class NlpConfigModel(BaseModel):
    """The NER / observation-extraction cascade (``nlp``)."""

    model_config = _CFG

    model_id: str = "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall"
    artifact: str = "onnx/cascade-int8.onnx"
    max_batch_size: int = 16
    execution_providers: Optional[list[str]] = None


class OcrConfigModel(BaseModel):
    """OCR for tiered PDF extraction (``ocr``), off by default."""

    model_config = _CFG

    enabled: bool = False
    model_id: str = "monkt/paddleocr-onnx"
    rec_artifact: str = "languages/english/rec.onnx"
    char_dict_path: str = "languages/english/dict.txt"
    det_artifact: str = "detection/v5/det.onnx"
    image_height: int = 48
    image_width: int = 320
    normalization: str = "siglip"
    execution_providers: Optional[list[str]] = None


def _default_vector_algorithm() -> dict[str, Any]:
    return {"HnswSq": {"m": 16, "ef_construction": 100}}


def _default_blob_storage() -> dict[str, Any]:
    return {"kind": "lance"}


class UnikoConfigModel(BaseModel):
    """The full engine configuration (Rust ``UnikoConfig``)."""

    model_config = _CFG

    # Paths.
    catalog_path: Optional[str] = None
    schema_path: Optional[str] = None
    observation_rules_path: Optional[str] = None

    # Models.
    embedding: EmbeddingConfigModel = Field(default_factory=EmbeddingConfigModel)
    reranker: RerankerConfigModel = Field(default_factory=RerankerConfigModel)
    nlp: NlpConfigModel = Field(default_factory=NlpConfigModel)
    ocr: OcrConfigModel = Field(default_factory=OcrConfigModel)

    # Storage. Both are tagged enums, replaced whole rather than merged:
    # ``{"IvfPq": {...}}``, ``{"kind": "fs", "root": "/data/blobs"}``.
    vector_algorithm: dict[str, Any] = Field(default_factory=_default_vector_algorithm)
    vector_metric: str = "Cosine"
    blob_storage: dict[str, Any] = Field(default_factory=_default_blob_storage)

    # Streaming pipeline.
    ingest_queue_capacity: int = 200
    consolidation_queue_capacity: int = 32
    consolidation_threshold: int = 20
    consolidation_interval_secs: int = 900
    circuit_failure_threshold: int = 5
    circuit_recovery_ms: int = 60_000

    # Chunking.
    message_chunk_threshold: int = 1024
    action_output_artifact_threshold: int = 256
    max_chunk_tokens: int = 256
    min_chunk_tokens: int = 32
    chunk_overlap_tokens: int = 0

    # Extraction.
    nlp_srl_enabled: bool = True
    entity_strict_admission: bool = True
    entity_other_min_confidence: float = 0.9

    # Recall.
    recall_limit: int = 15
    recall_token_budget: int = 8192
    recall_min_score: float = 0.001
    recall_vector_weight: float = 0.5
    recall_bm25_weight: float = 0.5
    recall_sparse_enabled: bool = False
    query_variants: list[str] = Field(default_factory=list)
    rrf_k: float = 60.0
    recall_per_variant_limit: Optional[int] = None
    phase1_strategy: str = "boost"
    phase1_boost_alpha: float = 0.6
    phase2_coverage_threshold: float = 0.65
    phase2_mmr_lambda: float = 0.7
    phase2_mmr_duplicate_threshold: float = 0.85
    phase2_temporal_enabled: bool = True
    phase2_graph_enabled: bool = True
    phase2_graph_damping: float = 0.85
    phase2_graph_max_iter: int = 30
    phase2_graph_edge_weights: dict[str, float] = Field(default_factory=dict)

    # Lifecycle.
    half_life_days: float = 30.0
    prune_below: float = 0.05
    consolidation_cluster_objects: bool = True
    consolidation_date_augment_embedding: bool = True
    session_inactivity_secs: int = 3600

    @classmethod
    def from_native(cls, uni: Any) -> "UnikoConfigModel":
        """Build from a native ``Uniko`` (or ``TypedUniko``) handle via ``config()``.

        A redacted ``embedding.provider_options`` comes back as the string
        ``"<redacted>"``, so the result describes the instance but cannot
        rebuild one that needs those options.
        """
        native = uni._inner if hasattr(uni, "_inner") else uni
        return cls.model_validate(native.config())


__all__ = [
    "EmbeddingConfigModel",
    "NlpConfigModel",
    "OcrConfigModel",
    "RerankerConfigModel",
    "UnikoConfigModel",
]
