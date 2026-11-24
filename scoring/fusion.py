"""Score-level fusion of the deep cosine score with handcrafted similarity features.

Two fusion modes are provided:

``weighted``
    ``Final = w1 * S_deep + w2 * mean(H_hand)`` with ``w1 + w2 == 1``. The hand term is
    the mean of the "higher is more similar" handcrafted vector, so ``x_slant_diff`` is
    inverted first. Because the combination is linear, fusing each reference pair and
    averaging over references gives the same result as fusing the averaged scores.

``logistic``
    A logistic-regression classifier over ``[S_deep, *H_hand]``, fitted on a validation
    split and stored as plain JSON coefficients. It learns the fusion coefficients
    instead of assuming a fixed weight and outputs a probability in ``[0, 1]``.

Only the nine handcrafted features from :mod:`scoring.handcrafted_features` are used, so
the fused vector stays low-dimensional. :func:`_drop_collinear` removes near-duplicate
features before the logistic fit, which keeps the coefficients identifiable.

The verification protocol lives here too, so ``training.fit_fusion`` (fit on validation)
and ``training.evaluate`` (report on test) cannot disagree about what a "genuine",
"skilled", or "random" query means.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from scoring.reference_selection import select_reference_indices
from scoring.handcrafted_features import (
    DISTANCE_FEATURES,
    FEATURE_NAMES,
    SignatureDescriptors,
    hand_similarity_vector,
    pair_features,
)
from scoring.metrics import eer_threshold

CONFIG_VERSION = 1
QueryKind = Literal["genuine", "skilled", "random"]
FusionType = Literal["weighted", "logistic"]


def _optional_float(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is absent or not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class FusionConfig:
    """A portable fusion configuration, saved as JSON next to the model checkpoint.

    ``weights`` apply to the weighted mode. ``coef``/``mean``/``scale``/``intercept``
    apply to the logistic mode and are ordered as ``[S_deep, *features]``. Storing the
    standardization statistics explicitly avoids pickling a scikit-learn estimator, so
    the config stays human-readable and version-control friendly.
    """

    type: FusionType
    features: tuple[str, ...] = FEATURE_NAMES
    deep_weight: float = 0.7
    hand_weight: float = 0.3
    coef: tuple[float, ...] = ()
    mean: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    intercept: float = 0.0
    dropped_features: tuple[str, ...] = ()
    # Decision threshold in the fused score space. ``None`` means "not calibrated yet";
    # callers then fall back to the EER threshold measured on the validation split.
    threshold: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in {"weighted", "logistic"}:
            raise ValueError(f"unsupported fusion type: {self.type}")
        if not self.features:
            raise ValueError("fusion config must list at least one handcrafted feature")
        if self.deep_weight < 0 or self.hand_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.deep_weight + self.hand_weight <= 0:
            raise ValueError("fusion weights must not both be zero")
        if self.type == "logistic":
            expected = len(self.features) + 1
            for name, values in (
                ("coef", self.coef),
                ("mean", self.mean),
                ("scale", self.scale),
            ):
                if len(values) != expected:
                    raise ValueError(
                        f"logistic config needs {expected} {name} values, got {len(values)}"
                    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "version": CONFIG_VERSION,
            "type": self.type,
            "features": list(self.features),
            "weights": {"deep": self.deep_weight, "hand": self.hand_weight},
            "coef": list(self.coef),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "intercept": self.intercept,
            "dropped_features": list(self.dropped_features),
            "threshold": self.threshold,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FusionConfig":
        """Rebuild a config from :meth:`to_dict` output."""
        version = payload.get("version", CONFIG_VERSION)
        if version != CONFIG_VERSION:
            raise ValueError(f"unsupported fusion config version: {version}")
        weights = payload.get("weights") or {}
        if not isinstance(weights, Mapping):
            raise ValueError("fusion config 'weights' must be an object")
        return cls(
            type=payload["type"],  # type: ignore[arg-type]
            features=tuple(payload["features"]),  # type: ignore[arg-type]
            deep_weight=float(weights.get("deep", 0.0)),
            hand_weight=float(weights.get("hand", 0.0)),
            coef=tuple(float(value) for value in payload.get("coef", ())),
            mean=tuple(float(value) for value in payload.get("mean", ())),
            scale=tuple(float(value) for value in payload.get("scale", ())),
            intercept=float(payload.get("intercept", 0.0)),
            dropped_features=tuple(payload.get("dropped_features", ())),  # type: ignore[arg-type]
            threshold=_optional_float(payload.get("threshold")),
            metadata=dict(payload.get("metadata") or {}),
        )


def save_fusion_config(config: FusionConfig, path: str | Path) -> Path:
    """Write ``config`` as indented JSON and return the resolved path."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    return destination


def load_fusion_config(path: str | Path) -> FusionConfig:
    """Read a fusion config written by :func:`save_fusion_config`."""
    source = Path(path).expanduser().resolve()
    return FusionConfig.from_dict(json.loads(source.read_text(encoding="utf-8")))


class FusionScorer:
    """Combine a deep cosine score with handcrafted features into one decision score.

    ``config.type`` selects the rule:

    * ``weighted``: ``w1 * deep + w2 * mean(hand_similarity_vector)``. The weights are
      normalised on construction, so the output stays in ``[0, 1]`` whenever the inputs
      are in ``[0, 1]``.
    * ``logistic``: ``sigmoid(coef . ((x - mean) / scale) + intercept)`` with
      ``x = [deep, *features]``. The output is a probability in ``[0, 1]``. A missing
      handcrafted feature falls back to its training mean, so a partially computed
      descriptor degrades gracefully instead of raising.

    The scorer is stateless and cheap, so it can be constructed once per process and
    called per pair.
    """

    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self.features = tuple(config.features)
        if config.type == "weighted":
            total = config.deep_weight + config.hand_weight
            self._deep_weight = config.deep_weight / total
            self._hand_weight = config.hand_weight / total
        else:
            self._coef = np.asarray(config.coef, dtype=np.float64)
            self._mean = np.asarray(config.mean, dtype=np.float64)
            self._scale = np.asarray(config.scale, dtype=np.float64)
            # Guard against a zero scale written by hand, which would divide by zero.
            self._scale = np.where(np.abs(self._scale) < 1e-12, 1.0, self._scale)

    def score(self, deep_similarity: float, hand: Mapping[str, float]) -> float:
        """Return the fused score for one ``(reference, query)`` pair."""
        if self.config.type == "weighted":
            return float(
                self._deep_weight * float(deep_similarity)
                + self._hand_weight * float(self.hand_mean(hand))
            )
        return float(self._logistic(float(deep_similarity), hand))

    def hand_mean(self, hand: Mapping[str, float]) -> float:
        """Return ``mean(hand_similarity_vector)`` over the configured features."""
        return float(self._hand_vector(hand).mean())

    def _hand_vector(self, hand: Mapping[str, float]) -> np.ndarray:
        values = np.empty(len(self.features), dtype=np.float64)
        for index, name in enumerate(self.features):
            value = hand.get(name)
            if name in DISTANCE_FEATURES:
                # A distance feature is "most similar" at 0, so an absent value must
                # fall back to 0 *before* the flip. Falling back to 1 first would turn
                # a missing measurement into the most dissimilar possible value.
                values[index] = 0.0 if value is None else 1.0 - float(value)
            else:
                # Absent similarity features are neutral at the "most similar" value.
                values[index] = 1.0 if value is None else float(value)
        return values

    def _logistic(self, deep_similarity: float, hand: Mapping[str, float]) -> float:
        x = np.empty(len(self.features) + 1, dtype=np.float64)
        x[0] = deep_similarity
        for index, name in enumerate(self.features):
            value = hand.get(name)
            x[index + 1] = self._mean[index + 1] if value is None else float(value)
        standardized = (x - self._mean) / self._scale
        logit = float(np.dot(self._coef, standardized) + self.config.intercept)
        # Clip before the exponential so a wild coefficient cannot overflow.
        logit = float(np.clip(logit, -50.0, 50.0))
        return 1.0 / (1.0 + np.exp(-logit))

    def score_pairs(
        self, deep: np.ndarray | Sequence[float], hand_rows: Sequence[Mapping[str, float]]
    ) -> np.ndarray:
        """Vectorised convenience wrapper over :meth:`score`."""
        deep_array = np.asarray(deep, dtype=np.float64)
        if len(deep_array) != len(hand_rows):
            raise ValueError("deep scores and hand feature rows must have equal length")
        return np.array(
            [self.score(value, row) for value, row in zip(deep_array, hand_rows)],
            dtype=np.float64,
        )

    def score_matrix(self, deep: np.ndarray | Sequence[float], hand_matrix: np.ndarray) -> np.ndarray:
        """Score a whole ``(N, len(features))`` matrix without per-row dictionaries.

        The fitting path evaluates hundreds of thousands of reference pairs, so building
        one ``dict`` per pair would dominate the runtime. This method applies the same
        arithmetic as :meth:`score` directly to the matrix, with columns in
        ``self.features`` order. Absent-feature fallbacks do not apply here: the matrix
        must already contain a value for every configured feature.
        """
        deep_array = np.asarray(deep, dtype=np.float64).reshape(-1)
        matrix = np.asarray(hand_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError(
                f"hand matrix must have {len(self.features)} columns in features order"
            )
        if len(deep_array) != matrix.shape[0]:
            raise ValueError("deep scores and hand matrix rows must have equal length")

        if self.config.type == "weighted":
            similarity = _hand_similarity_matrix(matrix, self.features)
            return self._deep_weight * deep_array + self._hand_weight * similarity.mean(axis=1)

        x = np.column_stack([deep_array, matrix])
        standardized = (x - self._mean) / self._scale
        logit = np.clip(standardized @ self._coef + self.config.intercept, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logit))

    def score_rows(
        self,
        deep: np.ndarray | Sequence[float],
        hand_matrix: np.ndarray,
        names: Sequence[str] = FEATURE_NAMES,
    ) -> np.ndarray:
        """Score a matrix whose columns follow ``names`` instead of ``self.features``.

        :meth:`score_matrix` wants exactly the configured columns in the configured
        order, which is convenient when a matrix is built per config. The fitting paths
        build one matrix for all nine :data:`FEATURE_NAMES` columns and then score it
        with configs that may have dropped some, so this wrapper selects the configured
        columns by name first. It is a pure column gather, so the scores are identical
        to calling :meth:`score_matrix` on a pre-selected matrix.
        """
        matrix = np.asarray(hand_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ValueError("hand matrix shape does not match the feature name list")
        index_by_name = {name: position for position, name in enumerate(names)}
        try:
            columns = [index_by_name[name] for name in self.features]
        except KeyError as error:  # pragma: no cover - configs are built from these names
            raise KeyError(f"configured feature is missing from the matrix: {error}") from error
        return self.score_matrix(deep, matrix[:, columns])


def _drop_collinear(
    hand_matrix: np.ndarray,
    names: Sequence[str],
    threshold: float = 0.95,
) -> tuple[list[int], list[str]]:
    """Return kept column indices and dropped names for ``|r| >= threshold``.

    Near-duplicate features make logistic coefficients unidentifiable and inflate their
    variance. Features are kept in declared order and a later feature is dropped when it
    correlates too strongly with an already kept one, so the outcome is deterministic and
    the first listed feature always wins.
    """
    matrix = np.asarray(hand_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("hand matrix shape does not match the feature name list")
    if matrix.shape[0] < 2:
        return list(range(len(names))), []

    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(matrix, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)

    kept: list[int] = []
    dropped: list[str] = []
    for column in range(len(names)):
        if any(abs(correlation[column, other]) >= threshold for other in kept):
            dropped.append(names[column])
        else:
            kept.append(column)
    return kept, dropped


def _hand_similarity_matrix(hand_matrix: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Return ``hand_matrix`` with distance features flipped to "higher is similar"."""
    similarity = np.asarray(hand_matrix, dtype=np.float64).copy()
    for column, name in enumerate(names):
        if name in DISTANCE_FEATURES:
            similarity[:, column] = 1.0 - similarity[:, column]
    return similarity


def fit_weighted_fusion(
    deep: np.ndarray,
    hand_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    names: Sequence[str] = FEATURE_NAMES,
    step: float = 0.05,
    tie_tolerance: float = 1e-6,
) -> tuple[FusionConfig, dict[str, object]]:
    """Grid-search the deep/hand weight that minimises EER on the fitting split.

    ``labels`` is 1 for genuine queries and 0 for negatives. The hand term is the mean of
    the "higher is more similar" vector, so ``x_slant_diff`` is inverted first. The full
    grid is returned in the metadata so the choice can be inspected.

    EER ties are common when one modality already separates the fit split perfectly
    (synthetic data, or a very easy split). Returning the first minimum would silently
    discard the deep score and produce a "hand-only" config, which defeats the point of
    fusion. Among the weights whose EER is within ``tie_tolerance`` of the best, the one
    closest to 0.5 is kept, so both modalities keep contributing and the operating point
    sits in the middle of the near-optimal plateau.
    """
    deep_array = np.asarray(deep, dtype=np.float64)
    hand_array = np.asarray(hand_matrix, dtype=np.float64)
    label_array = np.asarray(labels).astype(bool)
    if not (len(deep_array) == len(hand_array) == len(label_array)):
        raise ValueError("deep, hand, and label arrays must have equal length")
    if label_array.sum() == 0 or (~label_array).sum() == 0:
        raise ValueError("the fitting split needs both genuine and negative pairs")

    hand_mean = _hand_similarity_matrix(hand_array, names).mean(axis=1)
    genuine = label_array
    negatives = ~label_array

    grid: list[dict[str, float]] = []
    candidates: list[tuple[float, float]] = []
    for weight in np.arange(0.0, 1.0 + 1e-9, step):
        fused = weight * deep_array + (1.0 - weight) * hand_mean
        _, current_eer = eer_threshold(fused[genuine], fused[negatives])
        candidates.append((float(weight), float(current_eer)))
        grid.append({"deep_weight": round(float(weight), 4), "eer": round(float(current_eer), 6)})

    best_eer = min(value for _, value in candidates)
    plateau = [weight for weight, value in candidates if value <= best_eer + tie_tolerance]
    best_weight = min(plateau, key=lambda weight: abs(weight - 0.5))

    config = FusionConfig(
        type="weighted",
        features=tuple(names),
        deep_weight=best_weight,
        hand_weight=1.0 - best_weight,
        metadata={"fit_eer": best_eer, "grid": grid},
    )
    return config, {"fit_eer": best_eer, "deep_weight": best_weight, "grid": grid}


def fit_logistic_fusion(
    deep: np.ndarray,
    hand_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    names: Sequence[str] = FEATURE_NAMES,
    C: float = 1.0,
    drop_correlated: float | None = 0.95,
) -> tuple[FusionConfig, dict[str, object]]:
    """Fit a logistic-regression fusion rule and return it as a portable config.

    The design matrix is ``[deep, *hand]``. Standardization statistics are stored in the
    config instead of a pickled estimator, so the JSON stays readable and portable.

    Args:
        deep: Deep cosine scores on the fitting split.
        hand_matrix: Handcrafted features in ``names`` order.
        labels: 1 for genuine queries, 0 for negatives.
        names: Feature names matching ``hand_matrix`` columns.
        C: Inverse L2 regularization strength for scikit-learn.
        drop_correlated: Drop features with ``|r| >=`` this value against a kept one.
            ``None`` disables the guard.
    """
    from sklearn.linear_model import LogisticRegression

    deep_array = np.asarray(deep, dtype=np.float64)
    hand_array = np.asarray(hand_matrix, dtype=np.float64)
    label_array = np.asarray(labels).astype(int)
    if not (len(deep_array) == len(hand_array) == len(label_array)):
        raise ValueError("deep, hand, and label arrays must have equal length")
    if len(np.unique(label_array)) < 2:
        raise ValueError("the fitting split needs both genuine and negative pairs")

    kept_indices = list(range(len(names)))
    dropped: list[str] = []
    if drop_correlated is not None:
        kept_indices, dropped = _drop_collinear(hand_array, names, threshold=drop_correlated)
    kept_names = [names[index] for index in kept_indices]

    design = np.column_stack([deep_array, hand_array[:, kept_indices]])
    mean = design.mean(axis=0)
    scale = design.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (design - mean) / scale

    # class_weight="balanced" because random negatives vastly outnumber genuine queries.
    estimator = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", solver="lbfgs")
    estimator.fit(standardized, label_array)

    coef = estimator.coef_.reshape(-1)
    intercept = float(estimator.intercept_.reshape(-1)[0])
    config = FusionConfig(
        type="logistic",
        features=tuple(kept_names),
        coef=tuple(float(value) for value in coef),
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        intercept=intercept,
        dropped_features=tuple(dropped),
        metadata={
            "C": C,
            "drop_correlated": drop_correlated,
            "n_samples": int(len(label_array)),
            "positives": int(label_array.sum()),
        },
    )
    return config, {"dropped_features": dropped, "coef": coef.tolist(), "intercept": intercept}


@dataclass(frozen=True)
class Query:
    """One query image and the signer it is compared against.

    ``query_index`` is the position of this query in ``Enrollment.queries``. It is
    carried so per-pair fitting rows can be grouped back into the per-query scores that
    evaluation actually averages over, which is what the cross-validated EER needs.
    """

    index: int
    target_signer: str
    kind: QueryKind
    query_index: int = -1


@dataclass(frozen=True)
class Enrollment:
    """Reference indices per signer plus the query list to score."""

    references: dict[str, list[int]]
    queries: list[Query]
    skipped_signers: tuple[str, ...] = ()


def enroll_and_build_queries(
    records: Sequence[object],
    references: int,
    *,
    max_random_queries: int | None = None,
    seed: int = 0,
    reference_selection: str = "first",
    embeddings: np.ndarray | None = None,
    already_preprocessed: bool = True,
    ink: str = "auto",
    image_height: int = 256,
    image_width: int = 512,
) -> Enrollment:
    """Enrol references and list queries, skipping undersized signers."""
    by_signer = defaultdict(lambda: {"gen": [], "forg": []})
    for index, record in enumerate(records):
        signer_id = str(getattr(record, "signer_id"))
        is_forgery = bool(getattr(record, "is_forgery"))
        by_signer[signer_id]["forg" if is_forgery else "gen"].append(index)

    enrollments = {}
    reference_sets = {}
    skipped = []
    for signer_id, groups in by_signer.items():
        genuine = groups["gen"]
        if len(genuine) < references:
            skipped.append(signer_id)
            continue
        selected = select_reference_indices(
            records,
            genuine,
            references,
            mode=reference_selection,
            embeddings=embeddings,
            already_preprocessed=already_preprocessed,
            ink=ink,
            image_height=image_height,
            image_width=image_width,
        )
        enrollments[signer_id] = selected
        reference_sets[signer_id] = set(selected)
    if not enrollments:
        raise ValueError(
            f"no signer has at least {references} genuine images"
        )

    queries = []
    for signer_id, groups in by_signer.items():
        if signer_id not in enrollments:
            continue
        selected = reference_sets[signer_id]
        queries.extend(
            Query(index=index, target_signer=signer_id, kind="genuine")
            for index in groups["gen"]
            if index not in selected
        )
        queries.extend(
            Query(index=index, target_signer=signer_id, kind="skilled")
            for index in groups["forg"]
        )
        for other, other_groups in by_signer.items():
            if other == signer_id:
                continue
            other_selected = reference_sets.get(other, set())
            queries.extend(
                Query(index=index, target_signer=signer_id, kind="random")
                for index in other_groups["gen"]
                if index not in other_selected
            )

    if max_random_queries is not None:
        fixed = [q for q in queries if q.kind != "random"]
        random_queries = [q for q in queries if q.kind == "random"]
        if len(random_queries) > max_random_queries:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(
                len(random_queries),
                size=max_random_queries,
                replace=False,
            )
            random_queries = [random_queries[int(i)] for i in np.sort(chosen)]
        queries = fixed + random_queries

    queries = [
        Query(
            index=q.index,
            target_signer=q.target_signer,
            kind=q.kind,
            query_index=position,
        )
        for position, q in enumerate(queries)
    ]
    return Enrollment(
        references=enrollments,
        queries=queries,
        skipped_signers=tuple(sorted(skipped)),
    )


def score_queries(
    embeddings: np.ndarray,
    descriptors: Sequence[SignatureDescriptors],
    enrollment: Enrollment,
    scorer: FusionScorer | None = None,
) -> dict[str, np.ndarray]:
    """Score every enrolled query and return deep and fused score arrays by kind.

    For each query the function iterates the target signer's references, computes the
    deep cosine similarity and the handcrafted features per reference, and averages over
    references. When ``scorer`` is given the fused score is computed *per reference pair*
    and then averaged, so per-reference evidence is preserved; for the weighted mode this
    is mathematically identical to fusing the averaged scores.

    Returns:
        A dict with ``deep_<kind>`` arrays always, plus ``fused_<kind>`` arrays when a
        scorer is supplied. Kinds are ``genuine``, ``skilled``, and ``random``.
    """
    embedding_array = np.asarray(embeddings, dtype=np.float64)
    buckets: dict[str, dict[str, list[float]]] = {
        kind: {"deep": [], "fused": []} for kind in ("genuine", "skilled", "random")
    }

    for query in enrollment.queries:
        reference_indices = enrollment.references[query.target_signer]
        deep_scores = embedding_array[reference_indices] @ embedding_array[query.index]
        buckets[query.kind]["deep"].append(float(deep_scores.mean()))
        if scorer is not None:
            pair_scores = [
                scorer.score(
                    float(deep_scores[position]),
                    pair_features(descriptors[reference], descriptors[query.index]),
                )
                for position, reference in enumerate(reference_indices)
            ]
            buckets[query.kind]["fused"].append(float(np.mean(pair_scores)))

    results: dict[str, np.ndarray] = {}
    for kind, values in buckets.items():
        results[f"deep_{kind}"] = np.asarray(values["deep"], dtype=np.float64)
        if scorer is not None:
            results[f"fused_{kind}"] = np.asarray(values["fused"], dtype=np.float64)
    return results


@dataclass(frozen=True)
class PairMatrix:
    """Per-``(reference, query)`` fitting rows produced by the shared protocol.

    ``score_queries`` averages over references, which is what evaluation needs. Fitting
    needs the individual pairs, so this container keeps one row per reference pair and
    carries the signer id of each row so a grouped cross-validation can hold out whole
    signers rather than leaking identities across folds.
    """

    deep: np.ndarray
    hand: np.ndarray
    labels: np.ndarray
    signers: np.ndarray
    kinds: np.ndarray
    query_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.deep.shape[0])

    def subset(self, indices: np.ndarray) -> "PairMatrix":
        """Return a row-subset view, preserving column order."""
        return PairMatrix(
            deep=self.deep[indices],
            hand=self.hand[indices],
            labels=self.labels[indices],
            signers=self.signers[indices],
            kinds=self.kinds[indices],
            query_ids=self.query_ids[indices],
        )


def build_pair_matrix(
    embeddings: np.ndarray,
    descriptors: Sequence[SignatureDescriptors],
    enrollment: Enrollment,
    *,
    names: Sequence[str] = FEATURE_NAMES,
) -> PairMatrix:
    """Expand an enrollment into one row per ``(reference, query)`` pair.

    Every query is compared against each of its target signer's references, which is the
    same unit the deep score averages over. Rows are labelled 1 for genuine queries and 0
    for skilled and random negatives, so a single classifier can be fitted on the union.

    Args:
        embeddings: ``(N, D)`` L2-normalised embeddings indexed like the records.
        descriptors: Per-record descriptors indexed like the records.
        enrollment: Output of :func:`enroll_and_build_queries`.
        names: Handcrafted feature order for the ``hand`` matrix columns.
    """
    embedding_array = np.asarray(embeddings, dtype=np.float64)
    deep_rows: list[float] = []
    hand_rows: list[np.ndarray] = []
    labels: list[int] = []
    signers: list[str] = []
    kinds: list[str] = []
    query_ids: list[int] = []

    for query in enrollment.queries:
        reference_indices = enrollment.references[query.target_signer]
        deep_scores = embedding_array[reference_indices] @ embedding_array[query.index]
        for position, reference in enumerate(reference_indices):
            features = pair_features(descriptors[reference], descriptors[query.index])
            deep_rows.append(float(deep_scores[position]))
            hand_rows.append(np.array([features[name] for name in names], dtype=np.float64))
            labels.append(1 if query.kind == "genuine" else 0)
            signers.append(query.target_signer)
            kinds.append(query.kind)
            query_ids.append(query.query_index)

    return PairMatrix(
        deep=np.asarray(deep_rows, dtype=np.float64),
        hand=np.vstack(hand_rows) if hand_rows else np.empty((0, len(names))),
        labels=np.asarray(labels, dtype=np.int64),
        signers=np.asarray(signers, dtype=object),
        kinds=np.asarray(kinds, dtype=object),
        query_ids=np.asarray(query_ids, dtype=np.int64),
    )
