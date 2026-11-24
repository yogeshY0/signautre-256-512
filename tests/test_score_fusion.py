"""Focused tests for Stage 4 score-level fusion.

Covered here:

* ``FusionConfig`` validation and JSON round-trips (including the decision threshold);
* weighted and logistic scoring, the missing-feature fallback, and the guarantee that
  the vectorised paths (``score_matrix`` / ``score_rows``) agree with the per-pair path;
* the collinearity guard and both fitters;
* the verification protocol -- enrolment, per-pair rows, and per-reference averaging --
  which ``training.evaluate`` and ``training.fit_fusion`` must agree on.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scoring.fusion import (
    FusionConfig,
    FusionScorer,
    _drop_collinear,
    _hand_similarity_matrix,
    build_pair_matrix,
    enroll_and_build_queries,
    fit_logistic_fusion,
    fit_weighted_fusion,
    load_fusion_config,
    save_fusion_config,
    score_queries,
)
from scoring.handcrafted_features import (
    FEATURE_NAMES,
    PROFILE_BINS,
    SignatureDescriptors,
    hand_similarity_vector,
    pair_features,
)
from scoring.metrics import eer_threshold, rates_at_threshold
from training.fit_fusion import _query_scores_from_rows
from training.stage3_metric_learning import SignatureRecord

DISTANCE_NAME = "x_slant_diff"


def _descriptor(index: int) -> SignatureDescriptors:
    """Build a descriptor with varied, deterministic values for protocol tests."""
    profile = np.full(PROFILE_BINS, 1.0 / PROFILE_BINS, dtype=np.float64)
    profile[index % 7] += 0.01
    profile /= profile.sum()
    return SignatureDescriptors(
        bbox=(0, 0, 384, 192),
        aspect=4.0 + 0.05 * index,
        stroke_width=3.0 + 0.02 * index,
        density=0.2 + 0.001 * index,
        endpoint_density=0.05 + 0.0005 * index,
        junction_density=0.02 + 0.0003 * index,
        slant=0.1 + 0.01 * index,
        ink_area=1000 + index,
        skeleton_length=500 + index,
        h_profile=profile,
        v_profile=profile,
        hu=np.full(7, 0.01 * index),
    )


def _hand_rows(values: np.ndarray) -> np.ndarray:
    """Return a hand matrix whose ``hand_similarity_vector`` mean equals ``values``.

    Every similarity column carries the value directly and the one distance column
    carries ``1 - value``, so the inverted mean is exactly ``value``.
    """
    values = np.asarray(values, dtype=np.float64)
    rows = np.empty((values.size, len(FEATURE_NAMES)), dtype=np.float64)
    for column, name in enumerate(FEATURE_NAMES):
        rows[:, column] = 1.0 - values if name == DISTANCE_NAME else values
    return rows


def _records(*, count_per_signer: int = 7, forgeries: int = 2) -> list[SignatureRecord]:
    """Build a synthetic two-signer manifest (``a`` and ``b``)."""
    records: list[SignatureRecord] = []
    for signer in ("a", "b"):
        for index in range(count_per_signer):
            records.append(
                SignatureRecord(Path(f"{signer}_genuine_{index}.png"), signer, False)
            )
        for index in range(forgeries):
            records.append(
                SignatureRecord(Path(f"{signer}_forgery_{index}.png"), signer, True)
            )
    return records


def _fusion_fixture():
    """Return records, enrolment, embeddings, and descriptors for protocol tests."""
    records = _records(count_per_signer=6, forgeries=1)
    enrollment = enroll_and_build_queries(records, 4)
    rng = np.random.default_rng(6)
    embeddings = rng.normal(size=(len(records), 8))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    descriptors = [_descriptor(index) for index in range(len(records))]
    return records, enrollment, embeddings, descriptors


class FusionConfigTests(unittest.TestCase):
    def test_weighted_round_trip_preserves_fields(self) -> None:
        config = FusionConfig(
            type="weighted",
            deep_weight=0.6,
            hand_weight=0.4,
            threshold=0.73,
            metadata={"already_preprocessed": True},
        )
        restored = FusionConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)
        self.assertEqual(restored.threshold, 0.73)
        self.assertEqual(restored.metadata, {"already_preprocessed": True})

    def test_logistic_round_trip_preserves_coefficients(self) -> None:
        features = FEATURE_NAMES[:3]
        count = len(features) + 1
        config = FusionConfig(
            type="logistic",
            features=features,
            coef=tuple(float(value) for value in np.linspace(-1.0, 1.0, count)),
            mean=tuple([0.0] * count),
            scale=tuple([1.0] * count),
            intercept=0.25,
            dropped_features=FEATURE_NAMES[3:],
            threshold=0.5,
        )
        restored = FusionConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)
        self.assertEqual(restored.dropped_features, FEATURE_NAMES[3:])

    def test_save_and_load_round_trip_through_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "fusion.json"
            config = FusionConfig(
                type="weighted", deep_weight=0.55, hand_weight=0.45, threshold=0.61
            )
            saved = save_fusion_config(config, path)
            self.assertTrue(saved.exists())
            self.assertEqual(load_fusion_config(saved), config)
            payload = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 1)
            self.assertEqual(payload["threshold"], 0.61)

    def test_missing_threshold_stays_none(self) -> None:
        config = FusionConfig.from_dict(
            {
                "type": "weighted",
                "features": list(FEATURE_NAMES),
                "weights": {"deep": 0.7, "hand": 0.3},
            }
        )
        self.assertIsNone(config.threshold)
        self.assertEqual(config.dropped_features, ())

    def test_rejects_an_unknown_fusion_config_version(self) -> None:
        payload = FusionConfig(type="weighted").to_dict()
        payload["version"] = 999
        with self.assertRaisesRegex(ValueError, "unsupported fusion config version"):
            FusionConfig.from_dict(payload)

    def test_rejects_an_unknown_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported fusion type"):
            FusionConfig(type="average", deep_weight=0.5, hand_weight=0.5)

    def test_rejects_zero_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not both be zero"):
            FusionConfig(type="weighted", deep_weight=0.0, hand_weight=0.0)

    def test_rejects_a_negative_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            FusionConfig(type="weighted", deep_weight=-0.1, hand_weight=1.1)

    def test_rejects_an_empty_feature_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one handcrafted feature"):
            FusionConfig(type="weighted", features=())

    def test_rejects_logistic_config_with_mismatched_coefficients(self) -> None:
        count = len(FEATURE_NAMES) + 1
        with self.assertRaisesRegex(ValueError, "needs 10 coef values"):
            FusionConfig(
                type="logistic",
                coef=(0.1,),
                mean=tuple([0.0] * count),
                scale=tuple([1.0] * count),
            )

    def test_rejects_logistic_config_with_mismatched_scale(self) -> None:
        count = len(FEATURE_NAMES) + 1
        with self.assertRaisesRegex(ValueError, "needs 10 scale values"):
            FusionConfig(
                type="logistic",
                coef=tuple([0.0] * count),
                mean=tuple([0.0] * count),
                scale=(1.0,),
            )



class WeightedScorerTests(unittest.TestCase):
    def _config(self, deep: float = 0.7, hand: float = 0.3) -> FusionConfig:
        return FusionConfig(type="weighted", deep_weight=deep, hand_weight=hand)

    def test_normalises_weights_on_construction(self) -> None:
        scorer = FusionScorer(self._config(deep=2.0, hand=1.0))
        hand = {name: 0.5 for name in FEATURE_NAMES}
        # 2:1 normalises to 2/3 and 1/3, so the output stays in [0, 1].
        self.assertAlmostEqual(
            scorer.score(1.0, hand), 2.0 / 3.0 + (1.0 / 3.0) * 0.5, places=12
        )

    def test_hand_mean_inverts_the_slant_distance(self) -> None:
        scorer = FusionScorer(self._config())
        hand = {name: 0.4 for name in FEATURE_NAMES}
        expected = (0.4 * 8 + (1.0 - 0.4)) / 9.0
        self.assertAlmostEqual(scorer.hand_mean(hand), expected, places=12)

    def test_missing_features_fall_back_to_neutral(self) -> None:
        # Hand-only config: similarity features default to "most similar" (1.0) and the
        # distance feature to "most similar" (0.0 before inversion), so the mean is 8/9.
        scorer = FusionScorer(self._config(deep=0.0, hand=1.0))
        self.assertAlmostEqual(scorer.hand_mean({}), 8.0 / 9.0, places=12)
        self.assertAlmostEqual(scorer.score(0.0, {}), 8.0 / 9.0, places=12)

    def test_missing_distance_feature_is_not_treated_as_most_dissimilar(self) -> None:
        # Regression guard: falling back to 1.0 *before* the flip would make an absent
        # slant measurement the most dissimilar possible value.
        scorer = FusionScorer(
            FusionConfig(
                type="weighted",
                features=(DISTANCE_NAME,),
                deep_weight=0.0,
                hand_weight=1.0,
            )
        )
        self.assertAlmostEqual(scorer.hand_mean({}), 0.0, places=12)

    def test_score_matrix_matches_score(self) -> None:
        scorer = FusionScorer(self._config(deep=0.65, hand=0.35))
        rng = np.random.default_rng(0)
        hand = rng.random((5, len(FEATURE_NAMES)))
        deep = rng.random(5)
        rows = [dict(zip(FEATURE_NAMES, row)) for row in hand]
        expected = np.array(
            [scorer.score(value, row) for value, row in zip(deep, rows)], dtype=np.float64
        )
        np.testing.assert_allclose(
            scorer.score_matrix(deep, hand), expected, rtol=0, atol=1e-12
        )

    def test_score_pairs_matches_score(self) -> None:
        scorer = FusionScorer(self._config())
        rng = np.random.default_rng(1)
        hand = rng.random((4, len(FEATURE_NAMES)))
        deep = rng.random(4)
        rows = [dict(zip(FEATURE_NAMES, row)) for row in hand]
        expected = np.array([scorer.score(value, row) for value, row in zip(deep, rows)])
        np.testing.assert_allclose(scorer.score_pairs(deep, rows), expected, rtol=0)

    def test_score_matrix_rejects_a_wrong_column_count(self) -> None:
        scorer = FusionScorer(self._config())
        with self.assertRaisesRegex(ValueError, "columns in features order"):
            scorer.score_matrix(np.zeros(2), np.zeros((2, 3)))

    def test_score_matrix_rejects_a_length_mismatch(self) -> None:
        scorer = FusionScorer(self._config())
        with self.assertRaisesRegex(ValueError, "equal length"):
            scorer.score_matrix(np.zeros(3), np.zeros((2, len(FEATURE_NAMES))))

    def test_score_pairs_rejects_a_length_mismatch(self) -> None:
        scorer = FusionScorer(self._config())
        with self.assertRaisesRegex(ValueError, "equal length"):
            scorer.score_pairs(np.zeros(2), [{}])



class LogisticScorerTests(unittest.TestCase):
    def _config(self) -> FusionConfig:
        count = len(FEATURE_NAMES) + 1
        coef = [0.0] * count
        coef[0] = 2.0
        return FusionConfig(
            type="logistic",
            coef=tuple(coef),
            mean=tuple([0.0] * count),
            scale=tuple([1.0] * count),
            intercept=-0.5,
        )

    def test_score_matches_a_manual_sigmoid(self) -> None:
        scorer = FusionScorer(self._config())
        hand = {name: 0.5 for name in FEATURE_NAMES}
        # Only the deep coefficient is non-zero, so the hand values drop out entirely.
        expected = 1.0 / (1.0 + np.exp(-(2.0 * 0.75 - 0.5)))
        self.assertAlmostEqual(scorer.score(0.75, hand), expected, places=12)

    def test_missing_feature_falls_back_to_its_training_mean(self) -> None:
        count = len(FEATURE_NAMES) + 1
        mean = [0.0] * count
        mean[1] = 0.25  # training mean of the first handcrafted feature
        coef = [0.0] * count
        coef[1] = 1.0
        scorer = FusionScorer(
            FusionConfig(
                type="logistic",
                coef=tuple(coef),
                mean=tuple(mean),
                scale=tuple([1.0] * count),
                intercept=0.0,
            )
        )
        absent: dict[str, float] = {}
        self.assertAlmostEqual(
            scorer.score(0.0, {FEATURE_NAMES[0]: 0.25}), scorer.score(0.0, absent)
        )
        self.assertGreater(scorer.score(0.0, {FEATURE_NAMES[0]: 0.75}), 0.5)

    def test_zero_scale_is_replaced_by_one(self) -> None:
        count = len(FEATURE_NAMES) + 1
        coef = [0.0] * count
        coef[0] = 1.0
        scorer = FusionScorer(
            FusionConfig(
                type="logistic",
                coef=tuple(coef),
                mean=tuple([0.0] * count),
                scale=tuple([0.0] * count),
            )
        )
        self.assertTrue(np.isfinite(scorer.score(0.5, {})))

    def test_score_matrix_matches_score(self) -> None:
        scorer = FusionScorer(self._config())
        rng = np.random.default_rng(2)
        hand = rng.random((4, len(FEATURE_NAMES)))
        deep = rng.random(4)
        expected = np.array(
            [
                scorer.score(value, dict(zip(FEATURE_NAMES, row)))
                for value, row in zip(deep, hand)
            ]
        )
        np.testing.assert_allclose(
            scorer.score_matrix(deep, hand), expected, rtol=0, atol=1e-12
        )


class ScoreRowsTests(unittest.TestCase):
    def test_gathers_configured_columns_by_name(self) -> None:
        features = (FEATURE_NAMES[2], FEATURE_NAMES[0])
        scorer = FusionScorer(
            FusionConfig(
                type="weighted", features=features, deep_weight=0.5, hand_weight=0.5
            )
        )
        rng = np.random.default_rng(3)
        hand = rng.random((3, len(FEATURE_NAMES)))
        deep = rng.random(3)
        columns = [FEATURE_NAMES.index(name) for name in features]
        expected = scorer.score_matrix(deep, hand[:, columns])
        np.testing.assert_allclose(
            scorer.score_rows(deep, hand), expected, rtol=0, atol=1e-12
        )

    def test_works_when_the_config_dropped_features(self) -> None:
        # A logistic config may list only the features the collinearity guard kept, while
        # the fitting matrix still carries all nine columns.
        features = (FEATURE_NAMES[0], FEATURE_NAMES[5])
        count = len(features) + 1
        coef = [0.0] * count
        coef[0] = 1.5
        scorer = FusionScorer(
            FusionConfig(
                type="logistic",
                features=features,
                coef=tuple(coef),
                mean=tuple([0.0] * count),
                scale=tuple([1.0] * count),
            )
        )
        rng = np.random.default_rng(4)
        hand = rng.random((3, len(FEATURE_NAMES)))
        deep = rng.random(3)
        columns = [FEATURE_NAMES.index(name) for name in features]
        np.testing.assert_allclose(
            scorer.score_rows(deep, hand),
            scorer.score_matrix(deep, hand[:, columns]),
            rtol=0,
            atol=1e-12,
        )

    def test_rejects_a_matrix_that_does_not_match_the_name_list(self) -> None:
        scorer = FusionScorer(FusionConfig(type="weighted"))
        with self.assertRaisesRegex(ValueError, "does not match the feature name list"):
            scorer.score_rows(np.zeros(2), np.zeros((2, 3)))


def _naive_eer_threshold(
    genuine: np.ndarray, negatives: np.ndarray
) -> tuple[float, float]:
    """Reference single-pass EER sweep, kept as ground truth for the vectorised one."""
    genuine_array = np.asarray(genuine, dtype=np.float64)
    negative_array = np.asarray(negatives, dtype=np.float64)
    best: tuple[float, float, float] | None = None
    for candidate in np.unique(np.concatenate([genuine_array, negative_array])):
        far = float((negative_array >= candidate).mean())
        frr = float((genuine_array < candidate).mean())
        gap = abs(far - frr)
        if best is None or gap < best[0]:
            best = (gap, float(candidate), (far + frr) / 2.0)
    assert best is not None
    return best[1], best[2]


class MetricsTests(unittest.TestCase):
    def test_rates_at_threshold_counts_the_threshold_as_accepted(self) -> None:
        far, frr = rates_at_threshold([0.85, 0.80], [0.90, 0.10], 0.85)
        self.assertAlmostEqual(far, 0.5)  # 0.90 accepted, 0.10 rejected
        self.assertAlmostEqual(frr, 0.5)  # 0.85 accepted, 0.80 rejected

    def test_rates_at_threshold_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            rates_at_threshold([], [0.5], 0.5)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            rates_at_threshold([0.5], [], 0.5)

    def test_eer_is_zero_for_perfectly_separated_scores(self) -> None:
        threshold, equal_error = eer_threshold([0.90, 0.95], [0.10, 0.20])
        self.assertAlmostEqual(threshold, 0.90)
        self.assertAlmostEqual(equal_error, 0.0)

    def test_eer_threshold_matches_a_naive_sweep(self) -> None:
        # The vectorised searchsorted implementation must reproduce the original loop.
        rng = np.random.default_rng(11)
        genuine = rng.normal(0.7, 0.1, size=60)
        negatives = rng.normal(0.3, 0.1, size=200)
        expected_threshold, expected_eer = _naive_eer_threshold(genuine, negatives)
        threshold, equal_error = eer_threshold(genuine, negatives)
        self.assertAlmostEqual(threshold, expected_threshold, places=12)
        self.assertAlmostEqual(equal_error, expected_eer, places=12)


class QueryScoresFromRowsTests(unittest.TestCase):
    def test_averages_contiguous_rows_back_to_queries(self) -> None:
        row_scores = np.arange(9, dtype=np.float64)
        query_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        kinds = np.array(
            [
                "genuine",
                "genuine",
                "genuine",
                "skilled",
                "skilled",
                "random",
                "random",
                "random",
                "random",
            ],
            dtype=object,
        )
        scores, query_kinds = _query_scores_from_rows(row_scores, query_ids, kinds)
        np.testing.assert_allclose(scores, [1.0, 3.5, 6.5])
        np.testing.assert_array_equal(query_kinds, ["genuine", "skilled", "random"])

    def test_returns_one_score_per_enrolled_query(self) -> None:
        _, enrollment, _, _ = _fusion_fixture()
        references = 4
        row_scores = np.arange(references * len(enrollment.queries), dtype=np.float64)
        query_ids = np.repeat(np.arange(len(enrollment.queries)), references)
        kinds = np.repeat(
            [query.kind for query in enrollment.queries], references
        )
        scores, query_kinds = _query_scores_from_rows(row_scores, query_ids, kinds)
        self.assertEqual(scores.size, len(enrollment.queries))
        self.assertEqual(query_kinds.size, len(enrollment.queries))


class CollinearityTests(unittest.TestCase):
    def test_drops_a_duplicated_column_and_keeps_the_earlier_one(self) -> None:
        rng = np.random.default_rng(12)
        column = rng.random(40)
        matrix = np.column_stack([column, column, rng.random(40)])
        kept, dropped = _drop_collinear(matrix, ("first", "duplicate", "third"))
        self.assertEqual(kept, [0, 2])
        self.assertEqual(dropped, ["duplicate"])

    def test_keeps_every_column_when_there_are_too_few_rows_to_correlate(self) -> None:
        kept, dropped = _drop_collinear(np.ones((1, 3)), ("a", "b", "c"))
        self.assertEqual(kept, [0, 1, 2])
        self.assertEqual(dropped, [])

    def test_rejects_a_matrix_that_does_not_match_the_name_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the feature name list"):
            _drop_collinear(np.zeros((4, 3)), FEATURE_NAMES)

    def test_hand_similarity_matrix_inverts_only_the_distance_column(self) -> None:
        matrix = np.ones((2, len(FEATURE_NAMES)), dtype=np.float64)
        similarity = _hand_similarity_matrix(matrix, FEATURE_NAMES)
        distance = FEATURE_NAMES.index(DISTANCE_NAME)
        self.assertAlmostEqual(float(similarity[0, distance]), 0.0)
        for index in range(len(FEATURE_NAMES)):
            if index != distance:
                self.assertAlmostEqual(float(similarity[0, index]), 1.0)
        # The caller's matrix is never mutated.
        self.assertTrue(np.all(matrix == 1.0))



class DropCollinearTests(unittest.TestCase):
    def test_drops_a_later_duplicate_of_a_correlated_feature(self) -> None:
        base = np.linspace(0.0, 1.0, 20)
        independent = np.array([0.0, 1.0] * 10, dtype=np.float64)
        matrix = np.column_stack([base, base * 2.0 + 0.1, independent])
        kept, dropped = _drop_collinear(matrix, FEATURE_NAMES[:3], threshold=0.95)
        self.assertEqual(kept, [0, 2])
        self.assertEqual(dropped, [FEATURE_NAMES[1]])

    def test_keeps_the_first_listed_of_an_anti_correlated_pair(self) -> None:
        base = np.linspace(0.0, 1.0, 20)
        independent = np.array([0.0, 1.0] * 10, dtype=np.float64)
        matrix = np.column_stack([base, base[::-1], independent])
        kept, dropped = _drop_collinear(matrix, FEATURE_NAMES[:3], threshold=0.95)
        self.assertEqual(kept, [0, 2])
        self.assertEqual(dropped, [FEATURE_NAMES[1]])

    def test_is_deterministic(self) -> None:
        rng = np.random.default_rng(5)
        base = rng.normal(size=30)
        matrix = np.column_stack([base, base + 1e-9, rng.normal(size=30)])
        first = _drop_collinear(matrix, FEATURE_NAMES[:3], threshold=0.95)
        second = _drop_collinear(matrix, FEATURE_NAMES[:3], threshold=0.95)
        self.assertEqual(first, second)

    def test_keeps_independent_columns(self) -> None:
        rng = np.random.default_rng(6)
        matrix = rng.normal(size=(50, 3))
        kept, dropped = _drop_collinear(matrix, FEATURE_NAMES[:3])
        self.assertEqual(kept, [0, 1, 2])
        self.assertEqual(dropped, [])

    def test_handles_a_single_row(self) -> None:
        kept, dropped = _drop_collinear(np.zeros((1, 3)), FEATURE_NAMES[:3])
        self.assertEqual(kept, [0, 1, 2])
        self.assertEqual(dropped, [])

    def test_rejects_a_matrix_that_does_not_match_the_name_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the feature name list"):
            _drop_collinear(np.zeros((2, 3)), FEATURE_NAMES[:2])


class HandSimilarityMatrixTests(unittest.TestCase):
    def test_inverts_only_the_distance_column(self) -> None:
        hand = np.full((2, len(FEATURE_NAMES)), 0.25)
        similarity = _hand_similarity_matrix(hand, FEATURE_NAMES)
        index = FEATURE_NAMES.index(DISTANCE_NAME)
        self.assertAlmostEqual(float(similarity[0, index]), 0.75, places=12)
        for column, name in enumerate(FEATURE_NAMES):
            if name != DISTANCE_NAME:
                self.assertAlmostEqual(float(similarity[0, column]), 0.25, places=12)
        # The input stays untouched so callers can keep reusing the raw matrix.
        self.assertAlmostEqual(float(hand[0, index]), 0.25, places=12)


class FitWeightedFusionTests(unittest.TestCase):
    def test_tie_break_keeps_both_modalities(self) -> None:
        # Both modalities separate the split perfectly, so every weight in the grid ties
        # at EER 0. The tie-break must land mid-grid instead of silently dropping the
        # deep score by returning the first minimum.
        deep = np.array([0.9, 0.8, 0.1, 0.05])
        hand = _hand_rows(deep)
        labels = np.array([1, 1, 0, 0])
        config, metadata = fit_weighted_fusion(deep, hand, labels)
        self.assertEqual(config.type, "weighted")
        self.assertAlmostEqual(config.deep_weight, 0.5, places=6)
        self.assertAlmostEqual(config.hand_weight, 0.5, places=6)
        self.assertAlmostEqual(config.deep_weight + config.hand_weight, 1.0, places=12)
        self.assertEqual(metadata["fit_eer"], 0.0)
        self.assertAlmostEqual(metadata["deep_weight"], config.deep_weight, places=12)
        self.assertEqual(len(metadata["grid"]), 21)
        self.assertEqual(config.features, FEATURE_NAMES)

    def test_pushes_the_weight_up_when_hand_features_mislead(self) -> None:
        # The hand term is anti-correlated with the labels, so a low deep weight cannot
        # separate the split at all. The optimum is the smallest deep weight that
        # overcomes the misleading hand signal, not the middle of the grid.
        deep = np.array([0.9, 0.8, 0.1, 0.05])
        hand = _hand_rows(np.array([0.05, 0.15, 0.95, 0.85]))
        labels = np.array([1, 1, 0, 0])
        config, metadata = fit_weighted_fusion(deep, hand, labels)
        self.assertGreater(config.deep_weight, 0.5)
        self.assertLess(config.deep_weight, 0.9)
        self.assertEqual(metadata["fit_eer"], 0.0)
        earlier = {entry["deep_weight"]: entry["eer"] for entry in metadata["grid"]}
        self.assertGreater(earlier[0.5], 0.0)

    def test_grid_is_recorded_in_the_config_metadata(self) -> None:
        deep = np.array([0.9, 0.7, 0.2, 0.1])
        hand = _hand_rows(np.array([0.8, 0.6, 0.3, 0.2]))
        labels = np.array([1, 1, 0, 0])
        config, _ = fit_weighted_fusion(deep, hand, labels)
        grid = config.metadata["grid"]
        self.assertEqual(len(grid), 21)
        self.assertEqual(set(grid[0]), {"deep_weight", "eer"})

    def test_rejects_a_single_class_split(self) -> None:
        deep = np.array([0.5, 0.4])
        with self.assertRaisesRegex(ValueError, "both genuine and negative"):
            fit_weighted_fusion(deep, _hand_rows(deep), np.array([1, 1]))

    def test_rejects_a_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            fit_weighted_fusion(np.array([0.5]), _hand_rows(np.array([0.5, 0.4])), np.array([1, 0]))



class FitLogisticFusionTests(unittest.TestCase):
    def _split(self, size: int = 40, seed: int = 7):
        rng = np.random.default_rng(seed)
        genuine_hand = rng.normal(0.8, 0.05, size=(size, len(FEATURE_NAMES)))
        negative_hand = rng.normal(0.2, 0.05, size=(size, len(FEATURE_NAMES)))
        hand = np.vstack([genuine_hand, negative_hand])
        deep = np.concatenate([np.full(size, 0.8), np.full(size, 0.2)])
        labels = np.concatenate([np.ones(size), np.zeros(size)]).astype(int)
        return rng, hand, deep, labels

    def test_fits_a_portable_config_that_ranks_genuine_higher(self) -> None:
        rng, hand, deep, labels = self._split()
        config, metadata = fit_logistic_fusion(deep, hand, labels)
        self.assertEqual(config.type, "logistic")
        self.assertEqual(len(config.coef), len(config.features) + 1)
        self.assertEqual(len(config.mean), len(config.features) + 1)
        self.assertEqual(len(config.scale), len(config.features) + 1)
        self.assertEqual(len(config.features) + len(config.dropped_features), len(FEATURE_NAMES))
        self.assertEqual(config.metadata["n_samples"], 80)
        self.assertEqual(config.metadata["positives"], 40)
        self.assertEqual(metadata["dropped_features"], list(config.dropped_features))

        scorer = FusionScorer(config)
        genuine_score = scorer.score(0.8, dict(zip(FEATURE_NAMES, hand[0])))
        negative_score = scorer.score(0.2, dict(zip(FEATURE_NAMES, hand[len(labels) // 2])))
        self.assertGreater(genuine_score, negative_score)
        for value in (genuine_score, negative_score):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_records_dropped_collinear_features(self) -> None:
        rng = np.random.default_rng(8)
        base = rng.normal(0.5, 0.1, size=(60, 1))
        rest = rng.normal(0.5, 0.1, size=(60, len(FEATURE_NAMES) - 2))
        hand = np.hstack([base, base + 1e-9, rest])
        deep = np.concatenate([np.full(30, 0.8), np.full(30, 0.2)])
        labels = np.concatenate([np.ones(30), np.zeros(30)]).astype(int)
        config, metadata = fit_logistic_fusion(deep, hand, labels)
        self.assertIn(FEATURE_NAMES[1], config.dropped_features)
        self.assertNotIn(FEATURE_NAMES[1], config.features)
        self.assertIn(FEATURE_NAMES[0], config.features)
        self.assertEqual(len(config.features) + len(config.dropped_features), len(FEATURE_NAMES))
        self.assertEqual(metadata["dropped_features"], list(config.dropped_features))

    def test_can_disable_the_collinearity_guard(self) -> None:
        rng = np.random.default_rng(9)
        base = rng.normal(0.5, 0.1, size=(60, 1))
        rest = rng.normal(0.5, 0.1, size=(60, len(FEATURE_NAMES) - 2))
        hand = np.hstack([base, base + 1e-9, rest])
        deep = np.concatenate([np.full(30, 0.8), np.full(30, 0.2)])
        labels = np.concatenate([np.ones(30), np.zeros(30)]).astype(int)
        config, _ = fit_logistic_fusion(deep, hand, labels, drop_correlated=None)
        self.assertEqual(config.dropped_features, ())
        self.assertEqual(config.features, FEATURE_NAMES)
        self.assertIsNone(config.metadata["drop_correlated"])

    def test_rejects_a_single_class_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "both genuine and negative"):
            fit_logistic_fusion(
                np.zeros(3),
                np.zeros((3, len(FEATURE_NAMES))),
                np.ones(3, dtype=int),
            )

    def test_rejects_a_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            fit_logistic_fusion(
                np.zeros(2),
                np.zeros((3, len(FEATURE_NAMES))),
                np.array([1, 0, 1]),
            )



class EnrollmentTests(unittest.TestCase):
    def test_enrols_the_first_references_and_lists_queries(self) -> None:
        records = _records()
        enrollment = enroll_and_build_queries(records, 5)
        self.assertEqual(enrollment.references["a"], [0, 1, 2, 3, 4])
        self.assertEqual(enrollment.references["b"], [9, 10, 11, 12, 13])
        kinds = [query.kind for query in enrollment.queries]
        self.assertEqual(kinds.count("genuine"), 4)
        self.assertEqual(kinds.count("skilled"), 4)
        self.assertEqual(kinds.count("random"), 4)
        genuine = {query.index for query in enrollment.queries if query.kind == "genuine"}
        skilled = {query.index for query in enrollment.queries if query.kind == "skilled"}
        random_queries = {query.index for query in enrollment.queries if query.kind == "random"}
        self.assertEqual(genuine, {5, 6, 14, 15})
        self.assertEqual(skilled, {7, 8, 16, 17})
        # Random queries are another signer's leftover genuine signatures.
        self.assertEqual(random_queries, {5, 6, 14, 15})

    def test_query_index_is_the_position_in_the_query_list(self) -> None:
        enrollment = enroll_and_build_queries(_records(), 5)
        self.assertEqual(
            [query.query_index for query in enrollment.queries],
            list(range(len(enrollment.queries))),
        )

    def test_every_query_targets_its_own_signer(self) -> None:
        records = _records()
        enrollment = enroll_and_build_queries(records, 5)
        for query in enrollment.queries:
            if query.kind == "random":
                self.assertNotEqual(records[query.index].signer_id, query.target_signer)
            else:
                self.assertEqual(records[query.index].signer_id, query.target_signer)

    def test_caps_only_the_random_queries(self) -> None:
        enrollment = enroll_and_build_queries(_records(), 5, max_random_queries=2)
        kinds = [query.kind for query in enrollment.queries]
        self.assertEqual(kinds.count("random"), 2)
        self.assertEqual(kinds.count("genuine"), 4)
        self.assertEqual(kinds.count("skilled"), 4)
        self.assertEqual(
            [query.query_index for query in enrollment.queries],
            list(range(len(enrollment.queries))),
        )

    def test_random_cap_is_deterministic(self) -> None:
        first = enroll_and_build_queries(_records(), 5, max_random_queries=2)
        second = enroll_and_build_queries(_records(), 5, max_random_queries=2)
        self.assertEqual(
            [query.index for query in first.queries],
            [query.index for query in second.queries],
        )

    def test_enrols_exactly_the_reference_count_without_genuine_queries(self) -> None:
        enrollment = enroll_and_build_queries(_records(), 7)
        self.assertEqual(len(enrollment.references), 2)
        self.assertEqual(enrollment.skipped_signers, ())
        self.assertFalse(any(q.kind == "genuine" for q in enrollment.queries))

    def test_skips_signers_with_too_few_genuine_images(self) -> None:
        records = _records(count_per_signer=7, forgeries=1)
        records.append(SignatureRecord(Path("c_0.png"), "c", False))
        enrollment = enroll_and_build_queries(records, 5)
        self.assertEqual(enrollment.skipped_signers, ("c",))
        self.assertEqual(len(enrollment.references), 2)

    def test_raises_when_no_signer_has_enough_genuine_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "no signer has at least"):
            enroll_and_build_queries(_records(), 8)



class BuildPairMatrixTests(unittest.TestCase):
    def test_emits_one_row_per_reference_pair(self) -> None:
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        matrix = build_pair_matrix(embeddings, descriptors, enrollment)
        expected_rows = sum(
            len(enrollment.references[query.target_signer]) for query in enrollment.queries
        )
        self.assertEqual(len(matrix), expected_rows)
        self.assertEqual(matrix.hand.shape, (expected_rows, len(FEATURE_NAMES)))
        for field_name in ("deep", "labels", "signers", "kinds", "query_ids"):
            self.assertEqual(getattr(matrix, field_name).shape, (expected_rows,))
        # Labels follow the query kind: only genuine queries are positives.
        np.testing.assert_array_equal(
            matrix.labels, (matrix.kinds == "genuine").astype(np.int64)
        )
        # Every query contributes exactly len(references) consecutive rows, which is what
        # lets the cross-validation regroup rows back into per-query scores.
        counts = np.bincount(matrix.query_ids)
        self.assertEqual(len(counts), len(enrollment.queries))
        self.assertTrue(np.all(counts == 4))

    def test_deep_rows_are_cosine_scores_to_each_reference(self) -> None:
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        matrix = build_pair_matrix(embeddings, descriptors, enrollment)
        query = enrollment.queries[0]
        references = enrollment.references[query.target_signer]
        expected = embeddings[references] @ embeddings[query.index]
        np.testing.assert_allclose(matrix.deep[:4], expected, rtol=0, atol=1e-12)

    def test_hand_rows_match_pair_features(self) -> None:
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        matrix = build_pair_matrix(embeddings, descriptors, enrollment)
        query = enrollment.queries[0]
        references = enrollment.references[query.target_signer]
        features = pair_features(descriptors[references[0]], descriptors[query.index])
        np.testing.assert_allclose(
            matrix.hand[0],
            [features[name] for name in FEATURE_NAMES],
            rtol=0,
            atol=1e-12,
        )

    def test_rows_carry_the_target_signer(self) -> None:
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        matrix = build_pair_matrix(embeddings, descriptors, enrollment)
        expected = np.concatenate(
            [
                np.repeat(query.target_signer, len(enrollment.references[query.target_signer]))
                for query in enrollment.queries
            ]
        )
        np.testing.assert_array_equal(matrix.signers, expected)

    def test_subset_preserves_column_and_row_order(self) -> None:
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        matrix = build_pair_matrix(embeddings, descriptors, enrollment)
        indices = np.array([3, 1])
        subset = matrix.subset(indices)
        np.testing.assert_array_equal(subset.deep, matrix.deep[indices])
        np.testing.assert_array_equal(subset.hand, matrix.hand[indices])
        np.testing.assert_array_equal(subset.signers, matrix.signers[indices])
        np.testing.assert_array_equal(subset.query_ids, matrix.query_ids[indices])
        self.assertEqual(len(subset), 2)


class QueryScoreAggregationTests(unittest.TestCase):
    def test_averages_rows_back_to_per_query_scores(self) -> None:
        rows = np.array([0.2, 0.4, 0.9, 0.1, 0.5, 0.7])
        query_ids = np.array([0, 0, 1, 2, 2, 2])
        kinds = np.array(["genuine", "genuine", "skilled", "random", "random", "random"])
        scores, query_kinds = _query_scores_from_rows(rows, query_ids, kinds)
        np.testing.assert_allclose(
            scores, [0.3, 0.9, (0.1 + 0.5 + 0.7) / 3.0], rtol=0, atol=1e-12
        )
        np.testing.assert_array_equal(query_kinds, ["genuine", "skilled", "random"])




class ScoreQueriesTests(unittest.TestCase):
    def test_returns_deep_scores_by_kind(self) -> None:
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        scores = score_queries(embeddings, descriptors, enrollment)
        self.assertEqual(set(scores), {"deep_genuine", "deep_skilled", "deep_random"})
        for kind in ("genuine", "skilled", "random"):
            expected = sum(1 for query in enrollment.queries if query.kind == kind)
            self.assertEqual(scores[f"deep_{kind}"].size, expected)

    def test_deep_scores_average_the_references(self) -> None:
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        scores = score_queries(embeddings, descriptors, enrollment)
        counters = {"genuine": 0, "skilled": 0, "random": 0}
        for query in enrollment.queries:
            references = enrollment.references[query.target_signer]
            expected = float((embeddings[references] @ embeddings[query.index]).mean())
            value = scores[f"deep_{query.kind}"][counters[query.kind]]
            counters[query.kind] += 1
            self.assertAlmostEqual(float(value), expected, places=12)

    def test_fused_scores_average_per_reference_pairs(self) -> None:
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        scorer = FusionScorer(
            FusionConfig(type="weighted", deep_weight=0.6, hand_weight=0.4)
        )
        scores = score_queries(embeddings, descriptors, enrollment, scorer)
        self.assertIn("fused_genuine", scores)
        counters = {"genuine": 0, "skilled": 0, "random": 0}
        for query in enrollment.queries:
            references = enrollment.references[query.target_signer]
            deep_scores = embeddings[references] @ embeddings[query.index]
            expected = float(
                np.mean(
                    [
                        scorer.score(
                            float(deep_scores[position]),
                            pair_features(descriptors[reference], descriptors[query.index]),
                        )
                        for position, reference in enumerate(references)
                    ]
                )
            )
            value = float(scores[f"fused_{query.kind}"][counters[query.kind]])
            counters[query.kind] += 1
            self.assertAlmostEqual(value, expected, places=12)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_fused_scores_match_pair_matrix_aggregation(self) -> None:
        # The explicit per-pair rows and the per-query evaluation path must agree, which
        # is what lets the fitter calibrate a threshold in the space evaluation uses.
        _, enrollment, embeddings, descriptors = _fusion_fixture()
        scorer = FusionScorer(
            FusionConfig(type="weighted", deep_weight=0.4, hand_weight=0.6)
        )
        matrix = build_pair_matrix(embeddings, descriptors, enrollment)
        row_scores = scorer.score_matrix(matrix.deep, matrix.hand)
        per_query, query_kinds = _query_scores_from_rows(
            row_scores, matrix.query_ids, matrix.kinds
        )
        scores = score_queries(embeddings, descriptors, enrollment, scorer)
        for kind in ("genuine", "skilled", "random"):
            np.testing.assert_allclose(
                scores[f"fused_{kind}"],
                per_query[query_kinds == kind],
                rtol=0,
                atol=1e-12,
            )


class MetricsTests(unittest.TestCase):
    def test_eer_threshold_matches_the_naive_loop(self) -> None:
        rng = np.random.default_rng(11)
        genuine = rng.normal(0.7, 0.1, size=60)
        negatives = rng.normal(0.4, 0.1, size=90)
        threshold, equal_error_rate = eer_threshold(genuine, negatives)

        candidates = np.unique(np.concatenate([genuine, negatives]))
        best_gap = np.inf
        best_threshold = None
        best_eer = None
        for candidate in candidates:
            false_acceptance = float((negatives >= candidate).mean())
            false_rejection = float((genuine < candidate).mean())
            gap = abs(false_acceptance - false_rejection)
            if gap < best_gap:
                best_gap = gap
                best_threshold = float(candidate)
                best_eer = (false_acceptance + false_rejection) / 2.0

        self.assertAlmostEqual(threshold, best_threshold, places=12)
        self.assertAlmostEqual(equal_error_rate, best_eer, places=12)

    def test_rates_at_threshold_accepts_at_the_threshold(self) -> None:
        false_acceptance, false_rejection = rates_at_threshold([0.4, 0.9], [0.5, 0.1], 0.5)
        self.assertAlmostEqual(false_acceptance, 0.5)
        self.assertAlmostEqual(false_rejection, 0.5)

    def test_rates_at_threshold_rejects_empty_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            rates_at_threshold([], [0.5], 0.5)

    def test_eer_threshold_rejects_empty_arrays(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            eer_threshold([0.5], [])


if __name__ == "__main__":
    unittest.main()

