"""Focused tests for validation-EER checkpoint selection.

Selection is the part of training that decides which weights survive, so these tests pin
down three things: the validation EER is computed with exactly the evaluation protocol,
it ranks a better-separated embedding space lower, and the comparison rule cannot be
fooled into overwriting a good checkpoint.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scoring.metrics import eer_threshold
from training import evaluate, train_siamese
from training.protocol import skilled_eer
from training.stage3_metric_learning import SignatureRecord

REFERENCES = 1


def _records() -> list[SignatureRecord]:
    """Three signers, each with two genuine signatures and one skilled forgery."""
    return [
        SignatureRecord(
            image_path=Path(f"{signer}/{kind}.png"),
            signer_id=signer,
            is_forgery=kind == "forgery",
            dataset="DS",
        )
        for signer in ("a", "b", "c")
        for kind in ("genuine_1", "genuine_2", "forgery")
    ]


def _embeddings(forgeries_placed_on_their_signer: bool) -> np.ndarray:
    """Return unit-norm 3-D embeddings in record order.

    Genuine samples of a signer share a direction. A forgery sits in a different
    signer's direction (well separated) or on its own signer (inseparable).
    """
    genuine = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [-1.0, 0.0, 0.0]}
    forgery = (
        {"a": genuine["a"], "b": genuine["b"], "c": genuine["c"]}
        if forgeries_placed_on_their_signer
        else {"a": genuine["b"], "b": [0.0, 0.0, 1.0], "c": [0.0, 0.0, 1.0]}
    )
    rows = []
    for signer in ("a", "b", "c"):
        rows.extend([genuine[signer], genuine[signer], forgery[signer]])
    return np.asarray(rows, dtype=np.float64)


def _select(embeddings: np.ndarray):
    """Run the selection metric with the embedding pass patched out."""
    records = _records()
    with mock.patch.object(
        train_siamese, "embed_records", return_value=embeddings
    ) as patched:
        value = train_siamese._validation_skilled_eer(
            None,
            records,
            REFERENCES,
            "cpu",
            already_preprocessed=True,
            image_height=256,
            image_width=512,
            batch_size=8,
        )
    return value, patched, records


class ValidationEerTests(unittest.TestCase):
    def test_matches_the_evaluation_protocol(self) -> None:
        embeddings = _embeddings(forgeries_placed_on_their_signer=False)
        value, _, records = _select(embeddings)

        # Selection goes through the shared protocol...
        expected = skilled_eer(records, embeddings, REFERENCES)
        # ...and evaluation scores the same records through its own split helper.
        scores, _ = evaluate._evaluate_split(
            records, embeddings, [], REFERENCES, None, 0
        )
        evaluated = eer_threshold(scores["deep_genuine"], scores["deep_skilled"])[1]

        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(value, evaluated)
        self.assertAlmostEqual(value, 0.0)

    def test_a_worse_embedding_space_gets_a_higher_eer(self) -> None:
        separated, _, _ = _select(_embeddings(forgeries_placed_on_their_signer=False))
        inseparable, _, _ = _select(_embeddings(forgeries_placed_on_their_signer=True))
        self.assertLess(separated, inseparable)

    def test_selection_uses_the_configured_reference_count(self) -> None:
        embeddings = _embeddings(forgeries_placed_on_their_signer=False)
        records = _records()
        with mock.patch.object(
            train_siamese, "embed_records", return_value=embeddings
        ), mock.patch.object(train_siamese, "skilled_eer", return_value=0.25) as metric:
            value = train_siamese._validation_skilled_eer(
                None,
                records,
                REFERENCES,
                "cpu",
                already_preprocessed=True,
                image_height=256,
                image_width=512,
                batch_size=8,
            )
        self.assertEqual(value, 0.25)
        metric.assert_called_once()
        call_args, call_kwargs = metric.call_args
        self.assertEqual(call_args[2], REFERENCES)
        self.assertEqual(call_kwargs["reference_selection"], "first")

    def test_passes_the_preprocessing_settings_to_the_embedding_pass(self) -> None:
        embeddings = _embeddings(forgeries_placed_on_their_signer=False)
        _, patched, _ = _select(embeddings)
        arguments = patched.call_args.kwargs
        self.assertEqual(arguments["image_height"], 256)
        self.assertEqual(arguments["image_width"], 512)
        self.assertEqual(arguments["batch_size"], 8)
        self.assertTrue(arguments["already_preprocessed"])
        self.assertEqual(arguments["device"], "cpu")


class SelectionRuleTests(unittest.TestCase):
    def test_lower_wins(self) -> None:
        self.assertTrue(train_siamese._improves(0.05, 0.08))
        self.assertFalse(train_siamese._improves(0.08, 0.05))

    def test_a_tie_keeps_the_earlier_checkpoint(self) -> None:
        self.assertFalse(train_siamese._improves(0.08, 0.08))

    def test_nan_never_wins(self) -> None:
        self.assertFalse(train_siamese._improves(float("nan"), 0.10))
        self.assertFalse(train_siamese._improves(float("nan"), float("nan")))


class MissingImageTests(unittest.TestCase):
    def test_missing_paths_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            record = SignatureRecord(Path(directory) / "missing.png", "s", False)
            with self.assertRaisesRegex(FileNotFoundError, "do not exist"):
                train_siamese._require_existing_images([record])

    def test_existing_paths_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ok.png"
            path.write_bytes(b"not really a png")
            train_siamese._require_existing_images(
                [SignatureRecord(path, "s", False)], [SignatureRecord(path, "s", True)]
            )


class TrainDatasetFilterTests(unittest.TestCase):
    """``--train-datasets`` must change the train split and nothing else."""

    @staticmethod
    def _records() -> list[SignatureRecord]:
        return [
            SignatureRecord(Path(f"{dataset}/{signer}.png"), signer, False, dataset)
            for dataset, signers in (
                ("BHSig260", ("bh1", "bh2")),
                ("CEDAR", ("ce1",)),
                ("SigResize", ("sr1", "sr2")),
            )
            for signer in signers
        ]

    def test_no_request_keeps_every_row(self) -> None:
        records = self._records()
        self.assertEqual(train_siamese._filter_train_datasets(records, None), records)
        self.assertEqual(train_siamese._filter_train_datasets(records, []), records)

    def test_request_keeps_only_those_corpora(self) -> None:
        kept = train_siamese._filter_train_datasets(self._records(), ["BHSig260", "CEDAR"])
        self.assertEqual([record.signer_id for record in kept], ["bh1", "bh2", "ce1"])
        self.assertEqual({record.dataset for record in kept}, {"BHSig260", "CEDAR"})

    def test_unknown_corpus_is_rejected_with_the_available_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "available: \\['BHSig260', 'CEDAR', 'SigResize'\\]"):
            train_siamese._filter_train_datasets(self._records(), ["SigResize2"])

    def test_signer_spanning_two_corpora_is_rejected(self) -> None:
        records = self._records() + [
            SignatureRecord(Path("SigResize/bh1.png"), "bh1", False, "SigResize")
        ]
        with self.assertRaisesRegex(ValueError, "span multiple datasets"):
            train_siamese._filter_train_datasets(records, ["BHSig260", "SigResize"])

    def test_empty_input_is_rejected(self) -> None:
        # A request against no rows cannot name a known corpus, so it fails on the
        # unknown-name check rather than silently returning an empty training set.
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            train_siamese._filter_train_datasets([], ["CEDAR"])


if __name__ == "__main__":
    unittest.main()
