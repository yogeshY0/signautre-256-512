# Kaggle: reference-2 multi-seed Stage 3 run (256x512)

Use the notebook at
[`training/signature-verification-3-dataset.ipynb`](../training/signature-verification-3-dataset.ipynb)
for three no-augmentation training runs (seeds 42, 43, and 44), dataset gates, per-source
evaluation, and artifact packaging. No training was run as part of this repository change.

## Dataset upload

Attach the cleaned dataset as a private Kaggle dataset. You can upload the extracted
dataset root containing `manifest_clean.csv` and the `BHSig260/`, `CEDAR/`, and
`SigResize/` image folders, or upload `training_clean_256x512.zip`. The notebook searches
under `/kaggle/input` for the manifest and, if only the archive is present, safely extracts
it under `/kaggle/working`. The manifest paths are relative to the dataset root.

The notebook checks row counts, dataset counts per split, file resolution, signer-disjoint
splits, and reference availability. It then runs both `training.verify_data` and
`verification.validate_dataset`. The latter checks every row for a readable 256x512
uint8 grayscale PNG and writes `/kaggle/working/validate_report.json`. Do not train if
either gate fails.

## Run settings

* Use a Kaggle T4 GPU with Internet enabled for pretrained ConvNeXt weights.
* Every run explicitly passes `--no-augment`; augmentation is opt-in in the training CLI.
  Original dataset files remain read-only.
* Input dimensions are explicitly set to `--image_height 256 --image_width 512`.
* Training uses 5 warmup epochs and up to 40 fine-tuning epochs. `PATIENCE = 8` can stop
  fine-tuning early while retaining the best validation checkpoint; set it to `0` to
  disable early stopping.
* The trainer reports the configured maximum float32 input batch shape and its tensor
  memory estimate. This excludes model activations, gradients, and optimizer state.
* Each checkpoint is selected by validation skilled-forgery EER using 2 medoid-selected
  references. Evaluation reports both 2- and 3-reference results for each seed.

## Evaluation and interpretation

The evaluation threshold is selected on validation signers and applied unchanged to test.
Review skilled-forgery results by dataset, especially SigResize, as well as random
impostor results, signer-level confidence intervals, worst signers, and the 10% validation
FRR operating point.

The reported operating point is diagnostic, not a production FAR requirement. Define a
target FAR from the product threat model and validate it on a sufficiently large,
untouched signer-disjoint holdout before making production claims. The current validation
and test results are development evidence, not a guarantee of production performance.

Fusion is deliberately excluded from this run: the prior validation gain was small and
was not confirmed on test. The standalone fusion scripts remain limited to their validated
192x384 input size and reject other sizes.

## Artifacts

The notebook writes three checkpoints, training logs, the dataset validation report,
evaluation JSON, and per-seed audit histograms under `/kaggle/working`. Its final cell
packages these as `/kaggle/working/artifacts_ref2_seeds.tar.gz`. Download that archive
before the Kaggle session ends.
