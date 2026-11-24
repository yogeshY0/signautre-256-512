# Kaggle: Stage 3 training, fusion and evaluation (256x512)

Ready-to-paste cells for a Kaggle notebook. Assumptions:

* the cleaned image dataset is mounted **read-only** under `/kaggle/input`, with
  `manifest_aug.csv` somewhere inside it and the three corpus folders next to it;
* the code comes from a GitHub clone (or a Kaggle code dataset).

Nothing here writes to `/kaggle/input`. Every output goes to `/kaggle/working`, which
is what the notebook's *Output* tab offers for download when the session ends.

> **Before running:** the commands below use `--select-by`, `--augment/--no-augment`,
> `--image_height/--image_width` and `--target-far`. If the clone does not have them,
> the repository has not been pushed since they were added. Either push first, or upload
> the repository folder as a Kaggle *code* dataset and point `REPO_DIR` at it.

## Cell 0 - setup

```bash
%%bash
set -euo pipefail
REPO_URL="https://github.com/acharyayush/Signature-Verification-FIRST-.git"
REPO_DIR="/kaggle/working/Signature-Verification-FIRST-"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git log --oneline -1

python -m pip install -q -r requirements.txt

python - <<'PY'
import cv2, timm, torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("timm", timm.__version__, "| cv2", cv2.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

## Cell 1 - path check

Finds the manifest, proves every image resolves, and writes `/kaggle/working/paths.env`
so every later cell reuses the same `MANIFEST` and `IMAGE_ROOT`.

```bash
%%bash
set -euo pipefail
python - <<'PY'
import csv, glob, os
from collections import Counter

candidates = sorted(glob.glob("/kaggle/input/**/manifest_aug.csv", recursive=True))
assert candidates, "no manifest_aug.csv under /kaggle/input - check the dataset mount"
manifest = candidates[0]
root = os.path.dirname(manifest)
print("manifest  :", manifest)
print("image root:", root)

rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
print("rows      :", len(rows))
print("splits    :", dict(Counter(r["split"] for r in rows)))
print("signers   :", {s: len({r["signer_id"] for r in rows if r["split"] == s})
                      for s in ("train", "val", "test")})
print("datasets  :", dict(Counter(r["dataset"] for r in rows)))
print("folders   :", sorted(n for n in os.listdir(root) if not n.startswith(".")))

missing = [r["image_path"] for r in rows
           if not os.path.isfile(os.path.join(root, r["image_path"]))]
print("missing   :", len(missing))
assert not missing, f"first missing path: {missing[0]}\nset IMAGE_ROOT by hand if the mount differs"
assert len(rows) == 20548, f"expected 20548 rows, got {len(rows)}"

with open("/kaggle/working/paths.env", "w", encoding="utf-8") as handle:
    handle.write(f'export MANIFEST="{manifest}"\n')
    handle.write(f'export IMAGE_ROOT="{root}"\n')
print("\nwrote /kaggle/working/paths.env")
PY
cat /kaggle/working/paths.env
```

## Cell 2 - verify the dataset

Read-only. Fails loudly, before anything is trained, if a row is broken.

```bash
%%bash
set -euo pipefail
source /kaggle/working/paths.env
cd /kaggle/working/Signature-Verification-FIRST-
python -m training.verify_data --manifest "$MANIFEST" --image-root "$IMAGE_ROOT"
```

Expect `RESULT: PASS (3/3 sections passed)`. The augmentation section reports 14,401
train images fed and `val 0, test 0`; the ink-retention line is the one to read if the
augmenter is ever changed.

## Cell 3 - train with augmentation (the run to keep)

45 epochs = 5 warmup with a frozen backbone + 40 fine-tune with cosine decay.
`--select-by eer` picks the checkpoint on validation skilled-forgery EER, which is the
number reported at the end.

```bash
%%bash
set -euo pipefail
source /kaggle/working/paths.env
cd /kaggle/working/Signature-Verification-FIRST-
python -u -m training.train_siamese \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --already-preprocessed \
  --image_height 256 \
  --image_width 512 \
  --seed 42 \
  --phase1-epochs 5 \
  --phase2-epochs 40 \
  --select-by eer \
  --augment \
  --workers 4 \
  --output-dir /kaggle/working/ckpt_aug 2>&1 | tee /kaggle/working/train_aug.log
```

## Cell 4 - baseline, identical except `--no-augment`

Same seed, same epochs, same sampler, same selection metric. The only difference is
`--augment` becoming `--no-augment`, so the comparison isolates augmentation.

```bash
%%bash
set -euo pipefail
source /kaggle/working/paths.env
cd /kaggle/working/Signature-Verification-FIRST-
python -u -m training.train_siamese \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --already-preprocessed \
  --image_height 256 \
  --image_width 512 \
  --seed 42 \
  --phase1-epochs 5 \
  --phase2-epochs 40 \
  --select-by eer \
  --no-augment \
  --workers 4 \
  --output-dir /kaggle/working/ckpt_base 2>&1 | tee /kaggle/working/train_base.log
```

Compare without re-reading whole logs - this prints the best epoch of each run:

```bash
%%bash
set -euo pipefail
python - <<'PY'
import json
for name in ("base", "aug"):
    best = None
    for line in open(f"/kaggle/working/train_{name}.log", encoding="utf-8"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if best is None or row["selection_value"] < best["selection_value"]:
            best = row
    print(f"{name:>4}  best skilled EER {best['validation_eer_skilled']:.4f}"
          f"  at {best['phase']} epoch {best['epoch']}"
          f"  (val loss {best['validation']['loss']:.4f})")
PY
```

## Cell 5 - fit the fusion config

Fitted on validation signers only. The out-of-fold threshold it stores is what
`evaluate` will use, so this cell must run before Cell 6.

```bash
%%bash
set -euo pipefail
source /kaggle/working/paths.env
cd /kaggle/working/Signature-Verification-FIRST-
python -u -m training.fit_fusion \
  --checkpoint /kaggle/working/ckpt_aug/best_model.pt \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --already-preprocessed \
  --image_height 256 \
  --image_width 512 \
  --references 5 \
  --device cuda \
  --workers 4 \
  --output /kaggle/working/fusion_aug.json
```

`--references 5` must match the value used during training, otherwise the validation
queries the fusion fitter scores are not the ones the checkpoint was selected on.

## Cell 6 - evaluate deep and fused

Thresholds are chosen on validation signers and applied unchanged to test. `--target-far`
adds the operating point a checklist review cares about; `per_dataset` is where CEDAR
shows up if it is still the weak corpus.

```bash
%%bash
set -euo pipefail
source /kaggle/working/paths.env
cd /kaggle/working/Signature-Verification-FIRST-
python -u -m training.evaluate \
  --checkpoint /kaggle/working/ckpt_aug/best_model.pt \
  --manifest "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --already-preprocessed \
  --image_height 256 \
  --image_width 512 \
  --references 5 \
  --device cuda \
  --workers 4 \
  --fusion-config /kaggle/working/fusion_aug.json \
  --target-far 0.001 \
  --fusion-report /kaggle/working/report_aug.json
```

## Reading the report

| Key | Meaning |
| --- | --- |
| `test.eer_skilled` | deep baseline, the number the model is selected on |
| `test.eer_random` | impostor pairs; should stay near zero if skilled improves |
| `per_dataset.val` | what the checkpoint saw during selection, per corpus |
| `per_dataset.test` | the numbers to quote, per corpus |
| `fusion.test` | deep + handcrafted; only worth shipping if `eer_skilled` drops and `eer_random` does not collapse |
| `operating_point.deep.test` | FAR/FRR at the FAR budget, threshold from validation |
| `operating_point.fused.test` | same for the fused score |

Report the **skilled** numbers, not the average of skilled and random. A run is only an
improvement if `per_dataset.test.CEDAR.eer_skilled` also moves; CEDAR holds 9 test
signers, so a single flipped pair moves it by several points and the confidence interval
is wide.

## Session notes

* `/kaggle/working` is wiped when the session ends. Download `best_model.pt`,
  `fusion_aug.json`, `report_aug.json` and both training logs before stopping.
* Only the best checkpoint is written, so an interrupted run loses the epochs after the
  last improvement. If 45 epochs does not fit the session, lower `--phase2-epochs` rather
  than splitting the run: the cosine schedule assumes one continuous fine-tune phase.
* Determinism is enforced (`use_deterministic_algorithms`), so the baseline and the
  augmented run share model initialisation and batch order and differ only in the train
  images they see.

