# Production-v2 Signature Preprocessing

## Scope

Production-v2 is a preprocessing-only, versioned implementation. It does not change
the ConvNeXt model, projection size, scoring, fusion, or checkpoints.

The legacy implementation remains available and is the default for existing callers.
Production-v2 defaults to `512×256`; `384×192` remains available explicitly.

## Cleaning policy

Production-v2 is **KEEP-by-default**. The artifact cleaner is run to identify
candidate artefacts, but it does not delete signature pixels. Candidate
artefacts remain visible and are recorded in metadata and the artifact mask.

This policy prevents cheque lines or signature components from being removed by
an uncertain rule. Destructive cleaning remains available only as an explicit
experimental option:

```python
Params(destructive_cleaning_enabled=True)
```

Do not enable destructive cleaning for public-dataset training until real
stroke-preservation tests demonstrate that it improves the signer-disjoint
benchmark.

## API

For the combined one-call API, use [`preprocessing/signature_preprocess.py`](../preprocessing/signature_preprocess.py):

```python
from preprocessing.signature_preprocess import preprocess_image

result = preprocess_image("crop.png")

result.gray               # uint8 256x512 canonical grayscale
result.mask               # uint8 0/255 binary ink mask
result.artifact_mask      # source-frame 0/128/255 artifact mask
result.metadata           # quality metadata
result.model_input()      # float32 3x256x512
```

The same API accepts a NumPy uint8/uint16/grayscale/BGR/BGRA array.

```bash
python preprocessing/signature_preprocess.py crop.png --out results/
python preprocessing/signature_preprocess.py crop-folder/ --out results/ --debug
python preprocessing/signature_preprocess.py crop.png --out results/ --size 384x192
```

```python
from preprocessing.v2.pipeline import apply_preprocessing

result = apply_preprocessing(image)

canonical = result.canonical_grayscale
binary_mask = result.binary_mask
artifact_mask = result.artifact_mask
metadata = result.metadata
skeleton_mask = result.skeleton_mask
distance_map = result.distance_map
model_input = result.model_input
```

Supported sizes include:

- `(512, 256)` — production-v2 default
- `(384, 192)` — legacy-compatible option

The tuple order is `(width, height)`.

## Pipeline stages

1. **Validate:** input type, dimensions, blank/saturated images, and contrast.
2. **Profile:** ink ratio, contrast, Laplacian blur, border contact, components, bbox,
   and aspect ratio.
3. **Polarity:** test dark-on-light and light-on-dark interpretations and normalize
   to dark ink on white.
4. **Illumination:** preserve faint strokes while estimating a slowly varying
   background. Already-binary images pass through unchanged.
5. **Ink mask:** combine Otsu, adaptive, Sauvola, and grayscale-confidence candidates
   and select deterministically.
6. **Artifact cleaning:** remove only clearly separate lines, edge slivers,
   specks, boxes, and stamps. Uncertain crossings are marked, not removed.
7. **Geometry:** deskew no more than 8 degrees, crop with safe padding, detect
   clipping, preserve aspect ratio, and pad to the requested canvas.

## Artifact mask values

| Value | Meaning |
|---:|---|
| `0` | Untouched |
| `128` | Suspected but kept |
| `255` | Confidently removed |

## Quality metadata

Production-v2 returns:

- `preprocessing_version`
- `selected_polarity`
- `polarity_confidence`
- `ink_ratio`
- `bbox_width`
- `bbox_height`
- `aspect_ratio`
- `component_count`
- `border_contact`
- `crop_clipped`
- `contrast_score`
- `blur_score`
- `artifact_detected`
- `artifact_uncertain`
- `quality_status`
- `warnings`

Invalid images raise `ValueError`; there is no generated blank fallback.

## Dataset CLI

Legacy remains the default:

```bash
python -m preprocessing.preprocess_dataset \
  --manifest data/manifest.csv \
  --output-dir data/processed \
  --mode legacy \
  --size 384x192
```

Production-v2 writes to a separate directory:

```bash
python -m preprocessing.preprocess_dataset \
  --manifest data/manifest.csv \
  --output-dir data/processed/v2_256x512 \
  --mode production-v2
```

The default v2 size is `512×256`.

The legacy-compatible resolution is available explicitly:

```bash
python -m preprocessing.preprocess_dataset \
  --manifest data/manifest.csv \
  --output-dir data/processed/v2_192x384 \
  --mode production-v2 \
  --size 384x192
```

Optional representations can be materialized:

```bash
python -m preprocessing.preprocess_dataset \
  --manifest data/manifest.csv \
  --output-dir data/processed/v2_256x512 \
  --mode production-v2 \
  --skeleton \
  --distance-map \
  --model-input
```

The dataset CLI refuses to write over a directory that already contains a manifest
or production-v2 `gray`, `mask`, or `artifacts` directory.

Production-v2 outputs:

```text
gray/
mask/
artifacts/
skeleton/
distance_map/
model_input/
metadata.jsonl
manifests/manifest_v2.csv
quality_report.json
```

The manifest preserves every original column and adds optional quality columns.

## Migration and rollback

To evaluate Production-v2 safely:

1. Keep existing processed outputs and manifests unchanged.
2. Generate a separate v2 output directory.
3. Keep the existing 192×384 model checkpoint and manifest contract.
4. Compare legacy and v2 only after the signer-disjoint split is fixed.
5. Remove or ignore v2 output if it performs worse; legacy calls require no changes.

The 256×512 output requires a separate model experiment and a separately retrained
checkpoint. It must not be loaded into the existing 192×384 ConvNeXt checkpoint.
