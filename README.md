# hypnoconv-bf-public

Public-facing minimal companion repository for the HypnoConv-BF-v1 paper line.

This package is intentionally limited in scope. It is designed to accompany the
paper and to expose only the minimal method and evaluation logic needed for
public understanding of the work.

## Included

- model definition for HypnoConv-BF-v1
- minimal dataset/window construction utilities
- minimal evaluation script
- minimal statistical comparison script
- dependency list
- release boundary notes

## Not Included

- pretrained weights
- full training pipeline
- unpublished experiments
- unrelated project history
- device-specific adaptation or deployment artifacts

## Expected Data Layout

The public evaluation utilities expect a processed directory containing:

```text
DATA_DIR/
  X.npy
  y.npy
  groups.npy
  sessions.npy
  epoch_ids.npy   # optional but recommended
```

Where:

- `X.npy` has shape `(N, C, 3000)`
- `y.npy` contains stage labels in `{0,1,2,3,4}`
- `groups.npy` contains subject identifiers
- `sessions.npy` contains session identifiers

## Minimal Usage Example

If a compatible checkpoint is available later, a minimal evaluation command is:

```bash
python eval_public.py \
  --data-dir /path/to/processed_data \
  --checkpoint /path/to/checkpoint.pth \
  --output-json eval_metrics.json
```

If no checkpoint is provided, the script still documents the expected loading
path and evaluation structure, but it will not reproduce published metrics.

## Weights Policy

Pretrained checkpoints are not included in this package and are not released at
this stage.

## Intended Usage

This package is intended for:

- paper understanding
- method inspection
- non-production research use
- internal or academic evaluation

It is not intended as a full commercial deployment package.

## License

This repository is released under the GNU Affero General Public License v3.0
(AGPL-3.0). See `LICENSE`.

## Commercial Licensing

For commercial deployment, commercial productization, hosted commercial service
integration, or other commercial licensing discussions, please contact:

- Beijing Brain Computer Interface Commercial Co., Ltd.
- Website: <https://www.bbci.net/en>
- Email: <contact@bbci.net>

Pretrained checkpoints are not included in this package and are not released at
this stage.
