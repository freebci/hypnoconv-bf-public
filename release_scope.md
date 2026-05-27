# Release Scope

This public companion package is a minimal subset of the internal project.

## Included

- `model_bf.py`
- `dataset_public.py`
- `eval_public.py`
- `stats_public.py`
- `requirements.txt`
- `README.md`

## Excluded

- pretrained checkpoints
- full training orchestration
- unpublished adaptation code
- device-specific deployment artifacts
- unrelated experiment lines

## License Intention

The current intended release posture is a Business Source License style
restriction rather than a permissive open-source license.

## Rationale

The goal is to make the paper's core method and evaluation logic auditable while
retaining control over unpublished and commercialization-sensitive assets.
