# Integration guide

[Back to the project](../README.md)

## Query tensors

| Argument | Shape | Meaning |
| :--- | :--- | :--- |
| `src_points` | `[B, T, N, 3]` | Normalized initial tooth surfaces, floating point |
| `tooth_ids` | `[B, T]` | Integer anatomical identifiers |
| `mask` | `[B, T]` | Boolean valid-tooth mask |
| `case_center` | `[B, 3]` | Normalization centre in millimetres; needed for output refinement |
| `scale` | `[B]` | Positive normalization scale; needed for output refinement |

`B` is batch size, `T` the padded tooth count and `N` the surface-point count. The manuscript uses 1,024 points per query tooth. Inputs must share the model's device, and floating tensors should use compatible dtypes.

The hard-coded anatomical graph supports Universal-like upper IDs **2–15** and lower IDs **18–31**. Remap dataset labels consistently before calling the model. The embedding vocabulary is larger than this anatomical graph; embedding support alone does not establish correct graph semantics for other identifiers.

## Retrieved cases

Construct a `RetrievedCaseBatch` after retrieval and anatomical correspondence have been established externally.

| Field | Shape / type | Meaning |
| :--- | :--- | :--- |
| `src_points` | `[B, K, T, P, 3]` | Historical initial surfaces |
| `dst_points` | `[B, K, T, P, 3]` | Historical target surfaces |
| `rotations` | `[B, K, T, 3, 3]` | Historical rotations |
| `translations` | `[B, K, T, 3]` | Historical normalized translations |
| `present` | `[B, K, T]`, boolean | Anatomical match availability |
| `scores` | `[B, K]` | Case retrieval scores |
| `identities` | optional nested string lists | Source-case identifiers |
| `trace` | optional list of dictionaries | Externally generated retrieval provenance |

`K` is the retrieved case count; the manuscript uses four. Supply at least one retrieval slot. The model does not perform retrieval, enforce training-only membership, or verify identifiers and trace contents. Historical target fields belong to the selected memory cases; the query target must not enter retrieval or inference.

Corresponding slots must refer to the same anatomical position, with mutually compatible coordinate and pose conventions. Sort candidates by retrieval rank if using the training-time dropout behaviour: slot zero is always retained.

## Coordinates

The model uses row-vector rigid transforms:

```text
X_pred = X_src @ R.T + t
X_norm = (X_mm - case_center) / scale
X_pred_mm = scale * X_pred_norm + case_center
t_global = scale * t_norm + case_center - case_center @ R.T
```

The wrapper performs the last two conversions when an output refiner is provided. Simply multiplying `t_norm` by `scale` is not sufficient when the normalization centre is nonzero.

## Checkpoint contract

`load_v8_weights` expects a dictionary containing:

- `model`: a state dictionary matching `LongHorizonCaseRAGOrthoSARA`, loaded strictly.
- `ema_parameters` (optional): parameter-name-to-tensor mapping with exactly the model's parameter names.
- `rag_strength` (optional): the fusion progress/strength value; defaults to 1.0 during loading.

Checkpoint compatibility with a particular experimental run must be checked using its actual weights. No trained checkpoint is included here. Only load trusted checkpoint files.

## Frozen refiner artifact

`OutputGraphRAG.from_artifact(path)` expects a JSON document containing `frozen: true`, a `config` object, and the fitted `residual_model` and `quality_gate_model` objects. Each head requires:

| Field | Residual head | Quality gate |
| :--- | :--- | :--- |
| `feature_mean` | 59 values | 67 values |
| `feature_scale` | 59 nonzero values | 67 nonzero values |
| `coefficients` | 60 × 6 | 68 × 3 |

The last coefficient row is the intercept. These are fitted numerical parameters, not interchangeable placeholders. The loader does not train either head. The artifact does not substitute for a clinical knowledge graph or a literature evidence store.

## Outputs

`DualRAGV8` always returns an `input_case_rag` dictionary. Selected fields include:

- `R`, `t_norm`, `pred_points`: final input-stage geometry.
- `base_R`, `base_t_norm`, `base_pred_points`: base-model predictions.
- `consensus_R`, `consensus_t_norm`, `consensus_weights`: historical consensus.
- `prior_R`, `prior_t_norm`, `prior_blend`: mixed historical/base prior.
- `rag_attention`, `rag_gate`: fusion diagnostics.
- `retrieved_identities`, `retrieval_trace`: supplied provenance, passed through.

When configured, `output_graph_rag` contains `R`, `t_global`, `pred_points_mm` and per-case `diagnostics`. Its diagnostics report accepted correction counts and maximum collision-related displacement. They are not a literature-linked clinical audit.

## Execution notes

- Call `.eval()` for inference to disable dropout.
- Use `torch.no_grad()` around inference. The collision optimizer locally enables gradients for displacement variables; do not enclose that path in `torch.inference_mode()`.
- Avoid entirely invalid cases. Validate tensor shapes, masks, tooth identifiers and coordinate scales in the calling pipeline.
- Model construction alone leaves `rag_strength` at zero. Restore it from the checkpoint or explicitly set the intended value with `set_rag_strength`.
- Clearance optimization updates tooth displacements, not the frozen network parameters.

## Validation status

The source passed Python syntax parsing during preparation of this release, and the copied model was checked against the original file by SHA-256. No trained weights or patient data were used for runtime or benchmark verification. The original code has not been altered to resolve the manuscript-alignment differences listed in the README.
