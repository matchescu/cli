# `matchescu ambiguity` — user manual

This package makes entity resolution datasets harder in a controlled way. It
adds synthetic *bridge* records that sit between two clusters. A bridge
belongs to one cluster but resembles a second one. The result is a dataset
with directional matches: `(A, B)` can match while `(B, A)` does not.

There are two commands:

- `analyse` finds good settings for a dataset and explains why they are good.
- `generate` writes the ambiguous datasets.

Both commands read every setting from one JSON configuration file. The
command line only says where the files are.

## Contents

1. [Concepts](#concepts)
2. [Configuration file](#configuration-file)
3. [Paths](#paths)
4. [The `analyse` command](#the-analyse-command)
5. [The `generate` command](#the-generate-command)
6. [Workflow for a new dataset](#workflow-for-a-new-dataset)
7. [Recipe: datasets for learning directional matching](#recipe-datasets-for-learning-directional-matching)
8. [Troubleshooting](#troubleshooting)

## Concepts

**Cluster.** A set of records that refer to the same entity. Clusters come
from the dataset's ground truth.

**Ambiguity score.** A score for how well two records could share a bridge.
It is computed field by field and summed:

| Field situation                          | Score                                               |
|------------------------------------------|-----------------------------------------------------|
| Same value on both records               | +2                                                  |
| Different strings on an ambiguity target | +2 × edit similarity (+2 if one contains the other) |
| Different strings on any other field     | −0.5                                                |
| Field present on one record only         | −0.5                                                |
| Any other difference (types, numbers)    | −1                                                  |

**Representative.** The member of a cluster with the highest mean ambiguity
score against the other members.

**Pairing.** Each bridge blends records of two different clusters. How those
records are chosen depends on `pairingStrategy`. The default,
`greedy-cluster-reps`, visits clusters in a fixed order. Each visited cluster is
a *home*. It picks the other cluster with the highest score against its own
representative; that cluster is its *target*. A cluster's score is the lowest
ambiguity between the home representative and the target's first `targetLinks`
ranked members, starting with the target representative. A cluster that already
received a bridge cannot be a target. A pairing is kept only when its score is
at least `minAmbiguity`.

The alternative, `simple-pairs`, does not use representatives. It blocks
records on the tokens of their ambiguity targets, keeps the highest-weight
neighbours of each record among the other clusters, and turns every kept pair
into a bridge. The home is the record in the lower-id cluster. Each bridge
links to the paired record only, so `targetLinks` is ignored. `pairingNeighbours`,
`pairingScope` and `pairingMaxBlockSize` tune this strategy. A pairing is kept
only when its score is at least `minAmbiguity`.

**Bridge.** One new record per kept pairing. It joins the home cluster. Its
fields are built from the two paired records, which are the representatives
under `greedy-cluster-reps`:

- equal values are copied;
- fields present on only one representative are left out;
- ambiguity targets that differ get a synthesised blend of the two values;
- secondary features are copied from the home record;
- other differing strings become empty, other differing values become null;
- candidates longer than `maxLengthRatio ×` the mean token count of the two
  values are dropped; when none fit, the shortest candidates are kept.

**Labels.** `0` no match, `1` match in both directions, `2` one-way match.
Every pair inside a cluster is labelled 1. Each bridge gets
`targetLinks` pairs `(target record, bridge)` labelled 2; `simple-pairs`
always writes one such pair. The bridge is always the right-hand record in
its pairs.

**Alpha.** Each blended value is the candidate text with the highest weighted
harmonic mean of its similarity to the home value (`s_home`) and to the
target value (`s_target`):

```
h = 1 / (alpha / s_home + (1 - alpha) / s_target)
```

`alpha = 0.5` treats both sides the same. Higher alpha leans the bridge text
towards its home cluster. Each alpha gives one dataset *variant*. The
bridges, clusters and labels are the same in every variant; only the bridge
text changes.

**Target floor.** Skips candidates whose `s_target` falls below
`targetFloor × s_target` of the `alpha = 0.5` pick. `0` disables it.

**Theta (θ).** The `minAmbiguity` value `analyse` suggests. It fits two
Gaussians to the pairing scores, one for genuine and one for superficial
resemblance, and returns the score where they are equally likely. The scores
cover `targetLinks` members of each target, so θ depends on `targetLinks`.
Re-run `analyse` after changing it.

**Separability.** The pair text alone may reveal direction. `analyse`
compares the similarity of *source* pairs `(home member, bridge)` with the
similarity of *target* pairs `(target record, bridge)`. The AUC of that
comparison is `content_auc_source_vs_target`. Separability is
`|2 × AUC − 1|`: 0 means the text reveals nothing, 1 means it reveals
everything.

**Surface gap.** Two logistic regressions try to tell target pairs from
source pairs using token-overlap features. One sees only features that do
not change when the pair is swapped. The other also sees order-aware
features. `surface_auc_gap` is the AUC of the second minus the AUC of the
first. A positive gap means direction is carried by *which record is on which
side*, which only an order-sensitive model can use.

**Detectability.** `detect_auc_lr` measures how easily a simple model spots
bridges by their shape: empty fields, token count and character count on the
ambiguity targets. 0.5 means bridges look like ordinary records.

**Bridge component.** Clusters tied together by bridges. A component must sit
in a single train/valid/test split, or the split leaks clusters.

## Configuration file

The file has a top-level `profiling` block, a top-level `analysis` block,
top-level pairing keys and an `inputData` list with one entry per dataset. Only
`inputData` is required.

```json
{
  "pairingStrategy": "greedy-cluster-reps",
  "profiling": {
    "negativeSimilarity": false,
    "surfaceBaseline": true,
    "syntheticDetectability": true,
    "seed": 42,
    "cvFolds": 5,
    "negPosRatio": 4.0,
    "splitRatio": [
      3,
      1,
      1
    ]
  },
  "analysis": {
    "selection": {
      "criterion": "order-headroom",
      "minSeparability": 0.2,
      "maxAuc": 0.9,
      "targetResolvable": 0.5,
      "maxDetectAuc": 0.7
    },
    "threshold": {
      "startValue": 0.0,
      "stepCount": 10,
      "stepSize": 1.0,
      "gmmRestarts": 50
    }
  },
  "inputData": [
    {
      "dataset": {
        "type": "csv",
        "directory": "./data/original/fodors_zagat_nophone",
        "sources": [
          {
            "fileName": "fodors_zagat_nophone_ids.csv",
            "traits": [
              {
                "type": "string",
                "keys": [
                  "name",
                  "addr",
                  "city",
                  "type"
                ]
              }
            ],
            "idCol": "id",
            "sourceCol": "source"
          }
        ],
        "pairwiseMapping": {
          "fileName": "fodors_zagat_nophone_mapping.csv",
          "leftIdCol": 0,
          "leftSourceCol": 1,
          "rightIdCol": 2,
          "rightSourceCol": 3,
          "hasHeader": true
        },
        "clusterMapping": "fodors_zagat_nophone_cluster_mapping.csv"
      },
      "ambiguityTargets": [
        "name",
        "addr",
        "type"
      ],
      "minAmbiguity": 4.25,
      "alphas": [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
      ],
      "targetFloor": 0.0,
      "targetLinks": 1
    }
  ]
}
```

### Pairing (top level, all datasets)

| Key                   | Default               | Read by        | Meaning                                                                                                                                                                                        |
|-----------------------|-----------------------|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `pairingStrategy`     | `greedy-cluster-reps` | both           | How records of different clusters are paired. `greedy-cluster-reps` writes one bridge per home cluster from cluster representatives. `simple-pairs` writes one bridge per blocked record pair. |
| `pairingNeighbours`   | `1`                   | `simple-pairs` | Highest-weight partners kept per record, at least 1.                                                                                                                                           |
| `pairingScope`        | `any`                 | `simple-pairs` | `any` keeps a pair when it is a top partner of either record. `mutual` keeps it only when it is a top partner of both.                                                                         |
| `pairingMaxBlockSize` | `100`                 | `simple-pairs` | Largest token block kept. Tokens shared by more records are ignored. At least 2.                                                                                                               |

### Per dataset (`inputData[]`)

| Key                 | Default  | Read by | Meaning                                                                                                                                                                                     |
|---------------------|----------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset`           | required | both    | Where the dataset lives and how its files look. Same format as the `matching` package's dataset configs.                                                                                    |
| `ambiguityTargets`  | none     | both    | Fields that receive blended values. The profile compares records on these fields only, so at least one is needed.                                                                           |
| `secondaryFeatures` | none     | both    | Fields copied from the home representative. Must not overlap `ambiguityTargets` and must not be one of the reserved values `source`, `target` or `target_cluster_id`.                       |
| `minAmbiguity`      | none     | both    | Minimum pairing score of a bridge. Omit it to keep every pairing.                                                                                                                           |
| `alphas`            | required | both    | One or more values in (0, 1). `analyse` searches them; `generate` writes one variant per value.                                                                                             |
| `targetFloor`       | `0.0`    | both    | In [0, 1]. See [Concepts](#concepts).                                                                                                                                                       |
| `targetLinks`       | `1`      | both    | One-way pairs per bridge, at least 1. Also sets how many target members the pairing score covers. Values above 1 add `_kN` to output names. Ignored by `simple-pairs`, which always uses 1. |
| `maxLengthRatio`    | none     | both    | Highest bridge-value length, in tokens, as a multiple of the mean length of the two blended values. At least 1. Omit it to disable the limit.                                               |
| `analysis`          | none     | analyse | Replaces the whole top-level `analysis` block for this dataset. Omitted sub-blocks take their defaults, not the top-level values.                                                           |

### Profiling (`profiling`, all datasets)

Both commands profile every variant. The core profile always runs. Three
extra parts are switched on here because they cost more time.

| Key                      | Default     | Meaning                                                                                 |
|--------------------------|-------------|-----------------------------------------------------------------------------------------|
| `negativeSimilarity`     | `false`     | Samples label-0 pairs and compares their similarity with the positives.                 |
| `surfaceBaseline`        | `false`     | Computes the surface gap.                                                               |
| `syntheticDetectability` | `false`     | Computes detectability.                                                                 |
| `seed`                   | `42`        | Seed for negative sampling and cross-validation folds.                                  |
| `cvFolds`                | `5`         | Folds of the cross-validation, grouped by bridge component. At least 2.                 |
| `negPosRatio`            | `4.0`       | Label-0 pairs sampled per positive pair. Greater than 0.                                |
| `splitRatio`             | `[3, 1, 1]` | Train/valid/test sizes used to judge whether the largest bridge component fits a split. |

### Analysis (`analysis`, default for all datasets, ignored by `generate`)

An `inputData[]` entry may carry its own `analysis` block. When it does,
that block replaces the top-level one for the dataset; when it does not,
the top-level block applies.

| Key                          | Default    | Meaning                                                                 |
|------------------------------|------------|-------------------------------------------------------------------------|
| `selection.criterion`        | `smallest` | How to choose alpha. See [criteria](#choosing-alpha).                   |
| `selection.minSeparability`  | `0.2`      | Lowest separability an alpha may have.                                  |
| `selection.maxAuc`           | `0.9`      | Highest content AUC (read in its stronger direction) an alpha may have. |
| `selection.targetResolvable` | `0.5`      | Share of resolvable bridges `gmm-balance` aims for.                     |
| `selection.maxDetectAuc`     | `0.7`      | Highest detectability `order-headroom` accepts. In [0.5, 1].            |
| `threshold.startValue`       | `0.0`      | First threshold of the sweep.                                           |
| `threshold.stepCount`        | `10`       | Number of thresholds. At least 1.                                       |
| `threshold.stepSize`         | `1.0`      | Distance between thresholds. At least 0.1.                              |
| `threshold.gmmRestarts`      | `50`       | Restarts of the theta fit. More restarts give a steadier theta.         |

Invalid values stop the command before any data is loaded, with a message
naming the key. Unknown keys are ignored silently, so a misspelt key falls
back to its default.

## Paths

```
matchescu [-d ROOT] ambiguity analyse  -c CONFIG [--plot-dir DIR]
matchescu [-d ROOT] ambiguity generate -c CONFIG -O DIR
```

- `-c`, `-O` and `--plot-dir` are relative to the current directory.
- Each `dataset.directory` in the configuration is relative to `ROOT`.
  `ROOT` defaults to the current directory.
- `-O` must already exist. `--plot-dir` is created when missing and defaults
  to the current directory.

## The `analyse` command

`analyse` does not write datasets. For each dataset it runs two searches and
writes a report.

**Threshold search.** It pairs clusters without any minimum, fits theta to
the pairing scores and counts bridges at `stepCount` thresholds. It ignores
`minAmbiguity`. A dataset with fewer than two pairing scores is skipped with a
warning.

**Alpha search.** It generates every alpha in `alphas` at the configured
`minAmbiguity`, profiles each variant and chooses one alpha. An alpha is *feasible* when its separability lies between
`minSeparability` and
`2 × maxAuc − 1`. The criterion then picks one feasible alpha.

### Choosing alpha

| Criterion        | Picks                                                                              | Needs                                       |
|------------------|------------------------------------------------------------------------------------|---------------------------------------------|
| `smallest`       | The smallest feasible alpha: the least lopsided text that still carries direction. | —                                           |
| `gmm-balance`    | The alpha whose share of resolvable bridges is closest to `targetResolvable`.      | —                                           |
| `surface-gap`    | The alpha with the largest surface gap.                                            | `surfaceBaseline`                           |
| `order-headroom` | The largest surface gap among alphas with `detect_auc_lr ≤ maxDetectAuc`.          | `surfaceBaseline`, `syntheticDetectability` |

Ties go to the smaller alpha. When a criterion's switches are off, `analyse`
stops at once with a usage error.

### Output files (in `--plot-dir`)

| File                            | Content                                                                                                                                    |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `ambiguity_methodology.csv`     | One row per dataset: theta, the fit, bridges and links at theta, the selection settings, the chosen alpha and the bridge components. **Start here.** |
| `ambiguity_alpha_sweep.csv`     | One row per dataset and alpha: `feasible`, `selected` and the full profile.                                                                |
| `ambiguity_threshold_sweep.csv` | Bridge count, link count and short-bridge count per threshold, at the configured `targetLinks`.                                             |
| `gmm_fit_<dataset>.png`         | Pairing-score histogram with the two Gaussians and theta.                                                                                  |
| `alpha_sweep_<dataset>.png`     | Separability and resolvable share per alpha, with the feasible band.                                                                       |
| `threshold_analysis.png`        | Bridge counts against thresholds for all datasets.                                                                                         |

The `dataset` column uses short display names for some datasets, for example
`fodors-zagat-nophone` for `fodors_zagat_nophone`.

### Columns that matter

In `ambiguity_methodology.csv`:

| Column                                              | Read it as                                                                                                                           |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `theta`                                             | Suggested `minAmbiguity`.                                                                                                            |
| `boundary_type`                                     | `bayes_optimal` is a proper fit; `mean_midpoint` means the two Gaussians did not separate and theta is a fallback.                   |
| `target_links`                                      | `targetLinks` the analysis ran with.                                                                                                 |
| `bridge_count_at_theta`, `bridge_count_per_cluster` | How many bridges theta leaves.                                                                                                       |
| `link_count_at_theta`, `link_count_per_cluster`     | How many one-way (label-2) pairs theta leaves.                                                                                       |
| `short_bridge_count_at_theta`                       | Bridges whose target cluster has fewer than `targetLinks` members, so they get fewer links than requested.                           |
| `alpha_star`                                        | The chosen alpha, empty when none qualifies.                                                                                         |
| `alpha_star_reason`                                 | `selected`, or why nothing was chosen (see [Troubleshooting](#troubleshooting)).                                                     |
| `n_feasible_alphas`                                 | Alphas inside the separability band.                                                                                                 |
| `largest_bridge_component_split_load`               | One-way pairs of the largest component divided by the one-way quota of the smallest split. Above 1, a leak-free split is impossible. |

In `ambiguity_alpha_sweep.csv`, per alpha:

| Column                                                         | Read it as                                                                                                                                                                |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `content_auc_source_vs_target`                                 | Above 0.5: the bridge text leans home. Below 0.5: it leans to the target.                                                                                                 |
| `content_separability`                                         | Direction readable from plain similarity.                                                                                                                                 |
| `content_auc_extra_vs_target`                                  | Similarity of ordinary matches against one-way pairs. High values mean one-way pairs are less similar than ordinary matches, which a model can use without reading order. |
| `surface_auc_symmetric`, `surface_auc_full`, `surface_auc_gap` | Surface models without and with order features, and their difference.                                                                                                     |
| `detect_auc_lr`, `detect_best_feature`                         | Detectability and its strongest shape feature.                                                                                                                            |
| `margin_pi_resolvable`                                         | Share of bridges whose direction plain similarity resolves, from a two-Gaussian fit to the per-bridge similarity margins.                                                 |
| `n_bridges`, `n_label2`, `n_label1`                            | Size of the directional task.                                                                                                                                             |
| `field_share_home_median`                                      | How far the blended values lean home (0.5 is neutral).                                                                                                                    |

## The `generate` command

`generate` writes, for each dataset and each alpha in `alphas`:

| Path in `-O`                            | Content                                                                                                                                           |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `amb_<name>_a<alpha>[_kN]/`             | Undirected variant: every positive pair has label 1.                                                                                              |
| `amb_<name>_a<alpha>[_kN]_dir/`         | Directed variant: one-way pairs have label 2. Use this one for directional matching.                                                              |
| `amb_<name>_bridge_log.csv`             | Every blended value: alpha, fields, `sim_home`, `sim_target`, `share_home`, whether the floor or a fallback was used.                             |
| `amb_<name>[_kN]_link_log.csv`          | Every one-way link and its ambiguity score.                                                                                                       |
| `amb_<name>[_kN]_bridge_components.csv` | The bridge components.                                                                                                                            |
| `metadata.csv`                          | One row per variant: the settings used, the pairing strategy and the full profile. Rows are replaced by name, so repeated runs into the same directory build one table. Write runs with different strategies to different output directories. |

`<name>` is the last part of `dataset.directory`. `<alpha>` is alpha in
hundredths, for example `a065` for 0.65. Each variant directory holds
`<variant>_ids.csv`, `<variant>_mapping.csv` (columns `left_id`,
`left_source`, `right_id`, `right_source`, `label`) and
`<variant>_cluster_mapping.csv`. Bridges have `source = ambiguity-generator`.

`generate` ignores the `analysis` block. It uses `profiling` for the
`metadata.csv` columns.

## Workflow for a new dataset

1. **Describe the dataset.** Write its `dataset` block, as for any matchescu
   benchmark. Check that it loads and has a cluster mapping or a pairwise
   mapping.
2. **Choose the fields.** Put the descriptive text fields that a blend can
   mix (names, titles, addresses) in `ambiguityTargets`. Put fields the
   bridge should keep from home (a year, an identifier) in
   `secondaryFeatures`. Every other field that differs between the two
   representatives is emptied in the bridge.
3. **Write a coarse search.** Leave `minAmbiguity` out, set
   `"alphas": [0.5, 0.6, 0.7, 0.8, 0.9]`, and keep the default `analysis`
   block.
4. **Run `analyse`.**
   ```bash
   matchescu -d <workspace> ambiguity analyse -c my-config.json --plot-dir reports/my-dataset
   ```
5. **Set the threshold.** Read `theta` in `ambiguity_methodology.csv` and
   check `gmm_fit_<dataset>.png`. If `boundary_type` is `bayes_optimal`, set
   `minAmbiguity` to `theta`. If it is `mean_midpoint`, pick a value from
   `threshold_analysis.png` that keeps enough bridges.
6. **Run `analyse` again.** The alpha search now uses the threshold you set.
   Refine `alphas` around the interesting range if needed.
7. **Choose alpha.** Read `alpha_star` and `alpha_star_reason`. If no alpha
   was chosen, see [Troubleshooting](#troubleshooting).
8. **Write the generate configuration.** Copy the file and set `alphas` to
   the values you want to publish, for example `[0.5, alpha_star]`.
9. **Run `generate`.**
   ```bash
   mkdir -p data/ambiguous
   matchescu -d <workspace> ambiguity generate -c my-generate-config.json -O data/ambiguous
   ```
10. **Check `metadata.csv`.** The rows must show your settings and the same
    profile values that `analyse` reported for these alphas.

## Recipe: datasets for learning directional matching

**Goal.** A BERT matcher with a natural language inference head should learn
direction from input order. Cross-attention should make that easier. A good
dataset therefore puts direction into the *order* of the pair. It must not
let a model find direction another way.

**What "optimal" means here.** Among the alphas where

1. the pair text carries some direction but plain similarity does not solve
   it (`content_separability` between 0.2 and 0.8, so AUC at most 0.9), and
2. bridges are hard to spot (`detect_auc_lr ≤ 0.7`),

choose the alpha with the largest `surface_auc_gap`. This is the
`order-headroom` criterion. The surface gap is a stand-in measured without a
neural model. It does not prove that a matcher will learn order. Confirm the
choice with a training run that compares a symmetric baseline with
cross-attention.

### Steps

1. **Configure the search.** Use the configuration shown under
   [Configuration file](#configuration-file), with these settings:
    - all three `profiling` switches on;
    - `"criterion": "order-headroom"`;
    - `"alphas": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]`;
    - `"targetFloor": 0.0` (a floor of 0.5 changed fewer than 5% of values in
      experiments 06 and 07);
    - `"targetLinks": 1` (the split keeps one one-way pair per bridge).
2. **Run `analyse`** and settle `minAmbiguity` as in the
   [workflow](#workflow-for-a-new-dataset), steps 5 and 6.
3. **Decide per dataset.**

   | Outcome                                              | Meaning                                         | Action | 
         |------------------------------------------------------|-------------------------------------------------|--------|
   | `selected` and `surface_auc_gap > 0`                 | Order carries direction.                        | Generate `[0.5, alpha_star]`. |
   | `selected` and `surface_auc_gap ≤ 0`                 | Direction is in similarity only, not in order.  | Do not use the dataset to test order learning. |
   | `no_feasible_alpha`, separability below 0.2 for every alpha | The text carries no direction.          | Add fields to `ambiguityTargets`, or drop the dataset. |
   | `no_feasible_alpha`, content AUC above 0.9 for every alpha | Similarity alone solves direction.       | Search a finer grid near 0.5, for example 0.50 to 0.62 in steps of 0.02. |
   | `no_undetectable_alpha`                              | Bridges are easy to spot at every feasible alpha. | Read `detect_best_feature`. When it is `n_tokens` or `n_chars`, set `maxLengthRatio` and run again. Otherwise drop the dataset, or raise `maxDetectAuc` and record that you did. |

4. **Check the chosen variant.**
    - `largest_bridge_component_split_load` must be at most 1.
    - Count one-way pairs: `n_label2 × smallest split share` is an upper
      bound on the one-way pairs in the test split. The split drops bridges
      that compete for the same cluster, so the real number can be much lower (fodors kept 239 of 595 bridges). Below
      about 50 one-way test pairs,
      results will be noisy.
    - A content AUC below 0.5 means the bridge text leans to the target. The
      direction signal is real but inverted; note it when you compare
      datasets. affiliationstrings behaved this way in experiment 07.
    - Detectability only looks at the ambiguity targets. Fields that the
      generator empties are not measured, yet a matcher sees them. Compare the
      empty-value shares of bridges and original records:
      ```python
      import polars as pl
 
      ids = pl.read_csv("data/ambiguous/amb_x_a065_dir/amb_x_a065_dir_ids.csv", infer_schema_length=0)
      print(
          ids.group_by(bridge=pl.col("source") == "ambiguity-generator")
          .agg(pl.all().exclude("id", "source").is_null().mean())
      )
      ```
      A field that is empty far more often on bridges is a shortcut. Add it to
      `ambiguityTargets` or `secondaryFeatures` and run again.
5. **Generate.** Keep only the datasets that passed. Set
   `"alphas": [0.5, alpha_star]` for each. The `0.5` variant is a control
   with the same bridges and labels but no lean in the text. Keep the same
   `profiling` block so `metadata.csv` records why each variant was chosen.

## Troubleshooting

| Symptom                                                | Cause and fix                                                                                                                                                                              |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Invalid value for '-c' / '--config-file'`             | A setting is invalid. The message names the key, for example `inputData.0.alphas`.                                                                                                         |
| `No such option: --alphas` (or another old flag)       | All settings moved into the configuration file. See [Configuration file](#configuration-file).                                                                                             |
| `the alpha criterion needs the … option(s)`            | Switch on the named `profiling` options (`surface_baseline` is `surfaceBaseline`, `synthetic_detectability` is `syntheticDetectability`).                                                  |
| Warning `skipping <dataset>: … pairing score(s) found` | The dataset has too few clusters for the theta fit.                                                                                                                                        |
| `alpha_star_reason` is `no_feasible_alpha`             | No alpha falls inside the separability band. Read `content_separability` per alpha in `ambiguity_alpha_sweep.csv`.                                                                         |
| `alpha_star_reason` is `no_bayes_optimal_fit`          | `gmm-balance` found no proper bridge-margin fit. Use another criterion.                                                                                                                    |
| `alpha_star_reason` is `no_surface_baseline`           | `surface-gap` found no surface gap. The cross-validation needs at least `cvFolds` bridge components; lower `cvFolds`.                                                                      |
| `alpha_star_reason` is `no_undetectable_alpha`         | No feasible alpha has a measured `detect_auc_lr` at or below `maxDetectAuc` together with a surface gap.                                                                                   |
| `ValueError: at least one field is required`           | Set `ambiguityTargets`.                                                                                                                                                                    |
| `analyse` is slow                                      | Pairing compares every pair of cluster representatives, and `analyse` repeats it for every threshold of the sweep. Lower `stepCount`, or give large datasets their own configuration file. |
