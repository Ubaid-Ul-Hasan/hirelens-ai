# Error Analysis — HireLens AI Scoring (Step 18)

**Generated from an actual run** of `python -m src.evaluation.run_experiment`
against `data/evaluation/eval_pairs.json` (8 hand-authored pairs; see that
file's `_meta` for caveats about sample size and label provenance). This run
used the TF-IDF baseline and hybrid matcher's **skill/lexical components
only** — sentence-transformers is not installed in the environment this was
generated in, so all "semantic" components fell back to the TF-IDF baseline
(see `weights_used.semantic_source` on any result). Numbers below are real
output, not illustrative placeholders.

## Headline results

| Method          | Bucket Accuracy | Mean Bucket Distance | Rank Correlation |
|-----------------|-----------------|-----------------------|-------------------|
| Baseline TF-IDF | 0.25            | 1.125                 | 0.869             |
| Hybrid          | 0.375           | 0.625                 | 0.945             |
| **Final Score** | **0.75**        | **0.25**              | **0.945**         |

(Bucket accuracy = predicted score lands in the same weak/moderate/strong
range as the label. Rank correlation checks ordering, independent of where
the bucket boundaries sit.)

## Finding 1 — TF-IDF's raw cosine similarity is not on the same scale as the bucket thresholds

Every baseline TF-IDF score in this run is below 0.30, including for pairs
labeled "strong" (e.g. `pair_01_strong_ml_fit` scored **0.155**). This is
not a ranking failure — TF-IDF's rank correlation (0.869) shows it *orders*
pairs sensibly — it's that short documents with a lot of non-overlapping
wording naturally produce low bag-of-words cosine similarity even when the
underlying content matches well. **Conclusion:** TF-IDF's absolute score
should not be read as a percentage match ("15% fit"); it's only meaningful
as a baseline for comparison and for ranking, which is exactly the role the
spec assigns it. The bucket-accuracy metric is somewhat unfair to TF-IDF
for this reason — a threshold recalibrated specifically for TF-IDF's scale
would score much better, but that isn't the point of a baseline comparison.

## Finding 2 — the Final Score's "strong" threshold (0.70) is a hair too conservative for two genuinely strong fits

Two pairs authored as unambiguous strong fits landed just under the 0.70
"strong" cutoff:

- `pair_02_strong_backend_fit` → **0.653**
- `pair_07_strong_data_engineering_fit` → **0.652**

Both are the two wrong predictions behind Final Score's 0.75 (6/8) bucket
accuracy. Looking at the score breakdown for `pair_02`: the candidate meets
100% of required *and* preferred skills, but the job description doesn't
state a minimum years-of-experience, so `experience_score` uses the neutral
1.0-when-unstated rule rather than a differentiated boost, and
`semantic_score` is only the TF-IDF fallback (low-scale, per Finding 1) —
pulling the blended score down even though the actual fit is excellent.
**Conclusion:** this is a real, reproducible interaction between two design
choices that are individually correct (neutral credit for unstated
requirements; honest TF-IDF fallback when no embedding model is available)
but combine to under-score strong fits when a JD is skills-only with no
explicit experience bar. This is worth revisiting once real embeddings are
in place (Finding 1 + this effect should both shrink, since embedding
similarity for genuinely well-matched text sits much higher than TF-IDF's).
Recorded here rather than silently tuning the threshold against 2 examples
from an 8-pair dev set, which would be overfitting to noise.

## Finding 3 — Hybrid outperforms raw TF-IDF but underperforms Final Score on this set

Hybrid's bucket accuracy (0.375) sits between baseline and final, which
tracks: it incorporates required/preferred skill coverage (a strong,
well-calibrated signal here) but weights semantic similarity at 30%, and
with semantic unavailable that weight gets redistributed onto lexical/
section-relevance signals that share TF-IDF's low-scale problem (Finding
1). Rank correlation for Hybrid (0.945) already matches Final Score,
suggesting the *ordering* is right and only the reported number's scale is
off in this no-embeddings configuration.

## What this does NOT show

With 8 pairs, these numbers are a sanity check, not a validated accuracy
claim. In particular:
- No claim is made about performance on real resumes/JDs outside this set.
- No claim is made about performance once embeddings are available — that
  requires re-running this exact script in an environment with
  sentence-transformers installed and comparing against these baseline
  numbers.
- The "expected_bucket" labels are the author's own judgment when writing
  the dataset, not independently verified ground truth.

## Recommended next step

Re-run `python -m src.evaluation.run_experiment` in an environment with
sentence-transformers installed, and diff the resulting `hybrid` and
`final_score` metrics against the numbers in this file. If Finding 2's two
mis-bucketed pairs move into the "strong" range once real semantic
similarity replaces the TF-IDF fallback, that's confirmation the
degradation-fallback behavior (not the scoring formula itself) explains
the gap — and no threshold tuning is needed.
