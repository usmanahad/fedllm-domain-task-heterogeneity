# Proposal-focused review

Reviewed September 11, 2026 against the proposal, both FedAvg notebooks, both
results folders, PLAN.md, the CPU symmetry audit, and the current runner.
The frozen baseline verification passed for all 12 result directories.

The main improvement is to reduce scope. Establish a clean domain non-IID result
under ordinary LoRA FedAvg, then test one affordable explanation. A new
aggregation method, complete representation mechanism, and 320-run confirmation
matrix are optional research extensions, not requirements for finishing this
project with Kaggle and approximately $7 of extra GPU credit.

## What is already useful

- The phase-1 notebook already compares pooled-data IID and domain-separated
  factor-FedAvg. Its final pooled evaluation NLL is 1.1467 versus 1.1544, a small
  gap of 0.0077. It is a one-seed native-data pilot with domain and task entangled.
- The later controlled matrix has three optimization seeds and four regimes.
  Domain-only recovery is 0.79 percentage points below IID, with a paired 95%
  interval of [-2.53, +0.95] points. This supports a small penalty in that setup.
- The first results folder is an earlier coupled pilot, not an extra independent
  replication to pool with the completed matrix.
- The token audit and LOO analysis are enough exploratory measurement for now.
  Shared-token associations are not causal evidence, and overlapping domain
  pairs are not independent experiments.

Keep the pooled multidomain IID baseline: it improves on the proposal's
single-domain IID comparison by holding the training pool constant. Describe
the setting as single-domain clients, without assuming zero vocabulary overlap
or proving it is the most severe possible non-IID setting.

## Recommended execution order

1. **Make one clean data version on CPU.** Remove known Finance and Medical
   instruction boilerplate before generating controlled windows. The existing
   audit finds instruction fragments in 44.7% of Finance reconstruction targets.
   Spot-check cleaned examples and source-disjoint splits, record immutable
   revisions and hashes, then freeze the pool. Preserve original source row IDs
   and split assignments where feasible. Keep eight-token targets, both tasks,
   16 clients, and the other settings initially; changing target length now adds
   another explanation for changed results. Compare new runs only with references
   and controls generated on the same clean pool.

2. **Run a four-run pilot on Kaggle.** Qwen 0.5B, one partition seed, one
   optimization seed: IID/domain-only crossed with factor-FedAvg/FedEx. Add one
   base evaluation and one compute-matched centralized reference, shared across
   those four conditions. This separates the domain penalty from dependence on
   the aggregation method. FFA and SVD can wait. Inspect correctness, learning
   curves, per-cell NLL and elapsed time before expanding, regardless of whether
   the hypothesis is supported.

3. **Repeat the core contrast.** Start with two additional partition seeds and
   one paired optimization seed per partition. Prioritize factor-FedAvg IID and
   domain-only; repeat FedEx as well if the pilot shows meaningful aggregation
   dependence. Three partitions are a modest replication, not automatic proof
   of equivalence. Report paired intervals and keep claims narrow if imprecise.
   Do not expand to every task regime or nested seed combination by default.

4. **Test one cheap explanation.** After the clean baseline, compare the existing
   shared prompt with one minimal prompt for IID and domain-only. Preserve the
   actual task, source text, and targets; reconstruction still needs an explicit
   missing-span marker. Use a common evaluation format and report format
   sensitivity. The effect of interest is the change in the non-IID-minus-IID
   gap, not absolute degradation. Two conditions are enough for the first screen.
   A null result is useful and should be reported; it does not block completion
   or justify trying many prompts until one works. This tests prompt commonality,
   not the entire natural-language symmetry hypothesis.

5. **Finish evaluation before adding mechanisms.** Evaluate selected, locked
   models on untouched controlled test data and one feasible native downstream
   task with its actual task metric. Add a second small model's IID/domain-only
   pair if compute permits. Downstream evaluation of a controlled-task-trained
   model is a transfer check; failure also may reflect training-task mismatch.
   Keep broad claims conditional on these checks.

## Small implementation details that could waste time

- `scripts/kaggle_run_primary.sh` names cached partitions only by regime and
  builds them only if absent. Changing `PARTITION_SEED` alone can reuse seed 42.
  Put data version and partition seed in partition/output paths and verify the
  loaded metadata. The underlying runner already records partition seed; a new
  partition framework is unnecessary.
- `save_checkpoints: false` and the compact archives prevent later evaluation
  of the trained model. Save the selected final adapter and, for FedEx, its
  cumulative base residual, or evaluate before discarding the live model.
  The current runner reports validation NLL; a test split existing on disk is
  not evidence that test evaluation happened.
- Time training and diagnostics separately. Full client-by-cell transfer and
  16-way LOO need not run for every performance screen. Keep them for selected
  mechanism runs, while preserving diagnostic/RNG settings within paired arms.
- Reuse base and centralized references when model, data, objective, budget,
  initialization and evaluation are unchanged. A changed intervention/objective
  needs an appropriate reference if normalized recovery is used.
- Report raw per-cell NLL and the paired domain-minus-IID gap alongside recovery.
  A one-sided lower-bound test establishes non-inferiority at its margin, not
  two-sided equivalence. Avoid fitting the proposed large mixed-effects model
  to a handful of partitions.

## Defer, and adjust the claims

Defer CKA/mediation, token-conditioned gradients, subspace ablation/rescue,
synthetic tokenizers, new mitigation algorithms, larger models, and broad stress
matrices. None should be a gate before the basic FedAvg comparison. If robustness
only appears under mild optimization, a later single local-step stress contrast
is more directly informative than a new aggregation method.

The proposal explicitly allows the useful conclusion that sophisticated
non-IID algorithms may be unnecessary. There is no need to invent a mitigation
when the measured penalty is already small.

Replace “FedEx prevents interference from causing loss” with “little final loss
was observed under FedEx.” Its causal advantage has not been measured here.
[FedEx-LoRA](https://aclanthology.org/2025.acl-long.67/) corrects factor-averaging
error, but algebraic exactness alone does not establish the performance effect
in this experiment. Likewise, favorable results do not yet establish language
symmetry as the cause or demonstrate superiority to CNNs; a cross-modal claim
needs a separate controlled comparison.

The practical deliverable is a reproducible domain-IID/non-IID comparison,
per-domain test results, a small replication, and one clearly interpreted
symmetry-related control. Mechanistic null results belong in that deliverable.
Use the existing long PLAN.md as a backlog, not a mandatory sequence. Its final
“Immediate next action” section also repeats baseline/token work already done.

## Implementation update: clean-v2 gate

The review is adopted with two qualifications: factor-FedAvg remains paired
with FedEx because conventional factor averaging is not exact effective-update
averaging, and the later minimal-prompt experiment will be described narrowly
as a prompt-commonality test.

Step 1 is now complete:

- [x] Added a versioned `clean_v2` adapter that removes only the three verified
  fixed Finance sentiment instructions and the verified fixed Medical
  truthfulness instruction from controlled source text. Native-task prompts are
  unchanged.
- [x] Pinned all four Hugging Face dataset revisions and the exact Qwen model/
  tokenizer revision used to construct the clean pool and initialize training.
- [x] Preserved legacy source IDs and used a committed row-index split anchor.
  Every still-eligible legacy row stays in its old split; ineligible rows are
  deterministically replaced from outside the old pool.
- [x] Kept both tasks, eight-token targets, exact cell counts, 16-client design,
  training budget, and model settings unchanged.
- [x] Verified 7,168 unique examples, 3,584 unique sources, zero source overlap
  across splits, and 512/128/256 examples per train/validation/test cell.
- [x] Verified that no known native instruction fragment remains in any prompt
  or target. A few ordinary Finance statements naturally contain sentiment
  words; these are reported separately and are not instruction leakage.
- [x] Rebuilt the pool twice byte-for-byte identically and froze its hashes.
- [x] Built and strictly audited clean-v2 IID and domain-only partitions for
  partition seed 42.
- [x] Made the runner reject a clean pool whose byte-level SHA-256 differs from
  the frozen clean-v2 value, even if a matching stale partition is present.
- [x] Updated partition/output paths to include data version and partition seed,
  preventing stale seed-42 partitions from being silently reused.
- [x] Fixed an additional reproducibility boundary: optimization seeds now
  govern LoRA initialization, Python/NumPy/PyTorch RNGs, client shuffling, and
  dropout. Each client resets its own RNG so enabling diagnostics cannot change
  the following training trajectory.
- [x] Added separate training, aggregation, diagnostic, validation, and total
  elapsed-time fields. Expensive transfer/gradient/LOO diagnostics are disabled
  in all four performance-screen arms and can be enabled later for one selected
  paired mechanism run.
- [x] Updated the local, upload-only Kaggle notebook to the four-run clean-v2
  pilot: IID/domain-only × factor-FedAvg/FedEx, optimization seed 42.

Clean-v2 canonical JSONL SHA-256:
`ab07a92891d0ccaeaa0a77ed570f5fbcf6c31395e3c1c5b79185e102b30d6418`.

Clean-v2 training-pool hash:
`d7c10649360a94bfa86554901d80ab3c17630465f38374922cec151f658f566f`.

Removing fixed instructions makes many short Finance statements ineligible
under the locked 24-context + 8-target requirement. Training-row retention is
40.4% for Finance and 94.1% for Medical; General and Code retain 100%. This is
explicitly recorded. Shortening context only for Finance was rejected because
it would create a new domain-specific task confound.

The next gate is the four-run Kaggle pilot in recommendation 2. Do not expand
the matrix until its correctness, elapsed time, learning curves, per-cell NLL,
and saved final artifacts have been inspected.
