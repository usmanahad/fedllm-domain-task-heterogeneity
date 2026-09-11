# Research Plan: Why Domain Non-IID Is Tolerable in Federated LLMs

## Status and how to use this plan

This document is the execution plan for extending the completed FlowerTune-based
study into a robust causal investigation of the language-symmetry hypothesis.
We will complete the stages in order. A later stage should not begin until the
acceptance gate for the current stage has passed.

Status notation:

- `[x]` completed and checked.
- `[ ]` not started.
- `[~]` partially completed or provisional.
- **Gate** means a decision point. The next stage starts only if the stated
  evidence and artifacts exist.

Current position: the baseline freeze and CPU symmetry audit are complete. A
versioned clean-v2 pool now removes the discovered Finance/Medical native-task
boilerplate, pins dataset and Qwen model/tokenizer revisions, preserves every eligible legacy source in
its original split, and has deterministic frozen hashes. Following
`PLAN_REVIEW.md`, the next gate is the small clean-v2 IID/domain-only ×
factor-FedAvg/FedEx pilot. The remainder of this document is a research backlog,
not a mandatory full matrix.

---

## 1. The research question in plain language

In federated learning, clients train locally and send model updates to a server.
The server combines those updates into one global model. The clients do not send
their private examples to the server.

The original question is:

> If different clients contain completely different subjects, such as medicine,
> finance, general instruction data, and programming code, why does the global
> language model not deteriorate as severely as non-IID image models often do?

The proposed explanation is **language symmetry**: different subjects still
share tokens, punctuation, grammatical patterns, prompt structures, and internal
representations. These shared elements may give otherwise different clients a
common learning direction.

The completed experiment found that domain non-IID caused little final
performance loss, but it did not prove why. The goal of the next work is to test
whether shared linguistic structure actually causes that robustness.

### Refined causal hypothesis

Raw token overlap cannot be the full mechanism. A transformer first converts
token IDs into embeddings and hidden representations. LoRA then changes
attention projections using gradients generated from those representations.

The precise hypothesis is therefore:

> Shared linguistic tokens and structures create a cross-domain shared
> representation and gradient subspace. This shared subspace gives clients a
> common descent direction, making domain heterogeneity relatively benign.
> Different tasks change the requested input-to-output mapping and produce
> task-specific directions that can conflict.

The proposed causal chain is:

```text
shared tokens and linguistic structures
                  ↓
similar hidden representations across domains
                  ↓
aligned gradients and effective LoRA updates
                  ↓
positive cross-domain functional transfer
                  ↓
small domain non-IID performance penalty
```

The research must test every arrow rather than infer the whole chain from final
validation loss.

---

## 2. Terms used throughout the project

### Client

A **client** represents one organization or device in federated learning. A
client receives the current global model, trains it on only its assigned local
examples, and returns a model update. In our simulation, clients run one after
another on the same GPU, but they are treated as separate data owners.

### Server

The **server** starts each round with the current global model. It sends that
model to every participating client, receives the clients' updates, and combines
them. The server never trains on the clients' private training rows directly.

### Federated round

A **round** is one complete cycle of:

1. Send the same global checkpoint to all selected clients.
2. Train each client locally for the configured number of steps.
3. Collect all client updates.
4. Aggregate them into the next global checkpoint.
5. Evaluate and record diagnostics.

The completed experiment used eight rounds and full participation, meaning all
16 clients participated in every round.

### Domain

A **domain** is the subject matter of the text. The four current domains are:

- **General:** general instruction-following text from Alpaca-GPT4.
- **Finance:** financial statements from FinGPT sentiment data. The sentiment
  label is excluded from the controlled-task source text.
- **Medical:** medical flashcard questions and answers.
- **Code:** programming instructions and generated code.

### Task

A **task** is what the model must do with the text. Domain and task are separate.
Medical and code examples can both use the same task even though their subject
matter is different.

The two controlled tasks are:

- **Continuation:** the model sees 24-96 context tokens and generates the next
  eight tokens.
- **Span reconstruction:** an internal eight-token span is replaced with
  `<MISSING>`. The model sees text on both sides and generates the removed span.

These are open-vocabulary tasks. The answer can be any tokens, rather than a
collapsed label such as `A/B/C/D`.

### Cell

A **cell** is one domain-task combination. Four domains multiplied by two tasks
produce eight cells:

```text
general/continuation           general/span_reconstruction
finance/continuation           finance/span_reconstruction
medical/continuation           medical/span_reconstruction
code/continuation              code/span_reconstruction
```

The controlled dataset contains, per cell:

- 512 training examples.
- 128 validation examples.
- 256 test examples.

All partition regimes reuse the same examples. Only the assignment of training
examples to clients changes.

### IID and non-IID

**IID** means that every client receives examples drawn from the same overall
distribution. **Non-IID** means that clients receive systematically different
distributions. In this project, the difference may come from domains, tasks, or
both.

### LoRA

**Low-Rank Adaptation (LoRA)** freezes the large pretrained model and learns two
small matrices, usually called `A` and `B`, for selected model layers. Their
product creates the effective weight change:

```text
effective LoRA weight change = scaling × B × A
```

The current model applies rank-16 LoRA to the query and value attention
projections. This makes client training and communication much cheaper than
fine-tuning every model parameter.

### Factor-wise FedAvg

**Factor-wise FedAvg** averages all client `A` matrices and all client `B`
matrices separately. This is easy but generally does not equal the average of
the actual products `B × A`.

### FedEx-LoRA

**FedEx-LoRA** computes the exact desired average in effective weight space and
adds the factor-averaging error as a residual to the frozen base weights. This
means the resulting effective model equals the weighted average of client LoRA
models up to floating-point precision.

### Gradient

A **gradient** is the instantaneous direction in which the loss says a parameter
should move. Gradients must be measured from the same global checkpoint to be
compared fairly.

### Client update or model delta

A **client update** is the change after several local optimizer steps. It is not
the same as a gradient. The project measures the effective LoRA update using the
induced `B × A` matrices.

### Gradient conflict

Two gradients are considered in **conflict** when their cosine similarity is
negative. This means following one direction would locally oppose the other.
Near-zero cosine means the directions are almost orthogonal, not necessarily
that they are harmless.

### Functional transfer

**Functional transfer** measures what a client update actually does to held-out
data. The update is applied by itself, and the change in validation loss is
measured on every cell:

```text
loss change = loss after client update - loss at the shared checkpoint
```

- Negative loss change means the client helped that cell.
- Positive loss change means the client harmed that cell.

### Leave-one-client-out effect

For **leave-one-client-out (LOO)** analysis, the server first evaluates the full
16-client aggregate. It then rebuilds the aggregate 16 times, omitting one
different client each time.

```text
LOO loss change = loss without client - loss with all clients
client harm score = -LOO loss change
```

- Positive harm means removing the client improved loss; the client was harmful.
- Negative harm means removing the client worsened loss; the client was helpful.

LOO is a marginal effect inside the aggregation process. It is stronger evidence
than judging a client from gradient cosine alone.

### Normalized recovery

Different cells have different raw NLL scales, so they should not be naively
averaged as if they had equal difficulty. For each cell:

```text
normalized recovery =
    (pretrained-base NLL - federated NLL)
    / (pretrained-base NLL - centralized NLL)
```

- `0` means federated training gained nothing over the pretrained model.
- `1` means it matched centralized training.
- Values above `1` mean it exceeded the centralized reference on that cell.

### Linguistic symmetry

**Linguistic symmetry** is a family of similarities rather than one number:

- **Lexical/token symmetry:** clients use some of the same subword tokens.
- **Structural symmetry:** clients share punctuation, word-order patterns,
  function words, delimiters, or prompt structure.
- **Semantic symmetry:** analogous expressions lead to similar meanings or
  embeddings.
- **Representational symmetry:** hidden activations occupy similar subspaces.
- **Gradient symmetry:** examples generate compatible learning directions.

### Ablation, negative control, dose-response, and rescue

- An **ablation** deliberately removes a suspected causal signal.
- A **negative control** makes a similarly sized change that should not remove
  the suspected mechanism.
- A **dose-response** uses several intervention strengths rather than only
  `on/off`.
- A **rescue** restores the removed signal. If the outcome also recovers, the
  causal explanation becomes much stronger.

---

## 3. Exactly what each current partition setting means

There are 4,096 controlled training examples: 512 examples in each of eight
cells. Every regime divides those same 4,096 examples across 16 clients. Each
client receives 256 examples. The amount of data therefore remains constant.

### IID

**Question isolated:** What happens when every client has the same mixture?

Each of the 16 clients receives approximately:

- 32 general continuation examples.
- 32 general reconstruction examples.
- 32 finance continuation examples.
- 32 finance reconstruction examples.
- 32 medical continuation examples.
- 32 medical reconstruction examples.
- 32 code continuation examples.
- 32 code reconstruction examples.

Every client sees every domain and both tasks. Client datasets differ only in
the particular rows they contain.

### Domain-only

**Question isolated:** What changes when clients have different subjects but
still perform both tasks?

The 16 clients are divided into four groups of four:

```text
clients 00-03: general only
clients 04-07: finance only
clients 08-11: medical only
clients 12-15: code only
```

Each client receives approximately:

- 128 continuation examples from its domain.
- 128 reconstruction examples from its domain.

The task mixture is constant; only the domain differs. This is the most direct
test of the original domain non-IID proposal.

### Task-only

**Question isolated:** What changes when clients perform different objectives
but see the same mixture of subjects?

The 16 clients are divided into two groups of eight:

```text
clients 00-07: continuation only
clients 08-15: span reconstruction only
```

Each client receives approximately 64 examples from each of the four domains,
all using its assigned task. The domain mixture is constant; only the task
differs.

### Coupled

**Question isolated:** What happens when domain and task are both separated?

There are eight domain-task cells and two clients per cell:

```text
clients 00-01: general continuation
clients 02-03: general reconstruction
clients 04-05: finance continuation
clients 06-07: finance reconstruction
clients 08-09: medical continuation
clients 10-11: medical reconstruction
clients 12-13: code continuation
clients 14-15: code reconstruction
```

Every client receives 256 examples from exactly one cell. This is the most
extreme current setting.

### Why all four are necessary

If we compared only IID with coupled, a performance difference could be caused
by domain, task, or their interaction. The four settings form a `2 × 2`
factorial design:

| Setting | Domain skew? | Task skew? |
|---|---:|---:|
| IID | No | No |
| Domain-only | Yes | No |
| Task-only | No | Yes |
| Coupled | Yes | Yes |

This allows separate estimates of:

- The domain-skew effect.
- The task-skew effect.
- The additional domain-by-task interaction.

---

## 4. What has already been established

- [x] The same example pool was used in all 12 runs.
- [x] Qwen2.5-0.5B-Instruct was run with three optimization seeds.
- [x] IID, domain-only, task-only, and coupled FedEx runs completed.
- [x] Base and centralized references completed for each seed.
- [x] True common-checkpoint gradients were measured at rounds 1 and 8.
- [x] Effective `B × A` updates were measured.
- [x] Cross-client functional-transfer matrices were measured.
- [x] Full and leave-one-client-out FedEx aggregates were measured at round 8.
- [x] All four regimes recovered approximately 70% of centralized improvement.
- [x] Domain-only recovery was 0.79 percentage points below IID.
- [x] Task-only recovery was 0.74 percentage points below IID.
- [x] Coupled recovery was 0.41 percentage points below IID.
- [x] Conflict was absent at round 1 and substantial at round 8.
- [x] In the coupled setting, clients helped the same task across domains but
  harmed the other task.
- [x] The full FedEx aggregate remained beneficial even when many individual
  client-cell effects were harmful.

Current defensible conclusion:

> Domain and task heterogeneity create measurable late-round interference, but
> FedEx prevents that interference from causing a practically meaningful final
> NLL loss in this Qwen2.5-0.5B controlled experiment. The LOO results suggest
> that task boundaries are more antagonistic than domain boundaries.

What is not yet established:

- [ ] Token or linguistic symmetry causes the robustness.
- [ ] Pretraining, model width, prompt templates, or FedEx do not fully explain
  the result.
- [ ] Ordinary factor-wise FedAvg is equally robust.
- [ ] The result replicates on a second model.
- [ ] The conclusion holds over independently sampled client partitions.
- [ ] The result holds on native downstream tasks.
- [ ] A proposed mitigation improves worst-cell outcomes.

---

## 5. Competing explanations that must be ruled out

The causal study must consider all of the following alternatives.

### Shared prompt boilerplate

Every continuation example begins with the same task instruction, and every
reconstruction example shares another instruction. The model may align because
of these templates rather than natural cross-domain language.

### Broad pretraining

Qwen was pretrained on diverse text. It may already understand all four domains,
making small local updates easy to combine even without meaningful linguistic
symmetry during federated training.

### Model capacity

Large or overparameterized models can be less sensitive to heterogeneous
updates. Width may explain robustness independently of language structure.

### LoRA bottleneck

Rank-16 LoRA restricts each client to a small update subspace. The restriction
may prevent clients from drifting far enough to cause large global damage.

### Exact FedEx aggregation

FedEx corrects the error introduced by separately averaging LoRA factors. The
observed robustness may depend on exact effective-weight aggregation.

### Mild optimization

Ten local steps may be too few to generate severe client drift. More local steps,
partial participation, or unequal client sizes may produce a different result.

### Controlled-task simplicity

Continuation and reconstruction are closely related language-modeling tasks.
The result may not transfer to classification, instruction following, reasoning,
or code execution.

### Fixed partition seed

The current three runs vary optimization randomness but keep the same client
assignment. They do not measure uncertainty over alternative client partitions.

Every later experiment is designed to isolate at least one of these explanations.

---

## 6. Preregistered hypotheses

### H1: domain robustness

When the task is held constant, domain-only FL is practically equivalent to IID
FL under the predefined normalized-recovery margin.

### H2: shared-token gradient alignment

Low-domain-information tokens, such as tokens common across all four domains,
produce more aligned cross-client gradients than high-domain-information tokens.

### H3: representation mediation

Clients with more similar hidden representations have more aligned gradients and
more positive functional transfer. Representation similarity statistically
mediates part of the relationship between linguistic symmetry and transfer.

### H4: necessity

Removing or downweighting shared linguistic signal widens the domain-only versus
IID performance gap more than a frequency-, difficulty-, and size-matched random
ablation.

### H5: dose-response

The domain non-IID penalty grows monotonically as shared linguistic signal is
reduced.

### H6: rescue

Restoring a shared representation or shared linguistic anchor after ablation
recovers a substantial portion of the induced non-IID penalty.

### H7: task specificity

Task-matched clients transfer positively across domains, while task-mismatched
clients generate more conflict and negative transfer.

### H8: aggregation dependence

If factor-wise FedAvg performs worse than FedEx under identical symmetry levels,
then aggregation correctness explains part—but not necessarily all—of the
observed robustness.

---

## 7. Evidence required for a causal claim

The project will not claim that linguistic symmetry causes robustness unless the
following evidence is obtained:

1. **Association:** measured symmetry predicts representation alignment,
   gradient alignment, and functional transfer.
2. **Specificity:** the relationship survives controls for domain, task, token
   frequency, base-model difficulty, sequence length, and prompt template.
3. **Necessity:** shared-signal ablation selectively worsens non-IID recovery.
4. **Negative-control separation:** a matched random ablation has a materially
   smaller effect.
5. **Dose-response:** stronger removal produces a larger non-IID penalty.
6. **Rescue:** restoring shared structure reverses a meaningful portion of the
   ablation effect.
7. **Replication:** the direction reproduces on at least two pretrained model
   families and several partition seeds.
8. **External validity:** at least one native FlowerTune evaluation agrees with
   the controlled-task result.

The causal claim must be weakened or rejected if the matched random control is
equally harmful, if only centralized performance changes, or if the effect
disappears on a second model.

### Working quantitative thresholds

These thresholds must be locked before confirmatory data is examined. Pilot data
may be used to revise them only if the revision is documented before the final
confirmation runs.

- Shared-signal ablation should widen the domain-only versus IID gap by at least
  `0.05` normalized recovery relative to the normal objective.
- The paired confidence interval for that difference-in-differences should
  exclude zero.
- A matched random ablation should change the gap by less than `0.02`, or the
  direct shared-versus-random contrast should exclude zero.
- At least four of five ordered symmetry doses should follow the predicted
  direction; the continuous dose slope must also be reported.
- A rescue should recover at least 50% of the loss introduced by the selected
  shared-signal ablation.
- The sign of the primary effect must reproduce on both Qwen and SmolLM2.
- Worst-cell outcomes must be reported even if the macro criterion passes.

---

# Execution stages

## Stage 0: freeze the completed baseline

### Goal

Preserve the existing result as a fixed baseline so later code changes cannot
silently alter the comparison.

### Tasks

- [x] Record the data-pool hash.
- [x] Record the Git commit used for all runs.
- [x] Validate all 12 result directories and LOO matrices.
- [x] Produce final-performance, conflict, transfer, and LOO visualizations.
- [x] Copy the existing analysis summary into a small immutable baseline
  manifest containing hashes of all 12 `summary.json` and `rounds.json` files.
- [x] Add a regression test that verifies the reported headline numbers from
  the frozen summaries.

### Acceptance gate

- The baseline manifest can detect any changed or missing result file.
- The headline table can be reproduced with one command.

### Compute

CPU only. No Kaggle or RunPod GPU time.

---

## Stage 1: implement symmetry measurement

### Goal

Measure token, structure, representation, and gradient symmetry without changing
training yet.

### 1A. Token statistics

For each domain and task, compute:

- Token counts and probability distributions.
- Pairwise Jensen-Shannon divergence.
- Weighted token-overlap mass.
- Vocabulary Jaccard overlap.
- Tokens per word or source unit.
- Token-domain mutual information.
- The percentage of tokens appearing in all four domains.
- Separate statistics for prompt/context tokens and target tokens.

#### Domain-information score

For token `t`, estimate how informative it is about domain `D`:

```text
domain information of token t = contribution of t to I(Token; Domain)
```

Intuitively:

- A token such as punctuation or a common article should have low domain
  information.
- A medical term or programming keyword should have high domain information.

Do not classify a token using raw frequency alone. Rare tokens can still be
shared, and frequent tokens can still be domain-specific.

### 1B. Structural statistics

Measure:

- Punctuation and delimiter distributions.
- Function-token sequences.
- Token n-gram overlap.
- Sequence and clause lengths.
- Shared prompt-template tokens.
- Natural-language POS patterns for general, finance, and medical text.
- Code-specific structures such as indentation, brackets, operators, and
  keywords.

Structural analysis must report natural-language and code categories separately.

### 1C. Representation capture

Pass fixed probe examples through the pretrained global model and save compact
activation summaries from selected layers:

```text
early layer:     close to lexical/embedding information
middle layer:    contextual representation
late layer:      information closest to the output prediction
```

For Qwen2.5-0.5B, initially capture four evenly spaced layers rather than every
layer. Store float32 summary matrices on CPU and discard full GPU activations
after each batch.

Compute:

- Linear CKA between every pair of domain-task cells.
- Principal angles between activation subspaces.
- Same-domain versus cross-domain similarity.
- Same-task versus cross-task similarity.

### 1D. Token-conditioned gradients

Recompute common-checkpoint gradients while allowing loss only on selected
target-token groups:

- Low-domain-information tokens.
- High-domain-information tokens.
- Punctuation/structural tokens.
- Frequency-matched random tokens.

Every comparison must use the same checkpoint, examples, number of target
tokens, and microbatch size.

### Deliverables

- [x] `SymmetryProfile` data structure.
- [x] `artifacts/symmetry/token_profiles.json`.
- [x] `artifacts/symmetry/cell_pair_metrics.csv`.
- [ ] `artifacts/symmetry/representation_cka.csv`.
- [ ] `artifacts/symmetry/token_conditioned_gradients.json`.
- [x] Unit tests using small hand-constructed token distributions.
- [ ] A visualization linking token overlap, CKA, gradient cosine, and transfer.

### Acceptance gate

- Repeated runs produce identical token statistics.
- Every cell has the expected sample count.
- CKA equals one for identical activation matrices in a toy test.
- Token groups are matched within predefined tolerances for count, frequency,
  base-model NLL, and sequence position.
- No test-set data is used to define token groups or interventions.

### Compute

- Token and structural statistics: CPU.
- Activation capture: Kaggle P100 is sufficient.
- Use RunPod only if P100 activation capture is too slow after profiling.

---

## Stage 2: repair baseline statistical and external-validity boundaries

### Goal

Confirm that the existing result is not an artifact of one model, one partition,
or FedEx.

### 2A. Partition uncertainty

Generate independent partitions with several seeds. A partition seed changes
which concrete examples each client receives while preserving the regime's
domain/task proportions.

Use a nested design:

- Partition seed: the primary independent replication unit.
- Optimization seed: repeated model-training randomness within a partition.

Start with three partition seeds and two optimization seeds. Increase the number
after measuring variance and performing a power calculation.

### 2B. Model replication

Run the locked controlled experiment on:

- Qwen2.5-0.5B-Instruct.
- SmolLM2-360M-Instruct.

Use Qwen2.5-1.5B only as a conditional scale check after the direction is stable.

### 2C. Aggregation controls

Compare, with identical clients and seeds:

- Factor-wise FedAvg.
- FedEx-LoRA.
- FFA-LoRA.
- SVD effective-update aggregation.

Do not compare different aggregators using different partition files.

### 2D. Native evaluation

Evaluate the final global model on the existing FlowerTune domain suites where
feasible. Controlled NLL remains the primary mechanism metric; downstream scores
provide external validity.

### Deliverables

- [ ] Partition-seed-aware runner.
- [ ] SmolLM2 P100/A100 configuration with the same diagnostics.
- [ ] Aggregator comparison table.
- [ ] Native-evaluation table.
- [ ] Per-example evaluation losses.

### Acceptance gate

- Domain-only versus IID has a confidence interval over independent partitions.
- The direction is reported separately for each model and aggregator.
- The result is not described as general LLM behavior unless both model families
  agree.

### Compute

Use Kaggle for Qwen/SmolLM screening. Spend RunPod time only on paired conditions
that cannot finish reliably on the P100.

---

## Stage 3: remove the shared-prompt confound

### Goal

Determine whether common task instructions, rather than natural linguistic
structure, explain the cross-domain alignment.

### Conditions

#### Shared template

All clients use the current identical continuation or reconstruction instruction.

#### Client-specific paraphrase

Each client group receives a semantically equivalent instruction written with
different words. The requested output and source text remain unchanged.

#### Minimal template

Use only the minimum delimiter needed to distinguish context from target.

#### No boilerplate

Where technically safe, train directly on source context and target without a
natural-language instruction.

### Controls

- Prompt tokens stay excluded from target loss in every condition.
- Prompt length is matched or included as a covariate.
- Target tokens, source rows, client assignments, and optimizer seeds are fixed.
- Evaluate all conditions using one common evaluation format as well as their
  native training format.

### Primary test

Estimate the interaction:

```text
(domain-only - IID under client-specific/minimal prompts)
minus
(domain-only - IID under shared prompts)
```

If the domain gap grows when the shared template is removed, prompt symmetry
explains part of the current robustness.

### Deliverables

- [ ] Deterministic prompt-policy interface.
- [ ] Shared, paraphrased, minimal, and no-boilerplate policies.
- [ ] Prompt hashes stored in every manifest.
- [ ] IID/domain-only screening results.

### Acceptance gate

- Every policy preserves the same target and source split.
- At least two human reviewers confirm that paraphrased instructions request the
  same operation.
- Any claimed prompt effect is a regime-by-prompt interaction, not merely worse
  loss under a harder prompt.

### Compute

Kaggle P100 first. This is a relatively cheap causal screen and should happen
before representation surgery.

---

## Stage 4: shared-token loss intervention

### Goal

Test whether shared target-token learning is necessary for domain robustness.

### Intervention

Assign every target-token occurrence a domain-information score. Weight the loss
from low-domain-information tokens by:

```text
shared-token weight λ = 0, 0.25, 0.5, 1.0, or 2.0
```

- `λ = 0` removes learning signal from shared tokens.
- `λ = 1` is the ordinary objective.
- `λ = 2` emphasizes shared tokens.

Normalize each batch so the total token weight remains constant. Otherwise a
smaller gradient could be mistaken for a symmetry effect.

Increase controlled target length from eight to approximately 24-32 tokens so
each example contains enough token categories for reliable weighting.

### Negative controls

For every shared-token intervention, create a random-token intervention matched
on:

- Number of token occurrences.
- Token frequency.
- Base-model token NLL or surprisal.
- Position within the target.
- Domain and task.

Also include a high-domain-information-token intervention. If all token removals
have the same effect, the result is general information loss rather than evidence
for linguistic symmetry.

### Primary causal estimand

For each intervention strength:

```text
domain penalty = domain-only normalized recovery - IID normalized recovery
```

Then compare how the domain penalty changes from `λ = 1`.

### Expected result if the hypothesis is correct

- Lower `λ` reduces representation CKA.
- Lower `λ` reduces cross-domain gradient cosine.
- Lower `λ` increases cross-domain functional harm.
- Lower `λ` makes the domain-only penalty more negative.
- Frequency/difficulty-matched random weighting has a smaller effect.
- Higher `λ` partially reverses these changes.

### Deliverables

- [ ] Per-token weighted-loss implementation.
- [ ] Deterministic matched-control sampler.
- [ ] Toy tests verifying exact loss normalization.
- [ ] Dose-response plots.
- [ ] Difference-in-differences analysis.

### Acceptance gate

- The ordinary `λ = 1` implementation reproduces the existing loss exactly.
- Matched controls meet all predefined balance tolerances.
- The shared-token intervention changes the non-IID gap beyond the matched random
  intervention, with uncertainty reported over partitions.

### Compute

Use the P100 for `λ = 0, 1, 2` screening. Run intermediate doses only if those
three indicate a monotonic relationship. Preserve RunPod budget for the selected
confirmatory doses.

---

## Stage 5: shared-representation subspace ablation and rescue

### Goal

Test the proposed internal mechanism directly: whether a cross-domain shared
hidden-state subspace supports federated robustness.

This is the best use of the limited A100 allocation because activation capture,
subspace estimation, and repeated intervention forwards are memory- and
throughput-heavy.

### Constructing the shared subspace

1. Use only training or validation probes, never the final test set.
2. Collect hidden states for low-domain-information tokens in all domains.
3. Center the activations by layer.
4. Estimate a low-rank cross-domain basis using PCA plus a cross-domain
   consistency criterion.
5. Choose the subspace rank using validation data before looking at test effects.

The projection matrix onto this basis is called `P_shared`.

### Ablation

At selected transformer layers, change a hidden state `h` to:

```text
h_ablated = h - α × P_shared × h
```

where:

- `α = 0` leaves the model unchanged.
- `α = 0.25, 0.5, 0.75` removes progressively more shared signal.
- `α = 1` removes the full estimated shared component.

### Required controls

- A random subspace with the same rank.
- A random subspace matched for removed activation energy.
- A domain-specific subspace.
- The same intervention under centralized, IID, and domain-only training.
- At least two different transformer layers.

The energy-matched random control is essential: removing any high-energy
activation direction can damage a model.

### Rescue

After identifying a shared-subspace ablation that selectively worsens non-IID
learning, test one of:

- Restore the projected component after local training.
- Add a small shared-anchor representation-consistency loss.
- Learn a frozen, client-invariant projection from a public anchor corpus.

The rescue must be evaluated with the same ablation strength and client
partitions.

### Expected causal pattern

```text
shared-subspace ablation
    > matched random ablation in non-IID penalty

increasing α
    → decreasing CKA and gradient alignment
    → increasing functional harm
    → increasing non-IID penalty

shared-subspace rescue
    → reverses a meaningful portion of all four changes
```

### Deliverables

- [ ] Memory-safe activation recorder.
- [ ] Shared-subspace estimator.
- [ ] Forward-hook intervention with exact enable/disable behavior.
- [ ] Energy-matched random-subspace generator.
- [ ] Ablation and rescue run summaries.
- [ ] Layer-by-layer causal plot.

### Acceptance gate

- The hook produces bitwise-identical outputs at `α = 0` within numerical
  tolerance.
- Removed energy is matched between shared and random controls.
- The shared ablation has a larger domain-non-IID-specific effect than controls.
- The rescue restores at least 50% of the induced normalized-recovery loss or
  produces a clearly bounded null result.

### Compute

Run the first complete pilot on RunPod A100. Do not spend A100 time on a large
matrix until one layer, rank, and ablation strength have passed the pilot.

---

## Stage 6: clean synthetic vocabulary-overlap experiment

### Goal

Create a setting where vocabulary overlap can be changed without inheriting all
the confounds of a pretrained tokenizer.

### Design

Train a small autoregressive transformer on two or four pseudo-domains derived
from the same underlying corpus. Construct tokenizers with controlled vocabulary
overlap:

```text
0%, 25%, 50%, 75%, and 100% shared vocabulary
```

Hold constant:

- Vocabulary size.
- Number of training tokens.
- Token-frequency distribution.
- Model architecture and parameter count.
- Sequence length.
- Semantic relationship between pseudo-domains.
- Centralized and federated optimization budgets.

Run centralized, IID, and domain-only training at each overlap level.

### Why this is supplementary

This arm gives clean causal control over token overlap, but a small from-scratch
model is not the same as a pretrained instruction model. It supports the
mechanism; the FlowerTune experiments establish real-model relevance.

### Deliverables

- [ ] Deterministic pseudo-domain generator.
- [ ] Controlled tokenizer builder.
- [ ] Vocabulary-overlap validation report.
- [ ] Overlap-to-CKA-to-gradient-to-recovery dose-response plot.

### Acceptance gate

- Token overlap changes while frequency and vocabulary-size controls remain
  balanced.
- The result is reported as synthetic mechanism evidence, not direct proof about
  deployed LLMs.

### Compute

CPU for data/tokenizer preparation. P100 or a cheaper RunPod GPU for training.
Do not use A100 time unless profiling demonstrates a clear benefit.

---

## Stage 7: develop a symmetry-aware mitigation

### Goal

Use the causal result to improve worst-cell performance rather than introducing
an unrelated aggregation heuristic.

### Proposed update decomposition

For client update `u_i`, estimate:

```text
shared component:  u_i_shared   = P_shared × u_i
specific component: u_i_specific = (I - P_shared) × u_i
```

Then:

1. Aggregate the shared component globally using exact FedEx.
2. Cluster or conflict-project the task-specific components.
3. Preserve incompatible residual components in task adapters or experts.
4. Select or mix task components at inference using task metadata or a small
   router.

### Minimum baselines

- Uniform FedEx.
- Factor-wise FedAvg.
- Magnitude-normalized FedEx.
- Global conflict projection.
- Task-clustered FedEx.
- Shared global adapter plus task-specific adapters.

### Why scalar client removal is not the preferred method

Current LOO results show that a client can help its own and same-task cells while
harming the other task. A single global weight labels the whole client as good or
bad and loses this structure.

ProToken-style attribution may still help identify which clients contributed to
specific output tokens, but it must be validated against the cell-specific LOO
matrix before it is used for weighting.

### Success criteria

A mitigation is considered useful only if it:

- Improves worst-cell normalized recovery by at least `0.05`.
- Has a paired confidence interval excluding zero.
- Reduces macro recovery by no more than `0.02`.
- Does not improve one task by sacrificing the other without reporting it.
- Remains beneficial on at least two partition seeds not used for method tuning.

### Deliverables

- [ ] Common aggregator interface implementation.
- [ ] Toy-matrix proof of the shared/specific decomposition.
- [ ] Coupled-regime comparison.
- [ ] Privacy and communication-cost accounting.

### Acceptance gate

- The method passes the predefined worst-cell and macro constraints on held-out
  partition seeds.

---

## Stage 8: external-validity and stress tests

### Goal

Identify where the domain-robustness conclusion stops being true.

### Stress axes

- Number of clients: 8, 16, and 32.
- Participation: 25%, 50%, and 100% per round.
- Local steps: 1, 10, and a larger drift-producing value.
- Unequal client sizes.
- Noisy or low-quality clients.
- Native FlowerTune tasks.
- A larger Qwen model if earlier effects are stable.
- Multilingual clients as an extreme low-overlap condition.
- Full fine-tuning or a higher-capacity adapter as a LoRA-bottleneck check.

Do not run the full Cartesian product. Change one stress axis at a time around a
locked baseline, then confirm only the conditions that reveal a boundary.

### Deliverables

- [ ] Boundary map showing when non-IID remains benign and when it becomes
  harmful.
- [ ] Native downstream evaluation.
- [ ] Final claim table separating supported, unsupported, and conditional
  conclusions.

### Acceptance gate

- The paper identifies at least one robustness region and one failure region, or
  reports a statistically bounded failure to find the latter.

---

## 8. Statistical analysis plan

### Primary unit of replication

The partition seed is the primary independent unit. Clients within a run share a
global checkpoint and must not be counted as independent replicates.

### Nested randomness

Use:

```text
partition seed
    └── optimization seed
          └── clients and evaluation cells as repeated observations
```

### Primary outcomes

1. Macro normalized recovery.
2. Worst-cell normalized recovery.
3. Domain-only minus IID recovery.
4. Functional-transfer loss-change matrix.
5. LOO client-by-cell harm matrix.

### Mechanism outcomes

1. Token overlap and token-distribution divergence.
2. Representation CKA and subspace angles.
3. True-gradient cosine and conflict fraction.
4. Effective-update cosine and cancellation ratio.
5. Cross-domain same-task versus cross-task transfer.

### Models

Use a mixed-effects model such as:

```text
recovery ~
    domain_skew
    × task_skew
    × intervention_strength
    × aggregation_method
    + model_family
    + (1 | partition_seed)
    + (1 | partition_seed:optimization_seed)
```

Plain-language interpretation:

- Fixed effects estimate the experimental conditions of interest.
- Random effects account for repeated results from the same partition or
  optimization run.

### Causal intervention test

The primary test is the interaction between non-IID status and the symmetry
intervention. An intervention that makes every model worse equally does not
support the hypothesis.

### Confidence intervals

- Use paired intervals whenever conditions share a partition and optimizer seed.
- Use hierarchical bootstrap intervals over partition, then optimization seed.
- Keep the preregistered `-0.05` macro and `-0.10` worst-cell margins.
- Correct secondary layer/token-category comparisons for multiple testing.
- Report effect sizes and intervals even when p-values are not significant.

### Power analysis

Run a small number of new partition seeds first, estimate between-partition
variance, and calculate the required confirmatory replication count before
starting the final matrix. Do not choose the final number of seeds from the
observed significance of the same confirmatory data.

---

## 9. Compute plan: Kaggle plus approximately $7 of RunPod credit

The user estimates that the available credit provides roughly four to five A100
hours, or longer on a weaker GPU. Actual available hardware and rates may change,
so the runner must benchmark observed examples/second and projected cost before
launching a matrix.

### General compute rules

- Data preparation, token statistics, plotting, and unit tests run on CPU.
- Kaggle P100 handles smoke tests, small Qwen/SmolLM training screens, and cheap
  causal conditions.
- RunPod A100 is reserved for activation-heavy representation experiments and
  tightly paired confirmatory runs.
- Never run the baseline on one GPU type and the intervention on another without
  a calibration pair. Precision and kernel differences can become confounders.
- When an A100 comparison is needed, run both sides of the paired comparison in
  the same session and software environment.
- Use BF16 on A100 only after confirming loss agreement with the current FP16
  implementation on a short calibration run.
- Cache datasets and models once per session.
- Save JSON, CSV, manifests, and compact activation summaries immediately after
  every completed condition.
- Do not save large adapter checkpoints unless a later analysis requires them.

### Proposed A100 time budget

#### Block A: setup and calibration - up to 30 minutes

- Install the locked environment.
- Run tests and LoRA algebra checks.
- Benchmark one client step, one full round, activation capture, and one
  intervention forward.
- Estimate whether the planned pilot fits the remaining time.

**Stop rule:** if environment setup or numerical validation fails, do not launch
training. Preserve the credit for a corrected session.

#### Block B: representation pilot - approximately 60 minutes

- Capture four selected layers for fixed probes.
- Estimate one candidate shared subspace and one energy-matched random subspace.
- Verify memory usage and numerical behavior at `α = 0` and `α = 1`.

**Stop rule:** do not train if the control does not match removed activation
energy or if `α = 0` changes outputs.

#### Block C: paired causal screen - approximately 90 minutes

Run only:

```text
IID baseline
domain-only baseline
IID shared-subspace ablation
domain-only shared-subspace ablation
IID matched-random ablation
domain-only matched-random ablation
```

Use one partition and optimization seed for this engineering/causal pilot.

**Go rule:** proceed only if the shared ablation changes the domain-only-minus-IID
gap more than the matched random ablation.

#### Block D: rescue and replication - approximately 90 minutes

If Block C passes:

- Run one rescue condition for IID and domain-only.
- Repeat the most informative baseline/ablation pair on a second partition seed
  or SmolLM2.

If Block C fails:

- Do not force a rescue result.
- Use the time to diagnose layer, rank, energy matching, and whether the shared
  subspace was estimated correctly.

#### Block E: contingency and export - at least 30 minutes

- Complete interrupted paired conditions.
- Run final validation.
- Package results and logs.
- Record GPU model, driver, CUDA, PyTorch, Transformers, PEFT, precision, wall
  time, and observed cost.

### Why the full confirmation matrix should not run on the $7 credit

A robust multi-model, multi-partition, multi-intervention matrix is too large for
four to five A100 hours. The credit should identify the correct intervention and
produce one clean pilot. Kaggle and later lab compute should perform replication
after the intervention has been locked.

### Concrete screening and confirmation matrices

The matrices below prevent an uncontrolled Cartesian-product explosion.

#### Screen 1: prompt confound on Kaggle

```text
model:              Qwen2.5-0.5B
aggregator:         FedEx
regimes:            IID, domain-only
prompt conditions:  shared, client-specific, minimal, no-boilerplate
partition seeds:    1
optimizer seeds:    1
total runs:         2 × 4 = 8
```

Only the two most informative prompt conditions proceed.

#### Screen 2: token intervention on Kaggle

```text
model:              Qwen2.5-0.5B
aggregator:         FedEx
regimes:            IID, domain-only
token conditions:   λ=0, λ=1, λ=2, matched-random ablation
partition seeds:    1
optimizer seeds:    1
total runs:         2 × 4 = 8
```

Intermediate doses `0.25`, `0.5`, and `0.75` are added only after the endpoint
screen shows the predicted direction.

#### Screen 3: representation intervention on RunPod

```text
model:              Qwen2.5-0.5B
aggregator:         FedEx
regimes:            IID, domain-only
conditions:         normal, shared-subspace ablation, matched-random ablation
partition seeds:    1
optimizer seeds:    1
total runs:         2 × 3 = 6
```

If this passes its gate, use remaining A100 time for the two rescue runs and one
paired replication. Do not add layers, ranks, or doses during the same
confirmatory test.

#### Locked confirmation after screening

For the one selected mechanism intervention:

```text
models:             Qwen2.5-0.5B, SmolLM2-360M
aggregators:        FedEx, factor-wise FedAvg
regimes:            IID, domain-only
conditions:         normal, selected ablation, matched control, rescue
partition seeds:    initially 5
optimizer seeds:    2 within each partition
```

The upper-bound run count is:

```text
2 models × 2 aggregators × 2 regimes × 4 conditions
× 5 partition seeds × 2 optimizer seeds = 320 runs
```

This is deliberately an upper bound, not a command to launch 320 runs. After
three partition seeds, perform the preregistered variance/power calculation. Drop
factor-wise FedAvg from the remaining mechanism runs if its algebraic boundary
is already decisive, and do not run conditions that failed screening. The final
confirmatory matrix and exclusion decisions must be frozen before examining its
outcomes.

---

## 10. Artifact and reproducibility requirements

Every run must record:

- Git commit.
- Complete configuration.
- Dataset names and immutable revisions.
- Example-pool hash.
- Partition file hash and partition seed.
- Optimization seed.
- Prompt-policy hash.
- Token-group definition and hash.
- Model and tokenizer revisions.
- Aggregation method.
- GPU type and numerical precision.
- Package versions.
- Number of clients and selected clients per round.
- Examples, non-padding tokens, and target tokens processed.
- Wall-clock time.

Every intervention must record:

- Intervention type and strength.
- Layer and subspace rank where applicable.
- Removed activation energy.
- Matched-control identifier.
- Whether the intervention was active in training, evaluation, or both.

Every final result archive should contain:

- `summary.json`.
- `rounds.json`.
- Per-example evaluation table.
- Symmetry metrics.
- Configuration and manifests.
- No raw private-source data.
- No model checkpoint unless explicitly required.

---

## 11. Decision table for interpreting results

| Observation | Interpretation |
|---|---|
| Shared-token ablation selectively widens the non-IID gap | Evidence that shared-token training is necessary. |
| Random matched ablation has the same effect | General information loss; token-symmetry claim not supported. |
| Shared prompt removal widens the gap | Existing robustness partly comes from experimental boilerplate. |
| Prompt removal changes absolute loss but not the non-IID gap | Prompt makes the task harder but does not explain federated robustness. |
| Shared-subspace ablation exceeds random controls | Evidence for a shared representation mechanism. |
| Rescue reverses the ablation effect | Strong causal evidence of sufficiency. |
| CKA changes but transfer does not | Representation metric may not capture the functionally relevant subspace. |
| Gradients conflict but aggregate remains beneficial | Conflict is real but is absorbed by aggregation or local-step dilution. |
| Factor-FedAvg fails while FedEx succeeds | Aggregation correctness is an important boundary. |
| Pretrained GPT-2 is robust but random-init GPT-2 is not | Pretraining creates part of the shared structure. |
| Wider models are more robust at equal symmetry | Capacity is an additional mechanism. |
| Same-task cross-domain transfer stays positive | Supports domain-general shared structure. |
| Effects disappear on native evaluation | Controlled NLL result lacks external validity and claims must be narrowed. |

---

## 12. Publication-level contribution target

The desired final paper should contain three connected contributions.

### Contribution 1: controlled phenomenon

A benchmark that independently varies domain and task heterogeneity while
holding the example pool and optimization budget fixed.

### Contribution 2: causal mechanism

Evidence that a shared linguistic representation/gradient subspace explains why
domain non-IID is tolerated, using measurement, matched ablation, dose-response,
and rescue.

### Contribution 3: mechanism-derived solution

A symmetry-aware aggregation or adapter method that preserves globally useful
linguistic directions while isolating conflicting task-specific directions.

The intended final thesis is:

> The severity of federated LLM non-IID is governed less by domain identity than
> by the shared representational structure between clients. Pretraining and
> common linguistic patterns create a shared optimization subspace that makes
> domain heterogeneity tolerable, while different task objectives create
> conflicting residual directions that may require task-aware aggregation.

---

## 13. Immediate next action

The CPU preparation gate is complete. Run the locked clean-v2 Kaggle pilot:

1. One Qwen2.5-0.5B optimization seed and partition seed 42.
2. IID and domain-only partitions only.
3. Factor-FedAvg and FedEx-LoRA, for four federated runs total.
4. One shared pretrained-base evaluation and one compute-matched centralized
   reference on the same clean pool.
5. Stop after the pilot and inspect integrity, elapsed time, per-cell NLL,
   learning curves, aggregation dependence, and saved final artifacts before
   authorizing any replication or prompt intervention.

The local `kaggle-p100-runner.ipynb` is configured for exactly this gate and
remains excluded from Git.

---

## References that directly inform the plan

- Kornblith et al., [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html), ICML 2019. Introduces CKA for comparing neural representations.
- Karimireddy et al., [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html), ICML 2020. Formalizes client drift from heterogeneous data and introduces a control-variate correction.
- Tang et al., [Virtual Homogeneity Learning](https://proceedings.mlr.press/v162/tang22d.html), ICML 2022. Demonstrates that a shared virtual dataset can align heterogeneous federated representations.
- Singhal et al., [FedEx-LoRA](https://aclanthology.org/2025.acl-long.67/), ACL 2025. Provides exact effective-weight aggregation for federated LoRA.
- Jian and Liu, [Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](https://proceedings.mlr.press/v267/jian25a.html), ICML 2025. Motivates model width as a competing explanation.
- Wang et al., [Optimizing Cross-Client Domain Coverage for Federated Instruction Tuning](https://aclanthology.org/2025.findings-emnlp.52/), Findings of EMNLP 2025. Finds domain coverage can matter more than heterogeneity.
- Kallini et al., [False Friends Are Not Foes: Investigating Vocabulary Overlap in Multilingual Language Models](https://aclanthology.org/2025.findings-emnlp.1153/), Findings of EMNLP 2025. Supplies a controlled vocabulary-overlap and representation-analysis template.
