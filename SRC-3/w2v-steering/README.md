# Wav2Vec2 Steering

Wav2Vec2 XLS-R frontend with an AASIST backend, trained for speech anti-spoofing. The XLS-R weights are trainable.

## Research reports

- [Wav2vec2–AASIST experiments](docs/REPORT.md)
- [ALM research recap](docs/ALM_REPORT.md)
- [Native SASV reasoning and steering](docs/ALM_SASV_EXPERIMENTS.md)
- [WildSpoof 2026 and ElevenLabs external transfer protocol](docs/REPORT.md#8-external-wav2vec2-transfer-wildspoof-2026-and-elevenlabs-set)
- [ALM experiment status and restart guide](docs/ALM_EXPERIMENTS.md)

## Getting started

### Install Pixi

[Pixi](https://pixi.sh) creates the environment locked in `pixi.lock` (Python 3.12, PyTorch, Transformers).

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Open a new shell so `pixi` is on `PATH`.

### Install the environment

From the repository root:

```bash
pixi install
```

### Train on ASVspoof 5

The dataset root is `configs/data/default.yaml` (`ASVspoof5.train.metadata.txt`, `flac_T`, and the dev split under `flac_D`).
Training runs for 10 epochs in bf16 with a per-device batch size of 64 and 8 data-loader workers. Logs and checkpoints are written every 500 steps.
Dev EER is the checkpoint metric (`metric_for_best_model: eer`, `greater_is_better: false`), evaluated on the same interval.
The best checkpoint and the latest checkpoint are kept (`save_total_limit: 2`), and the best checkpoint is loaded when training finishes (`load_best_model_at_end: true`).
`ckpt_path` is empty by default, which starts a new run.
Set it to a `checkpoint-*` directory to resume.

```bash
pixi run python -m src.train
```

Logs, Trackio files, and the checkpoint are written under `logs/runs/`.

### Evaluate with one configurable entry point

All wav2vec2–AASIST evaluations use `eval.py`. Select a preset for the model and
dataset; dataset-specific readers and scoring utilities remain in their packages.

```bash
pixi run python eval.py evaluation=checkpoint ckpt_path=/path/to/checkpoint-750
pixi run python eval.py evaluation=asvspoof5
pixi run python eval.py evaluation=wildspoof
pixi run python eval.py evaluation=voxenes
pixi run python eval.py evaluation=mlaad
```

`python -m src.eval` is equivalent. Presets also cover `elevenlabs`,
`speaker_elevenlabs`, `speaker_wildspoof`, `speaker_asvspoof5`, and parallel
ASVspoof continuation. MLAAD defaults to the ASVspoof 5 A21 comparison.
Existing `steering.*`, `data.*`, and checkpoint overrides still apply.

See [the evaluation API guide](docs/EVALUATION_API.md) for presets, Python API,
preparation/smoke checks, device selection, output directories and resume commands.
ALM evaluation and adapter-fitting commands retain their existing interfaces.

### A21 few-shot steering experiment

`src.steering.run` adapts inference through a frozen detector using labeled A21 support.
This is a **few-shot adaptation protocol using part of the official evaluation split**,
not an untouched ASVspoof 5 benchmark result. A21 and A22 are related ToucanTTS variants;
A31 applies Malacopula to A22.

The experiment defaults to `checkpoint-3000` in the baseline run, selected by its saved
dev EER (19.88%). The final `checkpoint-28500` has saved dev EER 34.09%.
`steering.checkpoint` selects the checkpoint independently of the existing `ckpt_path`.
The detector architecture comes from that run's saved Hydra config, its weights are loaded
strictly, and its parameters remain frozen. The cached Hugging Face encoder config must
already exist; this entrypoint does not download pretrained weights.

Run the bounded smoke experiment:

```bash
pixi run python -m src.steering.run exp_name=sae_a21_smoke
```

Select an available compute device with `steering.device`. The default is CPU.
The defaults in `configs/steering/a21.yaml` use one transformer layer, 32 training clips,
16 labeled support clips, 16 tuning clips, and a 50-step SAE fit. These budgets validate
execution only. Obtain approval before expanding to full training.

The protocol hashes evaluation speakers into disjoint support, tuning, and test pools.
Only A21 and bonafide recordings enter support or tuning; all 16 evaluation attacks enter
the held-out comparison. Codec variants are deduplicated by source ID within each attack.
Speaker, recording, and source overlap across selected partitions causes an error. Audio
read failures also stop execution instead of substituting a different waveform or label.
Preprocessing matches the baseline's center crop, zero padding, and mean centering.

The per-token TopK SAE is fitted on balanced bonafide/spoof training clips. This is an
adaptation of the [Whisper steering method](https://arxiv.org/pdf/2606.07473), not an exact
reproduction of its pretrained BatchTopK SAEs. A standardized logistic probe fitted on
support selects sparse feature signs. Multiplicative steering uses
`h' = h + scale * W_dec(z * (exp(alpha * signs) - 1))`, preserving the reconstruction
residual and making zero strength exactly the baseline. The dense comparator uses the
support spoof-minus-bonafide mean direction. All interventions apply to every input
without inference-time access to labels or attack IDs.

Layer indices are one-based transformer block outputs, before final encoder normalization.
`steering.layers` supports a layer sweep. Layer and strength selection uses A21 tuning EER,
then minimum cost, with bonafide rejection and dev EER constraints. Ties prefer smaller
interventions, and zero strength is always available. Selection is saved before test audio
is loaded. The test comparison includes baseline, dense steering, SAE steering, SAE
reconstruction alone, and shuffled-feature steering at the selected SAE strength.

Each run saves:

- `manifest.json`, `.hydra/config.yaml`, and `provenance.json`: exact samples, configuration,
  and checkpoint SHA-256.
- `sae_layer_*.safetensors`, `directions_layer_*.safetensors`, and `sae_layer_*_fit.json`:
  dictionaries, directions, fit losses, and dev reconstruction error.
- `selection.json`: every candidate, retention checks, selected settings, and a single
  threshold chosen from the unmodified dev scores.
- `scores.json` and `report.json`: aligned held-out scores and pooled/per-attack EER,
  minimum detection cost, fixed-threshold cost, spoof acceptance, bonafide rejection,
  and changes from baseline. EER changes use percentage points; rates use fractions.

Metrics reuse the existing repository functions; the same bonafide cohort is used for
every attack. The smoke test's tiny cohorts do not establish effectiveness. Larger studies
need multiple support seeds, speaker-level uncertainty estimates, and codec/silence/energy
controls motivated by [Rubio et al.](https://www.isca-archive.org/odyssey_2026/rubio26_odyssey.pdf).
The current runner caches selected audio and activations in memory, so scale budgets with
that constraint in mind.

The approved expanded pilot is configured separately:

```bash
pixi run python -m src.steering.run steering=a21_pilot exp_name=sae_a21_pilot
```

It fits layers 6, 12, and 18 on 1,024 training clips for 500 SAE steps per layer.
Support and tuning each contain 128 A21 and 128 bonafide clips. The dev retention cohort
contains 64 clips per group; the held-out cohort contains 256 bonafide clips and 256 clips
for each of 16 attacks. It uses the original seed and extends the smoke subsets, so this
is an exploratory single-seed result, not an independent confirmation of the smoke run.

The completed `logs/runs/2026-10-09/sae_a21_pilot_13-48-02` run selected layer 6,
SAE strength 1.0, and 16 features. Held-out EER changed from 8.59% to 4.10% on A21,
19.14% to 11.52% on A22, and 12.50% to 10.16% on A31. Pooled EER changed from
18.90% to 13.60%; shuffled-feature steering remained near baseline. EER improved on
15 of 16 attacks, with a 0.39 percentage-point regression on A32. At the frozen dev
threshold, bonafide rejection increased from 8.20% to 12.50%, and fixed-threshold cost
worsened on six attacks. See `comparison.csv` for all attacks and `bootstrap.json`
for paired speaker-bootstrap intervals conditional on this one fitted intervention.
Further validation should repeat support seeds and assess dev-only threshold calibration.

### Supervised A21 fine-tuning comparator

```bash
pixi run python -m src.steering.finetune steering=finetune_a21 exp_name=finetune_a21
```

This comparator updates all wav2vec2+AASIST parameters from the same baseline checkpoint,
using the exact 128 A21 and 128 bonafide support recordings saved in the expanded SAE pilot.
It uses five epochs, AdamW at `1e-6`, effective batch size 16 (microbatches of two), bf16,
gradient checkpointing, and the same deterministic waveform preprocessing without augmentation.
A two-update fitting smoke precedes training; model weights, buffers, optimizer, and seed are
reset before the actual five-epoch fit. Batch-normalization statistics update during training.

`finetune_a21` selects the best trained epoch by A21 tuning EER, then minimum cost.
`finetune_guarded` additionally enforces the SAE experiment's dev EER and bonafide rejection
limits, allowing the original baseline as a fallback. Neither selector accesses held-out scores.
The new run contains the original baseline/SAE results plus both supervised comparators in
`report.json`, `scores.json`, and `comparison.csv`; the original SAE run is preserved.
The same original dev threshold is used for fixed-threshold metrics, which can expose a need
for recalibration after supervised fine-tuning. Selected weights are saved as
`best_a21.safetensors` and, if a trained guarded candidate is selected, `guarded.safetensors`.

Run `logs/runs/2026-10-09/finetune_a21_14-03-35` selected epoch 5. On the same held-out
cohort, A21/A22/A31 EER was 5.47%/16.21%/10.74%, respectively, and pooled EER was 16.00%.
The corresponding SAE+steering values were 4.10%/11.52%/10.16% and 13.60% pooled.
These are single-run point estimates. Supervised fine-tuning shifted score calibration:
99.22% of held-out bonafide recordings fell below the unchanged baseline threshold.
No trained epoch passed the fixed-threshold retention limits, so `finetune_guarded`
selected epoch zero. The trained model's ranking metrics and the guarded fallback are
reported separately; using the trained model operationally requires threshold calibration.

### MidSteer-inspired covariance transport and compute comparison

```bash
pixi run python -m src.steering.run_midsteer steering=midsteer_a21 exp_name=midsteer_a21
```

This adapts the rank-one case of [MidSteer, Eq. 19](https://arxiv.org/html/2605.05220v3)
to detection: transport A21 support representations toward known training-spoof
representations, not bonafide toward spoof. The implementation is independent;
the reference [controller](https://github.com/Atmyre/MidSteer/blob/0f3b31e15cdda6ad0d46167e10319e896d6f1541/core/controller.py)
also motivates clipping negative source projections in the second variant.
Neither variant uses an SAE or updates detector weights.

For source direction `s = mean(A21) - anchor`, target-minus-source `d`, and
regularized second moment `C`, the adapter is:

```text
q = solve(C, s) / (s.T @ solve(C, s))
h' = h + strength * d * ((h - anchor).T @ q)
```

`midsteer_affine` uses the mean and covariance of an equal-prior mixture of
bonafide, known training spoof, and A21 support tokens. The ridge is 0.01 times
the average covariance diagonal. `midsteer_anchored_gated` instead anchors at the
bonafide mean, uses the second moment around that anchor, and clips negative
projections to zero. Gating is nonlinear: the paper's affine optimality guarantee
does not apply, and fixing the anchor does not guarantee unchanged bonafide scores.
Both use 32 equally spaced tokens per clip and store just three 1,024-dimensional
vectors. The temporary covariance matrix is needed only during fitting.

The existing manifests, original dev threshold, and retention limits are reused.
The predeclared sweep is layers 6/12/18 and strengths 0/0.25/0.5/1. Selection is
saved before held-out audio is loaded. A22/A31 never enter fitting or selection.
This remains an exploratory few-shot experiment on a reused evaluation cohort,
not an untouched official benchmark. Original results and weights are preserved;
the new `comparison.csv` includes all previous methods and both new variants.

`compute.json` separates model/audio loading, activation extraction, closed-form
fitting, candidate evaluation, and final testing. A 500-step, one-layer SAE fit is
replayed on cached tokens solely for timing; it does not replace selected weights.
Earlier SAE and supervised run durations are extracted from logs and explicitly
include startup, data I/O, fitting, validation, and checkpoint writing. They are
not controlled training-only measurements or evidence of a cross-run speed ratio.

Inference is measured on identical preloaded audio with FP32, batch size four,
one warm-up per method, and five randomized rounds, including the saved A21-trained
model. New benchmark reports include explicit `mean_*` and `std_*` fields alongside
the legacy median fields for milliseconds per clip, throughput, real-time factor,
overhead, and extra peak tensor allocation. Standard deviation is calculated across
rounds with `ddof=1`; one round yields an unavailable std, not zero. Throughput is
calculated per round before averaging, and overhead is paired to the baseline from
the same round. Timings include
CPU-to-device transfer and returning scores, but exclude audio/checkpoint loading.
These synchronized wall times are subject to shared-server contention; they are
not energy, GPU busy time, or measured whole-model FLOPs. Analytical adapter costs
and artifact sizes are reported separately. If selection chooses zero strength,
an additional nonzero cost-only benchmark prevents confusing fallback latency
with the cost of an active adapter. No zero-overhead weight-folding claim is made
for these residual-block hooks.

Run `logs/runs/2026-10-09/midsteer_a21_14-35-11` selected affine layer 6,
strength 1, and anchored/gated layer 6, strength 0.5. The historical pilot cohort
contains 256 bonafide and 256 recordings per attack. The table below restores
single EER point estimates, not bootstrap means. These are pilot results, not
the full-evaluation results described in the next section. Latencies are
arithmetic means across five measurement rounds on an H100, processing
4.0375-second clips; they are throughput-normalized milliseconds per clip at batch
size four, not single-request latency. Models were fitted once. Earlier bootstrap
and timing-dispersion artifacts are retained as optional analyses, not the main report.

| Method | A21 EER % | A22 EER % | A31 EER % | Pooled EER % | Mean ms/clip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 8.59 | 19.14 | 12.50 | 18.90 | 14.336 |
| SAE + steering | 4.10 | 11.52 | 10.16 | 13.60 | 14.601 |
| A21 fine-tuning | 5.47 | 16.21 | 10.74 | 16.00 | 14.435 |
| MidSteer-inspired affine | 1.37 | 9.77 | 10.55 | 14.78 | 14.386 |
| Anchored/gated transport | 4.10 | 12.30 | 11.13 | 15.23 | 14.354 |

Original point estimates favor affine transport over SAE steering on A21/A22,
and favor SAE on A31 and pooled EER. At the unchanged operating threshold, bonafide rejection
is 8.20% baseline, 12.50% SAE, 10.94% affine, and 12.50% gated; supervised
fine-tuning remains poorly calibrated at 99.22%. These are exploratory single-seed
results, not evidence that either method dominates across attacks.

Both transport variants across all three layers took **0.51 s** of closed-form
fitting, including token preparation and transfer. Activation extraction took
**50.80 s**, and the candidate sweep took **200.68 s**; total adaptation through
selection was **277.96 s**, excluding the SAE timing replay. The replay measured
**1.80 s** for one SAE's 500 optimizer steps on already cached tokens. Historical
startup-through-selection times were **172.32 s** for the SAE/dense sweep and
**180.70 s** for supervised fine-tuning; different server loads and work scopes
prevent using these totals as a controlled speed comparison.

The selected transport artifact is **12,496 bytes** (3,072 fitted scalars), versus
**67,146,132 bytes** for the SAE dictionary (16,785,408 trained parameters) and
**1,263,629,896 bytes** for the supervised detector checkpoint (315,885,962 model
parameters). The SAE figure excludes its separately stored feature direction;
all adapters also require the unchanged baseline checkpoint. Mean paired affine
overhead was +0.36%, versus +1.85% for SAE (five timing rounds).
This supports negligible overhead in this run, not a literal zero-cost claim.
Per-attack EER/minimum cost/fixed-threshold cost are in `comparison.csv`;
`compute.json` and `compute.csv` contain stage timings and raw inference rounds.

### Full evaluation groups, without subsampling

The full-evaluation runner restores the existing frozen detectors and adapters;
it does not retrain, retune, bootstrap, or deduplicate codec variants. Each attack
EER uses **every eval recording of that attack and all 138,688 eval bonafide
recordings**. A21/A22/A31 contain 34,360 / 34,380 / 33,956 recordings respectively.
Pooled EER uses all 680,774 protocol recordings (542,086 spoof and 138,688 bonafide).
The same waveform preprocessing, FP32 precision, batch size four, and metric
implementations are retained. A strict pilot-score equivalence smoke precedes
full scoring. All protocol recordings were checked to have audio files on disk.

```bash
pixi run python eval.py evaluation=asvspoof5
```

Audio is streamed and scores checkpointed in protocol order under `chunks/`.
`progress.json` records completion and throughput; final `report.json` and
`comparison.csv` are written only after every recording has been scored.
The full run is `logs/runs/2026-10-09/full_eval_15-17-10`; consult its
`progress.json` for live status rather than the historical pilot table above.
Resume an interrupted run by supplying `steering.resume_dir=PATH_TO_RUN` with
the same scoring configuration. Source, checkpoint, selection, and protocol
hashes are verified before existing chunks are reused.

The run can also be continued with `eval.py evaluation=asvspoof5_parallel` after stopping
the original single-worker scorer. Its coordinator preserves the completed
prefix, assigns nonoverlapping batch-aligned ranges, and records assignments in
`parallel_plan.json`. Worker status and logs are under `parallel/`; the original
`progress.json` becomes an aggregate progress report. Original code, model,
selection, and protocol hashes are checked before launch. A separate smoke check
on each worker verifies the restored scores. The coordinator combines chunks in
official protocol order and writes final metrics only after complete, gap-free,
nonoverlapping coverage. Resume a parallel run using the same coordinator and
device assignments, not the single-worker command while shards are incomplete.

Two explicitly labeled cohorts are reported from the same scores:

- `full_eval`: all official eval rows, including the previously used support and
  tuning data. This answers the all-recordings comparison but is not an untouched
  benchmark for the adapted models.
- `heldout_speakers`: all 390,670 eval recordings in the original reserved test
  speaker pool. Both support and tuning speaker pools are excluded; overlap with
  fitting/selection data is checked before scoring.

All nine saved methods are included. Exact zero-intervention/epoch-zero fallbacks
reuse baseline scores. Existing pilot and bootstrap artifacts are preserved and
must not be mistaken for full-evaluation results while scoring is in progress.

### Optional uncertainty analyses (not the main report)

For the completed pilot run, the original artifacts remain unchanged and the derived
`compute_mean_std.json` / `compute_mean_std.csv` provide the updated summaries,
including source checksums. Regenerate these without training or inference:

```bash
pixi run python -m src.steering.summarize_compute logs/runs/2026-10-09/midsteer_a21_14-35-11/compute.json
```

For EER test-sampling uncertainty, resample whole held-out speakers with replacement
using the same draws for every method and attack. All selected recordings of a drawn
speaker are included with that speaker's multiplicity. The default budget is 1,000
valid draws, seed 20261009. Draws lacking a required class are rejected jointly and
counted. The original metric implementations are reused unchanged, and all saved
point metrics are recalculated and checked before resampling.

```bash
pixi run python -m src.steering.bootstrap logs/runs/2026-10-09/midsteer_a21_14-35-11 --replicates 1000 --seed 20261009
```

`bootstrap_mean_std.csv` reports original EER alongside bootstrap mean, sample std
(`ddof=1`), and percentile 95% interval for every method and attack. The JSON report
also includes paired EER differences against baseline and SAE, input/source hashes,
and validation metadata. `bootstrap_replicates.npz` preserves the per-replicate EERs
and speaker draws; array axis names and speaker order are in the JSON report.
These statistics describe test-speaker sampling uncertainty conditional on the
already fitted model, support data, and selected configuration—not variation across
training seeds. Bootstrap means may differ from original point estimates. Mean ± std
is not a confidence interval; the separate percentile intervals are exploratory and
not corrected for multiple comparisons. No training, tuning, or selection is rerun
on held-out test recordings.

Focused checks:

```bash
pixi run -e test pytest tests/test_steering.py -q
pixi run -e test pytest tests/test_midsteer.py tests/test_finetune.py -q
pixi run -e test pytest tests/test_compute.py -q
pixi run -e test pytest tests/test_bootstrap.py -q
pixi run -e test pytest tests/test_full_eval.py -q
pixi run -e test pytest tests/test_parallel_eval.py -q
pixi run -e dev format src/steering
pixi run -e dev format tests/test_steering.py
```
