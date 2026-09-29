# DASE7506 Mini Project 1

This submission contains a compact causal language model trained from scratch on the supplied WikiText-2 training split. It keeps the supplied data, BPE-2048 tokenizer, evaluator, baseline configuration, and 256-token causal-window protocol unchanged.

The final model combines three changes:

- rotary position embedding (RoPE) in self-attention;
- a 12,000-step continuation from a newly trained 1,200-step RoPE checkpoint, with late-checkpoint weight averaging; and
- a causal in-window repetition distribution selected on the validation split and mixed with the neural prediction.

The submitted CPU FP32 test result is **1.632893777909 BPB**, compared with **2.101260438087 BPB** for the supplied baseline. The final model uses 1,055,488 trainable parameters. Its reported median CPU scoring-time ratio is 2.244x, peak evaluation RAM is 1.577 GiB, and uncompressed inference assets occupy 4.051 MiB. These values satisfy the limits of 5x baseline CPU time, 4 GiB RAM, and 64 MiB of inference assets.

## 1. Environment setup

Python 3.12 was used for the recorded run. Open PowerShell in the extracted `code` directory and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The recorded environment used Python 3.12.10, PyTorch 2.7.1+cpu, NumPy 2.5.3, and tokenizers 0.21.4.

## 2. Evaluate the submitted checkpoint

No retraining is required. Run the supplied tests and then evaluate the matching checkpoint:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/retrain-selected-12000/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The evaluator writes `runs/retrain-selected-12000/test_cpu_fp32.json`. The expected fields are:

```text
bpb:        1.632893777909
targets:    428405
utf8_bytes: 1292013
protocol:   7506-mp1-wt2-v2
```

BPB, rather than `token_ppl`, is the submission metric. Small floating-point differences may occur across platforms. The final checkpoint SHA-256 is:

```text
d3e5c75ecd51b20ab4bf6668113530faae8c4725c7144db714ca71b09a17a569
```

## 3. Reproduce training from random initialization

The following commands reproduce the complete three-stage procedure. Use new output directories, because `train.py` refuses to overwrite a non-empty run directory unless continuation is explicitly resumed.

### Stage 1: train the RoPE model for 1,200 steps

```powershell
.\.venv\Scripts\python.exe train.py --implementation student --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --eval-every 300 --run-dir runs/reproduce-initial
```

### Stage 2: continue for 12,000 steps and average late checkpoints

```powershell
.\.venv\Scripts\python.exe train.py --stage continue --parent runs/reproduce-initial/checkpoint.pt --run-dir runs/reproduce-extended-12000 --steps 12000 --batch-size 32 --seed 23 --threads 4 --device cpu --precision fp32 --lr 0.0006 --eval-every 600 --average-start 6000 --average-every 100
```

If this stage is interrupted after `resume.pt` has been written, rerun the same command with `--resume`. The initial stage does not support resume.

### Stage 3: select the repetition mixture on validation

```powershell
.\.venv\Scripts\python.exe train.py --stage select --checkpoint runs/reproduce-extended-12000/checkpoint.pt --output-dir runs/reproduce-selected-12000 --device cpu
```

This stage evaluates a fixed validation grid and does not update the neural weights. It selects `copy_alpha=0.25`, maximum suffix order 2, and smoothing 2.0.

### Evaluate the reproduced model

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/reproduce-selected-12000/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The selected checkpoint ancestry processed 108,134,400 training targets: 9,830,400 in Stage 1 and 98,304,000 in Stage 2. Training is stochastic across environments, so the packaged checkpoint should be used when reproducing the submitted score exactly.

Run `train.py --help`, `train.py --stage continue --help`, or `train.py --stage select --help` for the complete argument lists.

## 4. Experimental controls

The required evidence is summarized below. Model development and mixture selection used validation only.

| Experiment | Training targets | Validation BPB | Purpose |
|---|---:|---:|---|
| Supplied GPT baseline | 9,830,400 | 2.0710836385 | Initial reference |
| RoPE model | 9,830,400 | 1.9223254395 | Equal-training-budget comparison |
| Final neural weights, repetition disabled | 108,134,400 | 1.6615916362 | Same-weight mechanism ablation |
| Same weights, repetition enabled | 108,134,400 | 1.6186115943 | Selected final predictor |

To retrain the supplied baseline under the same short budget:

```powershell
.\.venv\Scripts\python.exe train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --run-dir runs/reproduce-baseline
```

The complete repetition grid is stored in `runs/retrain-selected-12000/validation_grid.json`. Setting `copy_alpha=0` disables the repetition distribution while preserving the same neural weights and ancestry.

## 5. Model and interface

`student.py` implements a four-layer, width-128 decoder-only Transformer with four attention heads and context length 256. RoPE is applied to query and key vectors. Token embeddings and the output projection share weights.

For each prediction position, the optional repetition component searches only earlier positions in the current input window for matching one- or two-token suffixes. Earlier continuation tokens form a causal probability distribution, which is mixed with the neural softmax. It does not access future tokens, retain state across windows, or store validation or test answers.

The required interface is:

- `build_model(config)` constructs the model;
- `forward(ids)` returns logits with shape `[batch, time, 2048]`; and
- `predict_log_probs(ids)` returns normalized natural-log probabilities with the same shape.

Training disables the repetition mixture. The final selected checkpoint enables it through its saved configuration.

## 6. Repository contents

| Path | Purpose |
|---|---|
| `student.py` | Submitted RoPE model and causal repetition predictor |
| `train.py` | Initial training, continuation, averaging, and validation selection |
| `runs/retrain-selected-12000/checkpoint.pt` | Final checkpoint used for the reported score |
| `runs/retrain-selected-12000/test_cpu_fp32.json` | Recorded full-test result |
| `runs/retrain-selected-12000/validation_grid.json` | Validation-only repetition ablation and selection |
| `model.py` and `configs/baseline.json` | Unmodified supplied baseline |
| `common.py` and `evaluate.py` | Unmodified data checks and evaluation protocol |
| `data/` | Supplied dataset, tokenizer, and manifest |
| `tests/` | Contract and student-model tests |
| `checks/RESULTS.json` | Consolidated verification results |
| `check_checkpoint.py` | Checkpoint contract and causality checks |
| `measure_cpu.py` | Windows CPU-time and process-tree RAM measurement |
| `DASE7506_MP1_Report_CHANG_Yuan.pdf` | Final four-page report |
| `EXPERIMENT_LOG.md` | Development history, modifications, and supporting rationale |

Protected course files, including `common.py`, `evaluate.py`, the data, tokenizer, baseline configuration, and official contract test, were not modified.

## 7. Verification and resource measurement

The packaged checkpoint can be checked with:

```powershell
.\.venv\Scripts\python.exe check_checkpoint.py --checkpoint runs/retrain-selected-12000/checkpoint.pt --output checks/recheck/contract.json
```

On Windows, CPU and process-tree RAM can be remeasured with:

```powershell
.\.venv\Scripts\python.exe measure_cpu.py --checkpoint runs/baseline/checkpoint.pt --split test --output checks/recheck/baseline-1.json
.\.venv\Scripts\python.exe measure_cpu.py --checkpoint runs/retrain-selected-12000/checkpoint.pt --split test --output checks/recheck/student-1.json
```

For the reported comparison, baseline and student measurements were alternated for three runs. The median student/baseline scoring-time ratio was 2.2438495x, the conservative maximum-student/minimum-baseline ratio was 2.2857405x, and peak process-tree RAM was 1.577129 GiB. `measure_cpu.py` is Windows-specific; `/usr/bin/time -v` may be used for process resource measurement on Linux.

## 8. Training cost and limitations

The recorded 1,200-step parent training took 392.23 seconds and the 12,000-step continuation took 3,706.25 seconds on the development machine with four CPU threads. These times depend on hardware and software versions.

## 9. AI assistance disclosure

Codex assisted with implementation, experiment execution, debugging, verification, reproducibility checks. ChatGPT was also used to explain the model and training concepts. Student reviewed the full implementation, commands, experimental records, and reported results. All BPB, timing, memory, and asset measurements reported here were produced by local execution of the supplied evaluation workflow rather than estimated by AI.

An earlier model had been evaluated on the test split before the final 12,000-step continuation was requested. For the final run, the training schedule was fixed in advance, all model and mixture selection used validation, and the code and checkpoint were frozen before the final test evaluations. Repeated final test runs were used only for reproducibility and resource measurement.
