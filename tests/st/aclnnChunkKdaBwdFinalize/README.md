# ChunkKdaBwdFinalize ATK CPU Dual Validation

The final paired-BF16 A5 build passes 200/200 original CPU dual-reference
cases (1400 output comparisons). Its wheel SHA256 is
`a7330980c047ca3cadbb472e3746eb7bafd883152604b7dd3bf881888e785c62`.
This is single-operator validation, not full backward integration or CI approval.
All seven outputs are bytewise deterministic over 20 repeated launches for
B1/H96/T8192 and H3 packed lengths [64,1]. The measured 8.79% performance
regression has been accepted for this precision fix.

## Reference and Matrix

`kernel_ac_torch_ref.py` is the unmodified, GPU-aligned user reference,
SHA256 `fb8c6f9fc1b19031c9487e6394c2792b1f1d4e5b5cfed38d8f51772284d820c4`.
The executor checks this hash before using it. The golden uses FP64 without
output downcasting. The CPU benchmark uses the reference's FP32/BF16 path,
not the kernel's factored Cube operand model.

The fixed matrix has 100 dense and 100 packed cases, including empty packed
sequences, all major tail boundaries, optional q/k normalization, BF16/FP32
a_log, and B=1/H=96/T=8192 and 16384. `case_profiles.json` records actual
shapes and attributes. The two marker tensors and case_id in ATK JSON are
transport metadata, following the prepare delivery; they are not the public
aclnn tensor signature. Both executors construct identical immutable inputs
from the case seed. The PyAclnn executor constructs all real ABI arguments
and invokes the operator through ATK's AclnnBaseApi.

Accuracy ratios remain 5 / 1.5 / 1.5. Other ATK defaults, including the small
value error count check, are not disabled or relaxed. Process exit zero or
`success 200` means execution completed, not accuracy passed.

## A5 Execution

Activate the configured A5 development environment first, then select an
isolated wheel site containing the intended finalize OPP:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export FINALIZE_WHEEL_SITE=/absolute/path/to/validated/wheel_site
python generate_finalize.py
python generate_finalize.py --check
python -m atk case -f aclnn_finalize.yaml -p . -s 20260906
bash run_atk.sh smoke_case.json
bash run_atk.sh atk_finalize.json
python inspect_report.py /absolute/path/to/report.xlsx --summary
```

The YAML generator is a smoke entry point. `generate_finalize.py` creates the
full fixed 200-case matrix and extracts its first case as `smoke_case.json`.
`run_atk.sh` retains `-cp` and the plugin directory. It explicitly configures
the deterministic run mode because ATK 26.7.8 rejects its default None in
this single-process path. This mode alone does not prove repeated-launch
determinism. The runner sets process-local OPP/library paths and never
installs into the global CANN environment.

`diagnose_case0.py` compares the failing single-token dq against factored and
unfactored CPU expressions. It is diagnostic only and does not replace the
acceptance reference or alter its thresholds.
