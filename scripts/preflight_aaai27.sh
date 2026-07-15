#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: scripts/preflight_aaai27.sh [--output-dir DIR] [--require-clean-release]

Run the auditable AAAI-27 software/environment/data preflight without training.
The three local benchmark directories under datasets/ must be present.
Set PYTHON_BIN to force an interpreter. If it is unset, the local project
interpreter is used only when it exists; otherwise `python` is resolved from
PATH and validated before any checks run.
EOF
}

output_dir=""
require_clean_release=0
while (($#)); do
  case "$1" in
    --output-dir)
      if (($# < 2)) || [[ -z "$2" ]]; then
        echo "--output-dir requires a non-empty value" >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    --require-clean-release)
      require_clean_release=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

default_project_python="$(dirname "$repo_root")/BasicIRSTD/infrarenet/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_candidate="$PYTHON_BIN"
  python_selection="environment:PYTHON_BIN"
elif [[ -x "$default_project_python" ]]; then
  python_candidate="$default_project_python"
  python_selection="local_project_interpreter"
else
  python_candidate="python"
  python_selection="PATH_fallback"
fi

if [[ "$python_candidate" == */* ]]; then
  if [[ ! -x "$python_candidate" ]]; then
    echo "PYTHON_BIN is not an executable file: $python_candidate" >&2
    exit 2
  fi
  # Preserve the final virtualenv symlink so Python still discovers pyvenv.cfg.
  python_bin="$(realpath -s "$python_candidate")"
else
  python_bin="$(command -v "$python_candidate" || true)"
  if [[ -z "$python_bin" ]]; then
    echo "PYTHON_BIN command was not found: $python_candidate" >&2
    exit 2
  fi
fi
export PYTHON_BIN="$python_bin"

original_cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-<unset>}"
export CUDA_VISIBLE_DEVICES=0,1,2

# Fail before creating an evidence directory if the interpreter cannot provide
# the packages required by this repository's preflight.
"$python_bin" - <<'PY'
import sys

missing = []
for name in ("torch", "pytest"):
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit(
        f"Selected interpreter is unusable ({sys.executable}): " + "; ".join(missing)
    )
PY

if [[ -z "$output_dir" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  output_dir="$repo_root/outputs/preflight/$timestamp"
elif [[ "$output_dir" != /* ]]; then
  output_dir="$repo_root/$output_dir"
fi
output_dir="$(realpath -m "$output_dir")"

if [[ "$output_dir" == "$repo_root" ]]; then
  echo "The evidence directory cannot be the repository root" >&2
  exit 2
fi
if [[ "$output_dir" == "$repo_root"/* ]]; then
  output_relative="${output_dir#"$repo_root"/}"
  if ! git check-ignore -q -- "$output_relative"; then
    echo "An in-repository evidence directory must be Git-ignored: $output_dir" >&2
    exit 2
  fi
fi

if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to mix evidence into a non-empty directory: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"

tmp_root="$(mktemp -d /tmp/rc_irstd_preflight.XXXXXX)"
export PYTHONPYCACHEPREFIX="$tmp_root/pycache"
unset PYTEST_ADDOPTS || true

status_file="$output_dir/RUN_STATUS.txt"
printf 'RUNNING\n' > "$status_file"

finalize() {
  rc=$?
  trap - EXIT
  if ((rc == 0)); then
    status="PASS"
  else
    status="FAIL(exit=$rc)"
  fi
  {
    printf '%s\n' "$status"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$status_file"
  rm -rf -- "$tmp_root"
  exit "$rc"
}
trap finalize EXIT

log_step() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" \
    | tee -a "$output_dir/steps.log"
}

log_step "record repository state"
git rev-parse --show-toplevel > "$output_dir/git_toplevel.txt"
if [[ "$(realpath "$(cat "$output_dir/git_toplevel.txt")")" != "$repo_root" ]]; then
  echo "Script is not running from the expected Git worktree" >&2
  exit 2
fi
git rev-parse HEAD > "$output_dir/git_commit.txt"
git branch --show-current > "$output_dir/git_branch.txt"
git status --porcelain=v1 --untracked-files=all > "$output_dir/git_status_porcelain.txt"
git diff --binary --no-ext-diff HEAD > "$output_dir/git_diff.patch"
git diff --stat HEAD > "$output_dir/git_diff_stat.txt"
git diff --check | tee "$output_dir/git_diff_check.txt"

if ((require_clean_release)); then
  if [[ -s "$output_dir/git_status_porcelain.txt" ]]; then
    echo "Clean-release preflight requires an empty Git status" >&2
    exit 1
  fi
  if ! git tag --points-at HEAD | grep -Fxq 'aaai27-rc-irstd-v5-rc4'; then
    echo "Clean-release preflight requires tag aaai27-rc-irstd-v5-rc4 at HEAD" >&2
    exit 1
  fi
  if [[ ! -s outputs/release/RC-IRSTD_v5_rc4.zip ]] \
    || [[ ! -s outputs/release/RC-IRSTD_v5_rc4.zip.sha256 ]]; then
    echo "Clean-release preflight requires the source archive and SHA file" >&2
    exit 1
  fi
  sha256sum -c outputs/release/RC-IRSTD_v5_rc4.zip.sha256 \
    | tee "$output_dir/source_archive_validation.txt"
fi

: > "$output_dir/untracked_sha256.tsv"
while IFS= read -r -d '' path; do
  digest="$(sha256sum -- "$path" | awk '{print $1}')"
  size="$(stat -c '%s' -- "$path")"
  printf '%s\t%s\t%s\n' "$digest" "$size" "$path" \
    >> "$output_dir/untracked_sha256.tsv"
done < <(git ls-files --others --exclude-standard -z | sort -z)

log_step "record interpreter and host environment"
{
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repository=%s\n' "$repo_root"
  printf 'output_dir=%s\n' "$output_dir"
  printf 'python_selection=%s\n' "$python_selection"
  printf 'PYTHON_BIN=%s\n' "$python_bin"
  printf 'PYTHONPYCACHEPREFIX=/tmp/rc_irstd_preflight.<ephemeral>/pycache\n'
  printf 'CUDA_VISIBLE_DEVICES_original=%s\n' "$original_cuda_visible_devices"
  printf 'CUDA_VISIBLE_DEVICES_effective=%s\n' "$CUDA_VISIBLE_DEVICES"
  uname -a
} > "$output_dir/environment.txt"
"$python_bin" --version > "$output_dir/python_version.txt" 2>&1
"$python_bin" -m pip freeze > "$output_dir/pip_freeze.txt"

log_step "verify and record CUDA devices"
"$python_bin" - <<'PY' | tee "$output_dir/torch_environment.txt"
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("visible_device_count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, properties.total_memory)
if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
    raise SystemExit("AAAI-27 preflight requires visible CUDA GPUs 0, 1, and 2")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L > "$output_dir/nvidia_smi_list.txt"
  nvidia-smi -i 0,1,2 \
    --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader \
    > "$output_dir/nvidia_smi_query.csv"
else
  printf 'nvidia-smi not found; see torch_environment.txt\n' \
    > "$output_dir/nvidia_smi_list.txt"
fi

log_step "compile Python sources with bytecode redirected to /tmp"
"$python_bin" -m compileall -q \
  -x '(^|/)(\.git|datasets|outputs|repro_runs|\.pytest_cache)(/|$)' \
  "$repo_root" \
  2>&1 | tee "$output_dir/compileall.log"

log_step "run full pytest suite with the selected interpreter"
"$python_bin" -m pytest -q \
  -o "cache_dir=$tmp_root/pytest_cache" \
  2>&1 | tee "$output_dir/pytest.log"

log_step "validate repository shell scripts"
: > "$output_dir/shell_validation.log"
while IFS= read -r -d '' script; do
  if [[ ! -x "$script" ]]; then
    echo "Shell script is not executable: $script" >&2
    exit 1
  fi
  bash -n "$script"
  printf 'PASS\t%s\n' "$script" >> "$output_dir/shell_validation.log"
done < <(git ls-files -co --exclude-standard -z -- '*.sh' | sort -z)

log_step "validate source/config JSON only"
"$python_bin" - "$repo_root" <<'PY' \
  | tee "$output_dir/json_validation.log"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
source_roots = (
    root / "configs",
    root / "data_ext",
    root / "evaluation",
    root / "losses",
    root / "model",
    root / "rc",
    root / "rc_irstd",
    root / "scripts",
    root / "tests",
    root / "utils",
)
paths = set(root.glob("*.json"))
for source_root in source_roots:
    if source_root.is_dir():
        paths.update(source_root.rglob("*.json"))
for path in sorted(paths):
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)
    print("PASS", path.relative_to(root))
print(f"validated_json_files={len(paths)}")
PY

log_step "verify frozen official-train development splits"
"$python_bin" -m scripts.freeze_official_train_splits \
  --dataset NUAA-SIRST=datasets/NUAA-SIRST \
  --dataset NUDT-SIRST=datasets/NUDT-SIRST \
  --dataset IRSTD-1K=datasets/IRSTD-1K \
  --quarantine-config configs/aaai27_near_duplicate_quarantine.json \
  --output-dir splits/aaai27_v2 \
  --repository-root . \
  --seed 42 \
  --detector-diagnostic-fraction 0.20 \
  --meta-validation-fraction 0.20 \
  --context-size 32 \
  --query-size 64 \
  --check \
  2>&1 | tee "$output_dir/frozen_split_validation.log"

log_step "recompute frozen dataset byte and geometry contract"
"$python_bin" -m scripts.freeze_dataset_contract \
  --dataset NUAA-SIRST=datasets/NUAA-SIRST \
  --dataset NUDT-SIRST=datasets/NUDT-SIRST \
  --dataset IRSTD-1K=datasets/IRSTD-1K \
  --split-manifest splits/aaai27_v2/manifest.json \
  --repository-root . \
  --output audits/aaai27/dataset_contract_v1.json \
  --check \
  2>&1 | tee "$output_dir/dataset_contract_validation.log"

log_step "recompute original and effective near-duplicate audits"
"$python_bin" -m scripts.audit_near_duplicates \
  --dataset NUAA-SIRST=datasets/NUAA-SIRST \
  --dataset NUDT-SIRST=datasets/NUDT-SIRST \
  --dataset IRSTD-1K=datasets/IRSTD-1K \
  --repository-root . \
  --output "$output_dir/near_duplicates_original_recomputed.json" \
  2>&1 | tee "$output_dir/near_duplicates_original.log"
cmp \
  audits/aaai27/near_duplicates_original_official_splits_v2.json \
  "$output_dir/near_duplicates_original_recomputed.json"

"$python_bin" -m scripts.audit_near_duplicates \
  --dataset NUAA-SIRST=datasets/NUAA-SIRST \
  --dataset NUDT-SIRST=datasets/NUDT-SIRST \
  --dataset IRSTD-1K=datasets/IRSTD-1K \
  --split-role development_train \
  --split-role test \
  --split-file NUAA-SIRST:development_train=splits/aaai27_v2/nuaa-sirst/effective_development_train.txt \
  --split-file NUDT-SIRST:development_train=splits/aaai27_v2/nudt-sirst/effective_development_train.txt \
  --split-file IRSTD-1K:development_train=splits/aaai27_v2/irstd-1k/effective_development_train.txt \
  --repository-root . \
  --output "$output_dir/near_duplicates_effective_recomputed.json" \
  --require-pass \
  2>&1 | tee "$output_dir/near_duplicates_effective.log"
cmp \
  audits/aaai27/near_duplicates_effective_splits_v2.json \
  "$output_dir/near_duplicates_effective_recomputed.json"

log_step "validate frozen analysis plan"
analysis_plan_args=(
  --plan configs/aaai27_analysis_plan.json
  --repository-root .
  --output "$output_dir/analysis_plan_audit.json"
)
if ((require_clean_release)); then
  analysis_plan_args+=(--require-gate-minus-one)
fi
"$python_bin" -m scripts.validate_aaai27_analysis_plan \
  "${analysis_plan_args[@]}" \
  2>&1 | tee "$output_dir/analysis_plan_validation.log"

log_step "validate frozen Stage-1 pilot matrix and materialized invocations"
pilot_matrix_args=(
  --matrix configs/aaai27_stage1_pilot_matrix.json
  --analysis-plan configs/aaai27_analysis_plan.json
  --repository-root .
  --output "$output_dir/stage1_pilot_matrix_audit.json"
)
if ((require_clean_release)); then
  pilot_matrix_args+=(--require-release-artifacts)
fi
"$python_bin" -m scripts.validate_stage1_pilot_matrix \
  "${pilot_matrix_args[@]}" \
  2>&1 | tee "$output_dir/stage1_pilot_matrix_validation.log"

log_step "validate pyproject console-entrypoint routing"
"$python_bin" - "$repo_root" <<'PY' \
  | tee "$output_dir/entrypoint_validation.log"
import importlib
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1]).resolve()
with (root / "pyproject.toml").open("rb") as handle:
    project = tomllib.load(handle)
scripts = project["project"]["scripts"]
strict_expected = {
    "rc-irstd-audit": "scripts.audit_aaai_protocol:main",
    "rc-irstd-train-detector": "scripts.train_multisource_tail:main",
    "rc-irstd-export-scores": "evaluation.export_score_maps:main",
    "rc-irstd-export-labels": "evaluation.export_label_maps:main",
    "rc-irstd-build-source-reference": "rc.build_source_reference:main",
    "rc-irstd-build-meta": "rc.build_meta_episodes:main",
    "rc-irstd-train-calibrator": "rc.train_calibrator_risk_aligned:main",
    "rc-irstd-apply-calibrator": "rc.online_adapter:main",
    "rc-irstd-evaluate-adapter": "evaluation.evaluate_adapter_output:main",
    "rc-irstd-threshold-sweep": "evaluation.threshold_sweep:main",
}
for name, expected in strict_expected.items():
    if scripts.get(name) != expected:
        raise ValueError(f"strict entrypoint misrouted: {name}: {scripts.get(name)!r}")
for name, target in scripts.items():
    if name.startswith("rc-irstd-reference-"):
        if not target.startswith("rc_irstd."):
            raise ValueError(f"reference entrypoint escaped rc_irstd: {name}: {target}")
    elif target.startswith("rc_irstd."):
        raise ValueError(f"unprefixed entrypoint routes to reference package: {name}")
for name, target in sorted(scripts.items()):
    module_name, attribute = target.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"entrypoint is not callable: {name} -> {target}")
    print("PASS", name, target)
print(f"validated_entrypoints={len(scripts)}")

dispatcher = (root / "scripts" / "start_training.sh").read_text(encoding="utf-8")
dispatcher_fragments = {
    "detector": 'exec "$ROOT/scripts/train_rc_3gpu.sh"',
    "calibrator": 'exec "$ROOT/scripts/train_calibrator_risk_aligned.sh"',
    "export-scores": '-m evaluation.export_score_maps',
    "export-labels": '-m evaluation.export_label_maps',
    "build-source-reference": '-m rc.build_source_reference',
    "build-meta": '-m rc.build_meta_episodes',
    "online": '-m rc.online_adapter',
    "audit": '-m scripts.audit_aaai_protocol',
    "reference": 'exec "$ROOT/scripts/start_training_reference.sh"',
}
for mode, fragment in dispatcher_fragments.items():
    if fragment not in dispatcher:
        raise ValueError(f"strict dispatcher route missing for {mode}: {fragment}")
print(f"validated_dispatcher_routes={len(dispatcher_fragments)}")
PY

log_step "audit official train/test identity, separation and mask geometry"
"$python_bin" -m scripts.audit_aaai_protocol \
  --dataset-dirs \
    datasets/IRSTD-1K \
    datasets/NUAA-SIRST \
    datasets/NUDT-SIRST \
  --outer-target NUAA-SIRST \
  --pseudo-target NUDT-SIRST \
  --near-duplicate-audit audits/aaai27/near_duplicates_effective_splits_v2.json \
  --output "$output_dir/data_audit_three_domains.json" \
  2>&1 | tee "$output_dir/data_audit.log"

log_step "preflight complete"
