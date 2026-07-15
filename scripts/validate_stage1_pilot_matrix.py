"""Validate and materialise the frozen AAAI-27 Stage-1 pilot matrix.

The matrix is a contract, not a launcher.  Validation is fail-closed and the
reported argv resolves every split path to an absolute path before it can be
passed to ``train_multisource_tail`` (whose explicit split paths are otherwise
resolved relative to each dataset root).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "rc-irstd.aaai27-stage1-pilot-matrix.v1"
PLAN_SCHEMA_VERSION = "rc-irstd.aaai27-analysis-plan.v1"
DEVELOPMENT_DOMAINS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
SYNTHETIC_ALL_THREE_TARGET = "DEVELOPMENT-ONLY-ALL-THREE-DIAGNOSTIC"
DATASET_DIRS = {
    "NUAA-SIRST": "datasets/NUAA-SIRST",
    "NUDT-SIRST": "datasets/NUDT-SIRST",
    "IRSTD-1K": "datasets/IRSTD-1K",
}
SPLIT_SLUGS = {
    "NUAA-SIRST": "nuaa-sirst",
    "NUDT-SIRST": "nudt-sirst",
    "IRSTD-1K": "irstd-1k",
}
FIT_SPLITS = {
    domain: f"splits/aaai27_v2/{slug}/detector_fit.txt"
    for domain, slug in SPLIT_SLUGS.items()
}
DIAGNOSTIC_SPLITS = {
    domain: f"splits/aaai27_v2/{slug}/detector_diagnostic.txt"
    for domain, slug in SPLIT_SLUGS.items()
}
MATRIX_TOP_LEVEL_KEYS = {
    "schema_version",
    "contains_observed_results",
    "analysis_plan_binding",
    "release_contract",
    "protocol",
    "scheduling",
    "runs",
}
RUN_KEYS = {
    "run_id",
    "phase",
    "experiment_scope",
    "variant",
    "seed",
    "epochs",
    "fixed_last",
    "sources",
    "source_dirs",
    "source_split_files",
    "outer_fold_id",
    "outer_target",
    "held_out",
    "primary_diagnostic_domains",
    "evaluation_diagnostic_domains",
    "evaluation_diagnostic_files",
    "gpu_visible_devices",
    "data_parallel",
    "output_dir",
}
FORBIDDEN_MATRIX_TOKENS = (
    "official_test",
    "official-test",
    "official test",
    "img_idx/test",
    "/test_nuaa",
    "/test_nudt",
    "/test_irstd",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_strict_json(path: str | Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location} keys drift: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _resolve_repo_file(root: Path, relative: str, location: str) -> Path:
    path = _resolve_repo_path(root, relative, location)
    if not path.is_file():
        raise FileNotFoundError(f"{location} is not a file: {path}")
    return path


def _resolve_repo_dir(root: Path, relative: str, location: str) -> Path:
    path = _resolve_repo_path(root, relative, location)
    if not path.is_dir():
        raise FileNotFoundError(f"{location} is not a directory: {path}")
    return path


def _resolve_repo_path(root: Path, relative: str, location: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"{location} must be repository-relative: {relative}")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{location} escapes repository root: {relative}") from error
    return path


def _expected_protocol() -> dict[str, Any]:
    return {
        "seed": 42,
        "epochs": 30,
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "diagnostics_select_checkpoint": False,
        "deterministic": True,
        "optimizer": "Adagrad",
        "learning_rate": 0.05,
        "warm_epoch": 5,
        "risk_warmup_epochs": 5,
        "risk_ramp_epochs": 10,
        "base_size": 256,
        "crop_size": 256,
        "batch_per_domain": 3,
        "num_workers": 4,
        "epoch_steps_mode": "full_longest_domain",
        "tail_mode": "local-peak",
        "lambda_tail": 0.1,
        "lambda_miss": 0.1,
        "target_background_margin": 1.0,
        "tail_q": 0.05,
        "miss_q": 0.25,
        "object_pixel_q": 0.25,
        "tail_gamma": 10.0,
        "peak_kernel_size": 5,
        "peak_min_score": 0.05,
        "plateau_atol": 0.0,
        "grad_clip_norm": 0.0,
        "variants": {
            "D0": {
                "risk_objective": "segmentation-only",
                "lambda_margin": 0.0,
                "exclusion_radius": 2,
            },
            "D3": {
                "risk_objective": "margin",
                "lambda_margin": 0.2,
                "exclusion_radius": 2,
            },
        },
    }


def _expected_scheduling() -> dict[str, Any]:
    return {
        "physical_gpu_ids": [0, 1, 2],
        "all_three_and_lodo_overlap_allowed": False,
        "phases": [
            {
                "phase_id": "P0_all-three_D0",
                "after": [],
                "concurrent_run_ids": ["D0_all-three_s42"],
            },
            {
                "phase_id": "P1_all-three_D3",
                "after": ["P0_all-three_D0"],
                "concurrent_run_ids": ["D3_all-three_s42"],
            },
            {
                "phase_id": "P2_lodo_D0",
                "after": ["P1_all-three_D3"],
                "concurrent_run_ids": [
                    "D0_leave-NUAA_s42",
                    "D0_leave-NUDT_s42",
                    "D0_leave-IRSTD1K_s42",
                ],
            },
            {
                "phase_id": "P3_lodo_D3",
                "after": ["P2_lodo_D0"],
                "concurrent_run_ids": [
                    "D3_leave-NUAA_s42",
                    "D3_leave-NUDT_s42",
                    "D3_leave-IRSTD1K_s42",
                ],
            },
        ],
    }


def _expected_run_specs() -> list[dict[str, Any]]:
    all_domains = list(DEVELOPMENT_DOMAINS)
    diagnostics = [DIAGNOSTIC_SPLITS[domain] for domain in DEVELOPMENT_DOMAINS]
    folds = [
        (
            "leave-NUAA",
            "NUAA-SIRST",
            ["NUDT-SIRST", "IRSTD-1K"],
            0,
            "NUAA",
        ),
        (
            "leave-NUDT",
            "NUDT-SIRST",
            ["NUAA-SIRST", "IRSTD-1K"],
            1,
            "NUDT",
        ),
        (
            "leave-IRSTD1K",
            "IRSTD-1K",
            ["NUAA-SIRST", "NUDT-SIRST"],
            2,
            "IRSTD1K",
        ),
    ]

    result: list[dict[str, Any]] = []
    for variant, phase in (
        ("D0", "P0_all-three_D0"),
        ("D3", "P1_all-three_D3"),
    ):
        run_id = f"{variant}_all-three_s42"
        result.append(
            {
                "run_id": run_id,
                "phase": phase,
                "experiment_scope": "single_seed_stage1_gate",
                "variant": variant,
                "seed": 42,
                "epochs": 30,
                "fixed_last": True,
                "sources": all_domains,
                "source_dirs": [DATASET_DIRS[domain] for domain in all_domains],
                "source_split_files": [FIT_SPLITS[domain] for domain in all_domains],
                "outer_fold_id": "development-all-three",
                "outer_target": SYNTHETIC_ALL_THREE_TARGET,
                "held_out": [SYNTHETIC_ALL_THREE_TARGET],
                "primary_diagnostic_domains": all_domains,
                "evaluation_diagnostic_domains": all_domains,
                "evaluation_diagnostic_files": diagnostics,
                "gpu_visible_devices": [0, 1, 2],
                "data_parallel": True,
                "output_dir": f"outputs/stage1_pilot_30ep_v5_rc4/{run_id}",
            }
        )

    for variant, phase in (("D0", "P2_lodo_D0"), ("D3", "P3_lodo_D3")):
        for fold_id, target, sources, gpu, short_target in folds:
            run_id = f"{variant}_leave-{short_target}_s42"
            result.append(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "experiment_scope": "single_seed_stage1_gate",
                    "variant": variant,
                    "seed": 42,
                    "epochs": 30,
                    "fixed_last": True,
                    "sources": sources,
                    "source_dirs": [DATASET_DIRS[domain] for domain in sources],
                    "source_split_files": [FIT_SPLITS[domain] for domain in sources],
                    "outer_fold_id": fold_id,
                    "outer_target": target,
                    "held_out": [target],
                    "primary_diagnostic_domains": [target],
                    "evaluation_diagnostic_domains": all_domains,
                    "evaluation_diagnostic_files": diagnostics,
                    "gpu_visible_devices": [gpu],
                    "data_parallel": False,
                    "output_dir": f"outputs/stage1_pilot_30ep_v5_rc4/{run_id}",
                }
            )
    return result


def _validate_plan_and_frozen_inputs(
    matrix: Mapping[str, Any],
    *,
    matrix_path: Path,
    plan_path: Path,
    repository_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    binding = _mapping(matrix["analysis_plan_binding"], "analysis_plan_binding")
    _exact_keys(
        binding,
        {"path", "schema_version", "allowed_plan_statuses", "stage1_config", "split_manifest"},
        "analysis_plan_binding",
    )
    if binding["path"] != "configs/aaai27_analysis_plan.json":
        raise ValueError("analysis-plan canonical path drift")
    if binding["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("analysis-plan schema binding drift")
    allowed_statuses = [
        "frozen_stage1_protocol_pending_release",
        "frozen_stage1_pilot_authorized",
    ]
    if binding["allowed_plan_statuses"] != allowed_statuses:
        raise ValueError("analysis-plan allowed status binding drift")

    plan = _mapping(load_strict_json(plan_path), "analysis_plan")
    if plan.get("schema_version") != binding["schema_version"]:
        raise ValueError("bound analysis-plan schema mismatch")
    if plan.get("plan_status") not in binding["allowed_plan_statuses"]:
        raise ValueError("bound analysis-plan status does not authorize this matrix state")
    if plan.get("contains_observed_results") is not False:
        raise ValueError("analysis plan contains observed results")

    contracts = _mapping(plan.get("hash_contracts"), "analysis_plan.hash_contracts")
    matrix_contract = _mapping(
        contracts.get("stage1_pilot_matrix"),
        "analysis_plan.hash_contracts.stage1_pilot_matrix",
    )
    expected_matrix_path = matrix_path.resolve().relative_to(repository_root.resolve()).as_posix()
    if matrix_contract.get("path") != expected_matrix_path:
        raise ValueError("analysis plan points to a different Stage-1 pilot matrix")
    if matrix_contract.get("sha256") != sha256_file(matrix_path):
        raise ValueError("analysis-plan Stage-1 pilot matrix SHA-256 drift")

    bound_inputs: dict[str, Mapping[str, Any]] = {}
    for binding_name, plan_name in (
        ("stage1_config", "stage1_config"),
        ("split_manifest", "official_train_split_manifest"),
    ):
        matrix_input = _mapping(binding[binding_name], f"analysis_plan_binding.{binding_name}")
        _exact_keys(matrix_input, {"path", "sha256"}, f"analysis_plan_binding.{binding_name}")
        plan_input = _mapping(contracts.get(plan_name), f"analysis_plan.hash_contracts.{plan_name}")
        if dict(matrix_input) != dict(plan_input):
            raise ValueError(f"matrix/analysis-plan {binding_name} binding mismatch")
        input_path = _resolve_repo_file(
            repository_root,
            str(matrix_input["path"]),
            f"analysis_plan_binding.{binding_name}.path",
        )
        if sha256_file(input_path) != matrix_input["sha256"]:
            raise ValueError(f"{binding_name} SHA-256 drift")
        bound_inputs[binding_name] = _mapping(load_strict_json(input_path), binding_name)

    authorization = _mapping(plan.get("authorization"), "analysis_plan.authorization")
    data_roles = _mapping(plan.get("data_roles"), "analysis_plan.data_roles")
    gate = _mapping(plan["stage1_contract"]["single_seed_gate"], "single_seed_gate")
    deprecated_role_fields = {
        "outer_target_official_train_used",
        "outer_target_official_train_allowed_in_same_outer_fold",
    }
    present_deprecated_fields = deprecated_role_fields.intersection(data_roles)
    if present_deprecated_fields:
        raise ValueError(
            "analysis plan contains deprecated ambiguous role fields: "
            f"{sorted(present_deprecated_fields)}"
        )
    if authorization.get("official_test_model_evaluation") is not False:
        raise ValueError("sealed evaluation role is unexpectedly authorized")
    if data_roles.get("official_test_allowed_for_selection") is not False:
        raise ValueError("sealed evaluation role is unexpectedly allowed for selection")
    if data_roles.get("outer_target_official_train_used_for_detector_fit") is not False:
        raise ValueError("outer-target official train is unexpectedly used for fitting")
    if (
        data_roles.get(
            "outer_target_detector_diagnostic_used_for_development_evaluation"
        )
        is not True
    ):
        raise ValueError("outer-target development diagnostic role is not enabled")
    if data_roles.get("outer_target_diagnostic_selects_checkpoint") is not False:
        raise ValueError("outer-target diagnostics unexpectedly select checkpoints")
    if gate.get("official_test_absent_from_gate") is not True:
        raise ValueError("Stage-1 single-seed gate does not keep the sealed role absent")
    return plan, bound_inputs["stage1_config"], bound_inputs["split_manifest"]


def _validate_protocol_bindings(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage1_config: Mapping[str, Any],
) -> None:
    expected = _expected_protocol()
    if dict(protocol) != expected:
        raise ValueError("Stage-1 pilot protocol differs from the frozen exact contract")

    stage1 = _mapping(plan["stage1_contract"], "analysis_plan.stage1_contract")
    common = _mapping(stage1["common_training"], "stage1_contract.common_training")
    gpu = _mapping(stage1["gpu_protocol"], "stage1_contract.gpu_protocol")
    plan_variants = _mapping(stage1["variants"], "stage1_contract.variants")
    if protocol["seed"] != stage1["single_seed_pilot_seed"]:
        raise ValueError("pilot seed differs from analysis plan")
    if protocol["epochs"] != stage1["single_seed_pilot_epochs"]:
        raise ValueError("pilot epochs differ from analysis plan")
    common_pairs = {
        "optimizer": "optimizer",
        "learning_rate": "learning_rate",
        "warm_epoch": "warm_epoch",
        "risk_warmup_epochs": "risk_warmup_epochs",
        "risk_ramp_epochs": "risk_ramp_epochs",
        "base_size": "base_size",
        "crop_size": "crop_size",
        "checkpoint_selection": "checkpoint_selection",
        "deterministic": "deterministic",
    }
    for protocol_key, plan_key in common_pairs.items():
        if protocol[protocol_key] != common[plan_key]:
            raise ValueError(f"protocol.{protocol_key} differs from analysis plan")
    if protocol["batch_per_domain"] != gpu["batch_per_domain"]:
        raise ValueError("batch-per-domain differs from analysis plan")
    if list(gpu["physical_devices"]) != [0, 1, 2]:
        raise ValueError("analysis-plan physical GPU contract drift")

    config_registry = _mapping(stage1_config["stage1_variant_registry"], "stage1_config.variants")
    for variant in ("D0", "D3"):
        matrix_variant = _mapping(protocol["variants"][variant], f"protocol.variants.{variant}")
        planned_variant = _mapping(plan_variants[variant], f"stage1_contract.variants.{variant}")
        configured_variant = _mapping(config_registry[variant], f"stage1_config.variants.{variant}")
        for key in ("risk_objective", "lambda_margin"):
            if matrix_variant[key] != planned_variant[key] or matrix_variant[key] != configured_variant[key]:
                raise ValueError(f"{variant} {key} binding drift")
        if matrix_variant["exclusion_radius"] != planned_variant["exclusion_radius"]:
            raise ValueError(f"{variant} exclusion radius binding drift")
        if matrix_variant["exclusion_radius"] != stage1_config["gt_exclusion_radius"]:
            raise ValueError(f"{variant} exclusion radius/config drift")

    config_pairs = {
        "target_background_margin": "target_background_margin_logit",
        "tail_q": "background_tail_fraction",
        "miss_q": "hard_object_fraction",
        "object_pixel_q": "object_top_pixel_fraction",
        "tail_gamma": "smooth_worst_domain_gamma",
        "peak_kernel_size": "peak_kernel_size",
        "plateau_atol": "plateau_atol",
    }
    for protocol_key, config_key in config_pairs.items():
        if protocol[protocol_key] != stage1_config[config_key]:
            raise ValueError(f"protocol.{protocol_key} differs from Stage-1 config")
    schedule = _mapping(stage1_config["training_schedule"], "stage1_config.training_schedule")
    for protocol_key, config_key in (
        ("optimizer", "optimizer"),
        ("learning_rate", "lr"),
        ("warm_epoch", "warm_epoch"),
        ("risk_warmup_epochs", "risk_warmup_epochs"),
        ("risk_ramp_epochs", "risk_ramp_epochs"),
    ):
        if protocol[protocol_key] != schedule[config_key]:
            raise ValueError(f"protocol.{protocol_key} differs from Stage-1 schedule")


def _validate_manifest_splits(
    manifest: Mapping[str, Any], repository_root: Path
) -> None:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise TypeError("split_manifest.datasets must be an array")
    by_domain = {
        str(item["dataset_name"]): _mapping(item, "split_manifest.dataset")
        for item in datasets
    }
    if set(by_domain) != set(DEVELOPMENT_DOMAINS):
        raise ValueError("split-manifest development domains drift")
    for domain in DEVELOPMENT_DOMAINS:
        detector = _mapping(by_domain[domain]["detector"], f"{domain}.detector")
        expected_fit = FIT_SPLITS[domain]
        expected_diagnostic = DIAGNOSTIC_SPLITS[domain]
        manifest_fit = f"splits/aaai27_v2/{detector['fit_file']}"
        manifest_diagnostic = f"splits/aaai27_v2/{detector['diagnostic_file']}"
        if manifest_fit != expected_fit or manifest_diagnostic != expected_diagnostic:
            raise ValueError(f"{domain} detector split path drift")
        fit_path = _resolve_repo_file(repository_root, expected_fit, f"{domain}.fit")
        diagnostic_path = _resolve_repo_file(
            repository_root, expected_diagnostic, f"{domain}.diagnostic"
        )
        if sha256_file(fit_path) != detector["fit_sha256"]:
            raise ValueError(f"{domain} detector-fit SHA-256 drift")
        if sha256_file(diagnostic_path) != detector["diagnostic_sha256"]:
            raise ValueError(f"{domain} detector-diagnostic SHA-256 drift")


def normalized_run_invocation(
    run: Mapping[str, Any],
    protocol: Mapping[str, Any],
    release_contract: Mapping[str, Any],
    *,
    matrix_path: Path,
    plan_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Return a direct-module invocation; this function never executes it."""

    source_dirs = [
        str(_resolve_repo_dir(repository_root, value, f"{run['run_id']}.source_dir"))
        for value in run["source_dirs"]
    ]
    source_splits = [
        str(_resolve_repo_file(repository_root, value, f"{run['run_id']}.source_split"))
        for value in run["source_split_files"]
    ]
    output_dir = _resolve_repo_path(
        repository_root, str(run["output_dir"]), f"{run['run_id']}.output_dir"
    )
    variant = _mapping(protocol["variants"][run["variant"]], "protocol.variant")
    argv = [
        "-m",
        "scripts.train_multisource_tail",
        "--aaai27-pilot",
        "--analysis-plan",
        str(plan_path.resolve()),
        "--pilot-matrix",
        str(matrix_path.resolve()),
        "--pilot-run-id",
        str(run["run_id"]),
        "--release-tag",
        str(release_contract["tag"]),
        "--source-archive",
        str(_resolve_repo_path(repository_root, release_contract["source_archive"], "release.archive")),
        "--source-archive-sha256-file",
        str(
            _resolve_repo_path(
                repository_root,
                release_contract["source_archive_sha256_file"],
                "release.checksum",
            )
        ),
        "--source-dirs",
        *source_dirs,
        "--source-split-files",
        *source_splits,
        "--source-names",
        *[str(value) for value in run["sources"]],
        "--outer-fold-id",
        str(run["outer_fold_id"]),
        "--outer-target",
        str(run["outer_target"]),
        "--held-out-domains",
        *[str(value) for value in run["held_out"]],
        "--batch-per-domain",
        str(protocol["batch_per_domain"]),
        "--epochs",
        str(protocol["epochs"]),
        "--lr",
        str(protocol["learning_rate"]),
        "--warm-epoch",
        str(protocol["warm_epoch"]),
        "--risk-warmup-epochs",
        str(protocol["risk_warmup_epochs"]),
        "--risk-ramp-epochs",
        str(protocol["risk_ramp_epochs"]),
        "--base-size",
        str(protocol["base_size"]),
        "--crop-size",
        str(protocol["crop_size"]),
        "--num-workers",
        str(protocol["num_workers"]),
        "--risk-objective",
        str(variant["risk_objective"]),
        "--tail-mode",
        str(protocol["tail_mode"]),
        "--lambda-tail",
        str(protocol["lambda_tail"]),
        "--lambda-miss",
        str(protocol["lambda_miss"]),
        "--lambda-margin",
        str(variant["lambda_margin"]),
        "--target-background-margin",
        str(protocol["target_background_margin"]),
        "--tail-q",
        str(protocol["tail_q"]),
        "--miss-q",
        str(protocol["miss_q"]),
        "--object-pixel-q",
        str(protocol["object_pixel_q"]),
        "--tail-gamma",
        str(protocol["tail_gamma"]),
        "--peak-kernel-size",
        str(protocol["peak_kernel_size"]),
        "--exclusion-radius",
        str(variant["exclusion_radius"]),
        "--peak-min-score",
        str(protocol["peak_min_score"]),
        "--plateau-atol",
        str(protocol["plateau_atol"]),
        "--grad-clip-norm",
        str(protocol["grad_clip_norm"]),
        "--seed",
        str(protocol["seed"]),
        "--device",
        "cuda",
        "--deterministic",
        "--save-dir",
        str(output_dir.parent),
        "--run-name",
        output_dir.name,
    ]
    if run["data_parallel"]:
        argv.append("--data-parallel")
    return {
        "run_id": run["run_id"],
        "phase": run["phase"],
        "environment": {
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(value) for value in run["gpu_visible_devices"]
            )
        },
        "python_argv": argv,
        "output_dir": str(output_dir),
        "evaluation_diagnostic_files": [
            str(_resolve_repo_file(repository_root, value, f"{run['run_id']}.diagnostic"))
            for value in run["evaluation_diagnostic_files"]
        ],
    }


def validate_matrix(
    matrix_path: Path,
    repository_root: Path,
    *,
    plan_path: Path | None = None,
    require_release_artifacts: bool = False,
) -> dict[str, Any]:
    root = repository_root.resolve()
    matrix_path = matrix_path.resolve()
    matrix = _mapping(load_strict_json(matrix_path), "pilot_matrix")
    _exact_keys(matrix, MATRIX_TOP_LEVEL_KEYS, "pilot_matrix")
    if matrix["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported Stage-1 pilot matrix schema")
    if matrix["contains_observed_results"] is not False:
        raise ValueError("pilot matrix must not contain observed results")
    rendered_lower = matrix_path.read_text(encoding="utf-8").lower()
    for token in FORBIDDEN_MATRIX_TOKENS:
        if token in rendered_lower:
            raise ValueError(f"sealed evaluation token is forbidden in matrix: {token}")

    binding = _mapping(matrix["analysis_plan_binding"], "analysis_plan_binding")
    if plan_path is None:
        plan_path = _resolve_repo_file(root, str(binding["path"]), "analysis_plan_binding.path")
    plan, stage1_config, split_manifest = _validate_plan_and_frozen_inputs(
        matrix,
        matrix_path=matrix_path,
        plan_path=plan_path.resolve(),
        repository_root=root,
    )
    protocol = _mapping(matrix["protocol"], "protocol")
    _validate_protocol_bindings(protocol, plan, stage1_config)
    _validate_manifest_splits(split_manifest, root)

    release = _mapping(matrix["release_contract"], "release_contract")
    _exact_keys(
        release,
        {"tag", "source_archive", "source_archive_sha256_file"},
        "release_contract",
    )
    if release["tag"] != "aaai27-rc-irstd-v5-rc4":
        raise ValueError("release tag drift")
    if release["source_archive"] != "outputs/release/RC-IRSTD_v5_rc4.zip":
        raise ValueError("source archive path drift")
    if release["source_archive_sha256_file"] != "outputs/release/RC-IRSTD_v5_rc4.zip.sha256":
        raise ValueError("source archive checksum path drift")
    archive_path = _resolve_repo_path(root, release["source_archive"], "release.archive")
    checksum_path = _resolve_repo_path(
        root, release["source_archive_sha256_file"], "release.checksum"
    )
    if require_release_artifacts and (not archive_path.is_file() or not checksum_path.is_file()):
        raise FileNotFoundError("frozen release archive/checksum is absent")

    scheduling = _mapping(matrix["scheduling"], "scheduling")
    if dict(scheduling) != _expected_scheduling():
        raise ValueError("pilot GPU scheduling differs from the frozen four-phase contract")

    runs_raw = matrix["runs"]
    if not isinstance(runs_raw, list):
        raise TypeError("runs must be an array")
    expected_runs = _expected_run_specs()
    if len(runs_raw) != 8:
        raise ValueError("Stage-1 pilot matrix must contain exactly eight runs")
    runs: list[Mapping[str, Any]] = []
    for index, (raw_run, expected_run) in enumerate(zip(runs_raw, expected_runs)):
        run = _mapping(raw_run, f"runs[{index}]")
        _exact_keys(run, RUN_KEYS, f"runs[{index}]")
        if dict(run) != expected_run:
            raise ValueError(f"run contract drift: expected {expected_run['run_id']}")
        if not set(run["primary_diagnostic_domains"]).issubset(
            run["evaluation_diagnostic_domains"]
        ):
            raise ValueError(f"{run['run_id']} primary diagnostics are outside evaluation roles")
        for path in run["source_dirs"]:
            _resolve_repo_dir(root, path, f"{run['run_id']}.source_dirs")
        for path in run["source_split_files"] + run["evaluation_diagnostic_files"]:
            _resolve_repo_file(root, path, f"{run['run_id']}.split")
        runs.append(run)

    required_runs = plan["stage1_contract"]["single_seed_gate"]["required_runs"]
    run_ids = [str(run["run_id"]) for run in runs]
    if set(required_runs) != set(run_ids) or len(required_runs) != len(run_ids):
        raise ValueError("analysis-plan single-seed gate run IDs differ from matrix")
    if len({str(run["output_dir"]) for run in runs}) != len(runs):
        raise ValueError("pilot output directories must be unique")

    invocations = [
        normalized_run_invocation(
            run,
            protocol,
            release,
            matrix_path=matrix_path,
            plan_path=plan_path,
            repository_root=root,
        )
        for run in runs
    ]
    return {
        "artifact_type": "rc_irstd_aaai27_stage1_pilot_matrix_audit",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "contains_observed_results": False,
        "matrix_sha256": sha256_file(matrix_path),
        "run_count": len(runs),
        "run_ids": run_ids,
        "phase_count": len(scheduling["phases"]),
        "release_artifacts_required": require_release_artifacts,
        "normalized_invocations": invocations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/aaai27_stage1_pilot_matrix.json")
    parser.add_argument("--analysis-plan", default="configs/aaai27_analysis_plan.json")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--require-release-artifacts", action="store_true")
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.repository_root).expanduser().resolve()
    matrix_path = Path(args.matrix).expanduser()
    if not matrix_path.is_absolute():
        matrix_path = root / matrix_path
    plan_path = Path(args.analysis_plan).expanduser()
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    report = validate_matrix(
        matrix_path,
        root,
        plan_path=plan_path,
        require_release_artifacts=args.require_release_artifacts,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
