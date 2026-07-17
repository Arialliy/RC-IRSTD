from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from data_ext.stage2_role_contract import (
    _verify_seed_manifest,
    load_stage2_selection,
    verify_stage2_run_contract,
    verify_stage2_run_contract_sidecar,
)
from scripts.materialize_stage2_detector_run_contracts import (
    materialize_stage2_detector_run_contracts,
)


ROOT = Path(__file__).resolve().parents[1]
MATERIALIZATION = (
    ROOT
    / "outputs/stage2_manifests/rc4_k2_c14q28_20260716/materialization_index.json"
)
MATERIALIZATION_SHA = "b52a7938a13df78b8157a39fed02695ff9268bfffbbf7c665c10c1f66fe52d94"
SEEDS = ROOT / "outputs/stage2_protocol/RC4_STAGE2_SEED_DERIVATION_MANIFEST_V1_20260716.json"
SEEDS_SHA = "4f426ea44e09b4f086092a8a41d5d0cff156b20b2bb1433a2ba3bed5c987604b"
CONFIG = ROOT / "configs/aaai27_detector_tail_sep.json"
CONFIG_SHA = "186d4ff30b16aecfff35eb682c8ecb897fcee98308a1bfa3f946748ef77b01af"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _sha(path)


def _write_adjacent_sha256(path: Path) -> dict[str, str]:
    digest = _sha(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return {"path": path.name, "sha256": digest, "sidecar": sidecar.name}


def _stage2_runtime_fixture(
    contract_tree,
    directory_name: str,
    *,
    role: str = "detector_oof",
    fold: int | None = 0,
) -> dict:
    from scripts import train_multisource_tail as trainer

    output, index = contract_tree
    entry = _contract(index, role=role, fold=fold)
    contract, contract_sha = verify_stage2_run_contract_sidecar(ROOT / entry["path"])
    run_dir = output / directory_name
    run_dir.mkdir(exist_ok=False)
    fingerprint = {"python": "synthetic", "source_tree": "frozen"}
    config = {
        "schema_version": "synthetic-stage2-config-v1",
        "segmentation_loss_implementation": (
            trainer.stage1_segmentation_loss_implementation()
        ),
    }
    config_path = run_dir / "config.json"
    config_sha = _write_json(config_path, config)
    _write_adjacent_sha256(config_path)
    args = SimpleNamespace(stage2_run_contract=str(ROOT / entry["path"]))
    runtime = trainer.write_stage2_runtime_artifacts(
        run_dir,
        args,
        contract,
        contract_sha,
        config_sha,
        fingerprint,
    )
    runtime_path = run_dir / "stage2_runtime_contract.json"
    return {
        "output": output,
        "entry": entry,
        "contract": contract,
        "contract_sha": contract_sha,
        "run_dir": run_dir,
        "fingerprint": fingerprint,
        "config": config,
        "config_sha": config_sha,
        "runtime": runtime,
        "runtime_path": runtime_path,
        "runtime_payload": json.loads(runtime_path.read_text(encoding="utf-8")),
        "args": args,
    }


@pytest.fixture(scope="module")
def contract_tree():
    temporary = tempfile.mkdtemp(prefix="stage2-w01-contract-test-", dir=ROOT / "outputs")
    output = Path(temporary) / "contracts"
    index = materialize_stage2_detector_run_contracts(
        materialization_index=MATERIALIZATION,
        materialization_index_sha256=MATERIALIZATION_SHA,
        seed_manifest=SEEDS,
        seed_manifest_sha256=SEEDS_SHA,
        detector_config=CONFIG,
        detector_config_sha256=CONFIG_SHA,
        output_root=output,
    )
    try:
        yield output, index
    finally:
        shutil.rmtree(temporary)


def _contract(index: dict, *, role: str, fold: int | None = None) -> dict:
    matches = [
        item
        for item in index["contracts"]
        if item["detector_role"] == role
        and (fold is None or item["oof_fold_index"] == fold)
    ]
    assert matches
    return matches[0]


def test_materializes_exact_18_oof_9_fullfit_and_63_seed_mapping(contract_tree):
    _, index = contract_tree
    assert index["run_count"] == 27
    assert index["oof_run_count"] == 18
    assert index["full_fit_run_count"] == 9
    assert index["selection_count"] == 54
    assert index["official_test_accessed"] is False
    assert len({item["run_id"] for item in index["contracts"]}) == 27
    seed_payload = json.loads(SEEDS.read_text(encoding="utf-8"))
    _verify_seed_manifest(seed_payload)
    all_seeds = [
        seed
        for row in seed_payload["derived_seed_table"]
        for seed in row["derived_seeds_by_role"].values()
    ]
    assert len(all_seeds) == len(set(all_seeds)) == 63


def test_oof_and_fullfit_selections_are_exact_two_source_assignment_filters(contract_tree):
    _, index = contract_tree
    for role, fold in (("detector_oof", 0), ("detector_oof", 1), ("detector_full_fit", None)):
        run = _contract(index, role=role, fold=fold)
        payload = verify_stage2_run_contract(
            ROOT / run["path"],
            run["sha256"],
            SEEDS,
            MATERIALIZATION,
        )
        assert len(payload["source_domains"]) == 2
        assert payload["outer_target_domain"] not in payload["source_domains"]
        expected_role = (
            "detector_oof_train" if role == "detector_oof" else "detector_full_fit_train"
        )
        for binding in payload["selection_contracts"]:
            selection = load_stage2_selection(
                ROOT / binding["path"], binding["sha256"], expected_role
            )
            assignment_path = ROOT / selection["bindings"]["assignment"]["path"]
            assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
            selected_indexes = {
                item["source_role_record_index"] for item in selection["records"]
            }
            if role == "detector_oof":
                expected_indexes = {
                    item["source_role_record_index"]
                    for item in assignment["records"]
                    if item["oof_fold_index"] != fold
                }
                heldout_groups = {
                    item["exclusion_group_id"]
                    for item in assignment["records"]
                    if item["oof_fold_index"] == fold
                }
                assert heldout_groups.isdisjoint(
                    item["exclusion_group_id"] for item in selection["records"]
                )
            else:
                expected_indexes = {
                    item["source_role_record_index"] for item in assignment["records"]
                }
            assert selected_indexes == expected_indexes


def test_strict_boolean_hash_path_symlink_and_incomplete_mutations_fail(contract_tree):
    output, index = contract_tree
    run = _contract(index, role="detector_oof", fold=0)
    run_payload = json.loads((ROOT / run["path"]).read_text(encoding="utf-8"))
    selection_binding = run_payload["selection_contracts"][0]
    selection_path = ROOT / selection_binding["path"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    mutations = output / "mutations"

    boolean_mutation = copy.deepcopy(selection)
    boolean_mutation["official_test_accessed"] = 0
    path = mutations / "boolean.selection.json"
    digest = _write_json(path, boolean_mutation)
    with pytest.raises(TypeError, match="exact JSON boolean"):
        load_stage2_selection(path, digest, selection["selection_role"])

    record_mutation = copy.deepcopy(selection)
    record_mutation["records"][0]["image_id"] += "_mutated"
    path = mutations / "record.selection.json"
    digest = _write_json(path, record_mutation)
    with pytest.raises(ValueError, match="canonical_id/domain/image_id"):
        load_stage2_selection(path, digest, selection["selection_role"])

    path_mutation = copy.deepcopy(selection)
    path_mutation["id_list"]["path"] = "/tmp/absolute.ids.txt"
    path = mutations / "absolute.selection.json"
    digest = _write_json(path, path_mutation)
    with pytest.raises(ValueError, match="repository-relative"):
        load_stage2_selection(path, digest, selection["selection_role"])

    traversal_mutation = copy.deepcopy(selection)
    traversal_mutation["id_list"]["path"] = "outputs/../escape.ids.txt"
    path = mutations / "traversal.selection.json"
    digest = _write_json(path, traversal_mutation)
    with pytest.raises(ValueError, match="canonical POSIX|repository-relative"):
        load_stage2_selection(path, digest, selection["selection_role"])

    symlink = mutations / "selection-symlink.json"
    symlink.symlink_to(selection_path)
    with pytest.raises(ValueError, match="symlink"):
        load_stage2_selection(symlink, selection_binding["sha256"], selection["selection_role"])

    marker = output / ".stage2_contract_materialization_incomplete"
    marker.write_text("interrupted\n", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="incomplete"):
            verify_stage2_run_contract_sidecar(ROOT / run["path"])
    finally:
        marker.unlink()


def test_run_rejects_seed_override_and_outer_target_source(contract_tree):
    output, index = contract_tree
    run = _contract(index, role="detector_full_fit")
    original = json.loads((ROOT / run["path"]).read_text(encoding="utf-8"))
    mutations = output / "mutations"

    seed_mutation = copy.deepcopy(original)
    seed_mutation["derived_seed"] += 1
    path = mutations / "seed.run.json"
    digest = _write_json(path, seed_mutation)
    with pytest.raises(ValueError, match="seed table"):
        verify_stage2_run_contract(path, digest, SEEDS, MATERIALIZATION)

    outer_mutation = copy.deepcopy(original)
    outer_mutation["source_domains"][0] = outer_mutation["outer_target_domain"]
    path = mutations / "outer.run.json"
    digest = _write_json(path, outer_mutation)
    with pytest.raises(ValueError, match="outer target"):
        verify_stage2_run_contract(path, digest, SEEDS, MATERIALIZATION)


def test_stage2_trainer_path_never_calls_legacy_test_split_or_dataset_identity(contract_tree):
    from scripts import train_multisource_tail as trainer

    _, index = contract_tree
    run = _contract(index, role="detector_oof", fold=0)
    contract, _ = verify_stage2_run_contract_sidecar(ROOT / run["path"])

    class FakeDataset:
        def __init__(self, dataset_args, mode):
            assert mode == "train"
            self.names = [
                line.strip()
                for line in Path(dataset_args.split_file)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.list_dir = dataset_args.split_file
            self.imgs_dir = str(Path(dataset_args.dataset_dir) / "images")
            self.label_dir = str(Path(dataset_args.dataset_dir) / "masks")

        def __len__(self):
            return len(self.names)

    with patch.object(
        sys,
        "argv",
        [
            "train_multisource_tail",
            "--stage2-run-contract",
            str(ROOT / run["path"]),
            "--seed",
            str(run["derived_seed"]),
        ],
    ):
        args = trainer.parse_args()
    trainer.bind_stage2_run_contract_to_args(args, contract)
    trainer._validate_args(args)
    with patch.object(
        trainer,
        "audited_source_train_split",
        side_effect=AssertionError("official-test split path was touched"),
    ), patch.object(
        trainer,
        "build_detector_source_records",
        side_effect=AssertionError("whole-dataset identity path was touched"),
    ), patch.object(trainer, "IRSTD_Dataset", FakeDataset):
        datasets = trainer.build_source_datasets_from_stage2_contract(args, contract)
        records = trainer.build_stage2_detector_source_records(
            contract["source_domains"], datasets, contract
        )
    assert list(datasets) == contract["source_domains"]
    assert len(records) == 2
    assert all(record["official_test_accessed"] is False for record in records)
    assert trainer.protocol_scope(args, contract["source_domains"]) == (
        "stage2_development_detector_official_test_sealed"
    )


def test_runtime_seed_and_d3_contract_mismatches_fail(contract_tree):
    from scripts import train_multisource_tail as trainer

    _, index = contract_tree
    run = _contract(index, role="detector_oof", fold=1)
    contract, _ = verify_stage2_run_contract_sidecar(ROOT / run["path"])
    with patch.object(
        sys,
        "argv",
        ["train_multisource_tail", "--stage2-run-contract", str(ROOT / run["path"])],
    ):
        args = trainer.parse_args()
    with pytest.raises(ValueError, match="runtime --seed"):
        trainer.bind_stage2_run_contract_to_args(args, contract)

    with patch.object(
        sys,
        "argv",
        [
            "train_multisource_tail",
            "--stage2-run-contract",
            str(ROOT / run["path"]),
            "--seed",
            str(run["derived_seed"]),
            "--lambda-margin",
            "0.3",
        ],
    ):
        args = trainer.parse_args()
    with pytest.raises(ValueError, match="frozen D3"):
        trainer.bind_stage2_run_contract_to_args(args, contract)


def test_legacy_cli_serialization_and_source_requirement_are_unchanged():
    from scripts import train_multisource_tail as trainer

    with patch.object(
        sys,
        "argv",
        ["train_multisource_tail", "--source-dirs", "source-a", "source-b"],
    ):
        args = trainer.parse_args()
    assert args.stage2_run_contract is None
    assert "stage2_run_contract" not in trainer.serialised_training_args(args)
    with patch.object(sys, "argv", ["train_multisource_tail"]):
        with pytest.raises(SystemExit):
            trainer.parse_args()


def test_restricted_inference_checkpoint_is_weights_only_loadable(contract_tree):
    from scripts.train_multisource_tail import save_stage2_inference_checkpoint

    fixture = _stage2_runtime_fixture(
        contract_tree,
        "restricted-checkpoint-test",
        role="detector_full_fit",
        fold=None,
    )
    run = fixture["contract"]
    model = torch.nn.Linear(2, 1)
    args = SimpleNamespace(
        stage2_run_contract=str(ROOT / fixture["entry"]["path"]),
        seed=run["derived_seed"],
        outer_fold_id=run["outer_fold_id"],
        outer_target=run["outer_target_domain"],
        held_out_domains=[run["outer_target_domain"]],
        stage2_detector_role=run["detector_role"],
        stage2_oof_fold_index=run["oof_fold_index"],
        base_size=256,
        crop_size=256,
        risk_objective="margin",
        tail_q=0.05,
        miss_q=0.25,
        object_pixel_q=0.25,
        peak_kernel_size=5,
        exclusion_radius=2,
        plateau_atol=0.0,
        tail_gamma=10.0,
        target_background_margin=1.0,
        lambda_margin=0.2,
    )
    run_dir = fixture["run_dir"]
    runtime = fixture["runtime"]
    binding = save_stage2_inference_checkpoint(
        run_dir,
        model,
        epoch=3,
        args=args,
        names=run["source_domains"],
        detector_source_records=[{"source_name": name} for name in run["source_domains"]],
        run_config_sha256=fixture["config_sha"],
        input_run_contract_sha256=fixture["contract_sha"],
        stage2_runtime_artifacts=runtime,
        stage2_verified_contract=run,
        execution_fingerprint_payload=fixture["fingerprint"],
    )
    payload = torch.load(
        run_dir / binding["path"], map_location="cpu", weights_only=True
    )
    assert payload["format_version"] == "rc-irstd.detector-inference.v1"
    assert payload["run_contract_sha256"] == fixture["contract_sha"]
    assert payload["detector_role"] == run["detector_role"]
    assert payload["oof_fold_index"] == run["oof_fold_index"]
    assert payload["run_config_sha256"] == payload["stage2_runtime_artifacts"][
        "run_config"
    ]["sha256"]
    assert payload["inference_geometry"] == {
        "input_hw": [256, 256],
        "resize_mode": "resize",
    }
    assert payload["official_test_accessed"] is False
    assert "optimizer" not in payload and "rng_state" not in payload
    assert _sha(run_dir / binding["path"]) == binding["sha256"]


def _verify_runtime_fixture(fixture: dict) -> dict:
    from scripts.train_multisource_tail import verify_stage2_runtime_artifacts

    return verify_stage2_runtime_artifacts(
        fixture["run_dir"],
        fixture["contract"],
        fixture["contract_sha"],
        fixture["config_sha"],
        fixture["fingerprint"],
        input_run_contract_path=str(ROOT / fixture["entry"]["path"]),
    )


def test_stage2_runtime_exact_schema_rejects_self_consistent_rewrites(contract_tree):
    fixture = _stage2_runtime_fixture(contract_tree, "runtime-exact-schema-test")
    original = fixture["runtime_payload"]
    mutations = []

    payload = copy.deepcopy(original)
    payload["development_only"] = 1
    mutations.append(payload)
    payload = copy.deepcopy(original)
    payload["unexpected_field"] = "self-consistent"
    mutations.append(payload)
    payload = copy.deepcopy(original)
    payload.pop("run_config")
    mutations.append(payload)
    payload = copy.deepcopy(original)
    payload["artifact_type"] = "wrong-runtime-type"
    mutations.append(payload)
    payload = copy.deepcopy(original)
    payload["observed_results"] = {}
    mutations.append(payload)
    payload = copy.deepcopy(original)
    payload["checkpoint_selection"] = "best-on-development"
    mutations.append(payload)
    payload = copy.deepcopy(original)
    payload["expected_artifacts"]["metrics"] = "other.jsonl"
    mutations.append(payload)
    payload = copy.deepcopy(original)
    payload["release_artifact"]["sha256"] = "0" * 64
    mutations.append(payload)

    for mutation in mutations:
        _write_json(fixture["runtime_path"], mutation)
        _write_adjacent_sha256(fixture["runtime_path"])
        with pytest.raises((TypeError, ValueError)):
            _verify_runtime_fixture(fixture)


@pytest.mark.parametrize("bad_path", ["/tmp/environment.json", "../environment.json"])
def test_stage2_runtime_rejects_absolute_and_traversal_bindings(
    contract_tree, bad_path
):
    suffix = "absolute" if bad_path.startswith("/") else "traversal"
    fixture = _stage2_runtime_fixture(contract_tree, f"runtime-path-{suffix}-test")
    payload = copy.deepcopy(fixture["runtime_payload"])
    payload["environment_artifact"]["path"] = bad_path
    _write_json(fixture["runtime_path"], payload)
    _write_adjacent_sha256(fixture["runtime_path"])
    with pytest.raises(ValueError, match="path mismatch"):
        _verify_runtime_fixture(fixture)


def test_stage2_runtime_rejects_environment_symlink_and_nonregular(contract_tree):
    symlink_fixture = _stage2_runtime_fixture(contract_tree, "runtime-env-symlink-test")
    environment = symlink_fixture["run_dir"] / "environment.json"
    target = symlink_fixture["run_dir"] / "environment-target.json"
    environment.rename(target)
    environment.symlink_to(target.name)
    with pytest.raises(ValueError, match="symlink"):
        _verify_runtime_fixture(symlink_fixture)

    nonregular_fixture = _stage2_runtime_fixture(
        contract_tree, "runtime-env-nonregular-test"
    )
    environment = nonregular_fixture["run_dir"] / "environment.json"
    environment.unlink()
    environment.mkdir()
    with pytest.raises(FileNotFoundError, match="regular file"):
        _verify_runtime_fixture(nonregular_fixture)


def test_json_stable_read_rejects_file_replacement(contract_tree):
    from scripts import train_multisource_tail as trainer

    fixture = _stage2_runtime_fixture(contract_tree, "runtime-stable-read-test")
    environment = fixture["run_dir"] / "environment.json"
    original_sha256 = trainer.sha256_file
    environment_calls = 0

    def replacing_sha256(path):
        nonlocal environment_calls
        path = Path(path)
        if path == environment:
            environment_calls += 1
            if environment_calls == 2:
                _write_json(environment, {"replaced": True})
        return original_sha256(path)

    with patch.object(trainer, "sha256_file", side_effect=replacing_sha256):
        with pytest.raises(RuntimeError, match="changed while read"):
            trainer._load_json_object(environment, "synthetic environment")


def test_stage2_checkpoint_divergence_fails_before_state_restore(contract_tree):
    from scripts import train_multisource_tail as trainer

    fixture = _stage2_runtime_fixture(contract_tree, "runtime-checkpoint-closure-test")
    contract = fixture["contract"]
    with patch.object(
        sys,
        "argv",
        [
            "train_multisource_tail",
            "--stage2-run-contract",
            str(ROOT / fixture["entry"]["path"]),
            "--seed",
            str(contract["derived_seed"]),
        ],
    ):
        args = trainer.parse_args()
    trainer.bind_stage2_run_contract_to_args(args, contract)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    records = [
        {"source_name": name, "num_samples": 1}
        for name in contract["source_domains"]
    ]
    trainer.save_checkpoint(
        fixture["run_dir"],
        model,
        optimizer,
        0,
        args,
        contract["source_domains"],
        records,
        {"epoch": 0},
        fixture["config_sha"],
        fixture["fingerprint"],
        fixture["contract_sha"],
        fixture["runtime"],
        contract,
    )
    metrics = fixture["run_dir"] / "metrics.jsonl"
    metrics.write_text('{"epoch": 0}\n', encoding="utf-8")
    _write_adjacent_sha256(metrics)
    checkpoint_path = fixture["run_dir"] / "checkpoint_last.pt"
    original_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    args.resume = str(checkpoint_path)

    mutations = []
    payload = copy.deepcopy(original_checkpoint)
    payload["stage2_runtime_artifacts"]["run_config"]["sha256"] = "0" * 64
    mutations.append(payload)
    payload = copy.deepcopy(original_checkpoint)
    payload["stage2_runtime_artifacts"].pop("release_artifact")
    mutations.append(payload)

    with patch.object(trainer, "_load_model_state_dict") as load_model, patch.object(
        optimizer, "load_state_dict"
    ) as load_optimizer, patch.object(trainer, "_restore_rng_state") as restore_rng:
        for mutation in mutations:
            torch.save(mutation, checkpoint_path)
            trainer.write_checkpoint_sha256(checkpoint_path)
            with pytest.raises(ValueError, match="runtime|provenance"):
                trainer.load_resume_checkpoint(
                    args,
                    model,
                    optimizer,
                    contract["source_domains"],
                    records,
                    torch.device("cpu"),
                    None,
                    fixture["contract_sha"],
                    contract,
                    fixture["fingerprint"],
                    True,
                )
        load_model.assert_not_called()
        load_optimizer.assert_not_called()
        restore_rng.assert_not_called()
