#
# Example call:
#   conda activate conda-mmdetection
#   python custom/group_runs.py
#
import argparse
import re
import shutil
import sys
from pathlib import Path

from mmengine.config import Config


IGNORED_FIELDS = {"seed", "run_name", "work_dir", "opts"}


def read_config(config_path):
    config = Config.fromfile(config_path, import_custom_modules=False).to_dict()
    for field in IGNORED_FIELDS:
        config.pop(field, None)
    (config.get("randomness") or {}).pop("seed", None)
    for backends in (config.get("vis_backends") or [],
                     (config.get("visualizer") or {}).get("vis_backends") or []):
        for backend in backends:
            if backend.get("type") == "WandbVisBackend":
                (backend.get("init_kwargs") or {}).pop("name", None)
    return config


def group_runs(runs_dir, target_dir=None):
    runs_dir = Path(runs_dir).resolve()
    target_dir = Path(target_dir).resolve() if target_dir is not None else runs_dir.parent / "results"
    if not runs_dir.is_dir():
        raise ValueError(f"Not a directory: {runs_dir}")

    groups = {}
    for run in sorted(runs_dir.iterdir()):
        match = re.fullmatch(r"(.+)_\d+_\d+", run.name)
        if run.is_dir() and match:
            groups.setdefault(match[1], []).append(run)
    if not groups:
        raise ValueError(f"No run directories ending in _<number>_<number> in {runs_dir}")

    # Validate every group before moving anything.
    plans = []
    for group, runs in groups.items():
        target = target_dir / group
        reference_path = target / "config.py" if target.exists() or target.is_symlink() else None
        reference = read_config(reference_path) if reference_path else None
        moves = {}
        for run in runs:
            config_path = run / "config.py"
            checkpoints = sorted(path for path in run.rglob("*.pth") if path.is_file())
            if not config_path.is_file() or not checkpoints:
                print(f"Warning: skipping {run}: missing config.py or .pth checkpoints")
                continue
            config = read_config(config_path)
            if reference is None:
                reference, reference_path = config, config_path
            elif config != reference:
                raise ValueError(f"Configs differ: {reference_path} vs {config_path}")

            for checkpoint in checkpoints:
                name = f"{run.name}.pth" if len(checkpoints) == 1 else f"{run.name}_{checkpoint.stem}.pth"
                existing = moves.get(name, target / name)
                if existing.exists() or existing.is_symlink():
                    print(f"Skipping {checkpoint}: destination name already in use: {name}")
                    continue
                moves[name] = checkpoint
        if moves:
            plans.append((target, {"config.py": reference_path, **moves}))

    for target, moves in plans:
        print(f"Organizing into {target}")
        target.mkdir(parents=True, exist_ok=True)
        for name, source in moves.items():
            if source != target / name:
                print(f"  {source} -> {name}")
                transfer = shutil.copy2 if name == "config.py" else shutil.move
                transfer(str(source), target / name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs",
                        help="Directory containing the individual run folders")
    parser.add_argument("--target_dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results",
                        help="Output directory; defaults to the repository's results folder")
    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        args = argparse.Namespace(
            runs_dir=Path(__file__).resolve().parents[1] / "runs",
            target_dir=None,
        )
    group_runs(args.runs_dir, target_dir=args.target_dir)
