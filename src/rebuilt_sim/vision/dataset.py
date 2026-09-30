"""Generate labeled synthetic images from simulated matches.

For each match: scripted bots play it, each robot gets one or two randomized cameras, and at
random moments the scene is rendered from some of those cameras, run through the camera effects,
and saved with its labels. Everything is seeded: the same config gives the same dataset, byte for
byte. Matches never share a split, so near-identical frames can't leak from train to validation.

Output (``datasets/<name>/``, see docs/04-vision.md):

- ``images/<split>/m<match>_f<frame>_c<camera>.jpg``: the camera images (PNG with ``lossless``);
- ``records/<split>/m<match>.jsonl.gz``: one JSON line per image with every label and the full
  camera and match state (``load_records`` reads them);
- ``index.csv``: one row per image, to browse and filter in a spreadsheet;
- ``manifest.json`` and ``dataset_card.md``: how it was made (config, git commit, versions) and
  how to read it;
- ``masks/`` and ``depth/`` (optional): per-pixel object ids and depth, as 16-bit PNGs.

Training formats (COCO, YOLO) are made from this afterwards with ``export.py``
(``scripts/export_dataset.py``), where *you* choose classes, box style and filters: those are
training decisions (CLAUDE.md, Rule #1). The generator never filters anything out.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import constants as C
from ..bots import ScriptedPolicy
from ..constants import Alliance
from ..robot import DAY1_TIER_WEIGHTS
from ..rules import period_at
from ..sim import HiFiConfig, MatchConfig, make_match
from . import effects as fx_mod
from .camera import Camera, Intrinsics, Mount
from .labels import annotate
from .randomize import PRESETS, CameraPreset, RandomizationConfig, random_intrinsics, random_light, random_mount
from .render import Renderer
from .scene import build_scene, match_look, robot_camera

SCHEMA = "rebuilt-vision/1"
LATER_SEASON_TIERS = ("elite", "strong", "mid", "low", "climber", "elite_climber")


@dataclass
class DatasetConfig:
    name: str = "rebuilt_v1"
    out_dir: str = "datasets"
    matches: int = 20
    frames_per_match: int = 40  # moments sampled per match
    cameras_per_frame: int = 1  # cameras rendered at each moment
    seed: int = 0
    presets: tuple[str, ...] = ("limelight4", "ov9281", "ov9782", "limelight3")
    scale: float = 0.5  # output resolution as a fraction of the preset's native one
    supersample: int = 2  # render k x k samples per output pixel, for smooth edges
    hifi: bool = True  # high-fidelity physics: real FUEL flight heights and robot motion
    randomize: bool = True  # domain randomization of colors, lighting and lenses
    effects_strength: float = 1.0  # camera effects: 0 = clean images
    time_range: tuple[float, float] = (0.0, C.MATCH_END)  # match times to sample from (s)
    min_gap: float = 1.0  # s between sampled moments of a match, so frames aren't near-copies
    later_season: float = 0.3  # chance a match uses a stronger robot mix than Day 1 in İstanbul
    splits: tuple[tuple[str, float], ...] = (("train", 0.8), ("val", 0.1), ("test", 0.1))
    masks: bool = False
    depth: bool = False
    lossless: bool = False  # PNG images (about 6x the disk of JPEG); otherwise JPEG, quality 95 or the effect's
    workers: int = 1
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)


# ---------------------------------------------------------------------------- seeds and splits
STREAMS = ("tiers", "match", "bots", "appearance", "rigs", "times", "lighting", "cameras", "effects")


def streams(cfg: DatasetConfig, match_id: int) -> dict[str, np.random.Generator]:
    """One random stream per purpose, so changing one setting (say, frames per match) doesn't
    reshuffle everything else about the match."""
    return {name: np.random.default_rng([cfg.seed, match_id, i]) for i, name in enumerate(STREAMS)}


def split_of(match_id: int, cfg: DatasetConfig) -> str:
    """The split a match belongs to. It depends only on the dataset seed and the match, so adding
    matches never moves existing ones."""
    h = hashlib.sha256(f"{cfg.seed}:{match_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2 ** 64
    acc = 0.0
    for name, frac in cfg.splits:
        acc += frac
        if u < acc:
            return name
    return cfg.splits[-1][0]


def sample_times(rng: np.random.Generator, n: int, lo: float, hi: float, gap: float) -> np.ndarray:
    """n sorted moments in [lo, hi], at least ``gap`` apart (when they fit)."""
    gap = min(gap, (hi - lo) / max(n, 1))
    return np.sort(rng.uniform(lo, hi - (n - 1) * gap, n)) + gap * np.arange(n)


@dataclass
class Rig:
    robot: int
    preset: CameraPreset
    intrinsics: Intrinsics
    mount: Mount
    role: str
    lens_tries: int


# ---------------------------------------------------------------------------- one match
def _tiers(rng: np.random.Generator, cfg: DatasetConfig) -> list[str]:
    if rng.random() < cfg.later_season:
        return [str(t) for t in rng.choice(LATER_SEASON_TIERS, 6)]
    names = list(DAY1_TIER_WEIGHTS)
    p = np.array([DAY1_TIER_WEIGHTS[n] for n in names])
    return [str(t) for t in rng.choice(names, 6, p=p / p.sum())]


def generate_match(match_id: int, cfg: DatasetConfig) -> list[dict]:
    """Play one match, write its images and its records shard; returns its index rows."""
    root = Path(cfg.out_dir) / cfg.name
    split = split_of(match_id, cfg)
    rs = streams(cfg, match_id)
    rc = cfg.randomization if cfg.randomize else None
    tiers = _tiers(rs["tiers"], cfg)
    sim_seed = int(rs["match"].integers(2 ** 31))
    match = make_match(tiers[:3], tiers[3:], seed=sim_seed, config=MatchConfig(hifi=HiFiConfig() if cfg.hifi else None))
    policy = ScriptedPolicy(match, seed=int(rs["bots"].integers(2 ** 31)))
    look = match_look(match, rs["appearance"], cfg.randomize)
    light = random_light(rs["lighting"], rc)
    rigs = []
    for r in match.robots:
        for _ in range(1 + int(rs["rigs"].random() < cfg.randomization.second_camera)):
            preset = PRESETS[str(rs["rigs"].choice(cfg.presets))]
            mount, role = random_mount(r, rs["rigs"], cfg.randomization)
            intr, tries = random_intrinsics(preset, cfg.scale, rs["rigs"], rc)
            rigs.append(Rig(r.index, preset, intr, mount, role, tries))
    renderers: dict[int, Renderer] = {}
    times = sample_times(rs["times"], cfg.frames_per_match, *cfg.time_range, cfg.min_gap)
    records, rows = [], []
    k = cfg.supersample
    for f_i, t in enumerate(times):
        while match.t < t and not match.done:
            match.step(policy())
        scene, objects = build_scene(match, look)
        chosen = rs["cameras"].choice(len(rigs), min(cfg.cameras_per_frame, len(rigs)), replace=False)
        for c_i, rig_i in enumerate(int(i) for i in chosen):
            rig = rigs[rig_i]
            if rig_i not in renderers:
                if len(renderers) >= 3:  # each holds a ray per pixel: keep only a few in memory
                    renderers.pop(next(iter(renderers)))
                renderers[rig_i] = Renderer(rig.intrinsics.scaled(k))
            cam = robot_camera(match, rig.robot, rig.intrinsics, rig.mount)
            frame = renderers[rig_i].render(scene, Camera(rig.intrinsics.scaled(k), cam.position, cam.rotation), light)
            labels = annotate(frame, cam, objects, k)
            effects = frame_effects(rs["effects"], cfg, rig, match.robots[rig.robot])
            image = fx_mod.apply(fx_mod.downsample(frame.color, k), effects, rs["effects"])
            stem = f"m{match_id:05d}_f{f_i:03d}_c{c_i}"
            rec = {"schema": SCHEMA, **_save(root, split, stem, image, effects, frame, k, cfg)}
            rec.update({
                "split": split, "width": rig.intrinsics.width, "height": rig.intrinsics.height,
                "channels": 1 if effects.grayscale else 3,
                "match": _match_record(match, match_id, sim_seed, tiers, cfg),
                "camera": _camera_record(rig, cam, k),
                "robot": _robot_record(match, rig.robot, look),
                "render": {"light": asdict(light), "effects": effects.to_dict()},
                "objects": [_clean(o) for o in labels],
            })
            records.append(_clean(rec))
            rows.append(_index_row(rec))
    shard = root / "records" / split / f"m{match_id:05d}.jsonl.gz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    _write_shard(shard, records)
    return rows


def _write_shard(path: Path, records: list[dict]) -> None:
    """A gzipped JSON-lines file. No timestamp in the gzip header, so reruns give identical bytes."""
    data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records).encode("utf-8")
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(data)


def frame_effects(rng: np.random.Generator, cfg: DatasetConfig, rig: Rig, robot) -> fx_mod.Effects:
    """Camera effects for one frame. Motion blur comes from how far the robot turns and drives
    during the exposure: blur (px) = turn rate x exposure time x focal length (px)."""
    if not cfg.randomize or cfg.effects_strength <= 0:
        return fx_mod.Effects(grayscale=rig.preset.mono)
    rc = cfg.randomization
    fx = fx_mod.random_effects(rng, cfg.effects_strength, mono=rig.preset.mono, rc=rc)
    exposure_ms = rng.uniform(*(rc.exposure_ms_mono if rig.preset.mono else rc.exposure_ms_color))
    focal = rig.intrinsics.fx
    turn = abs(robot.omega) * exposure_ms / 1000.0 * focal
    drive = math.hypot(robot.vx, robot.vy) * exposure_ms / 1000.0 * focal / 3.0  # at a typical 3 m depth
    fx.motion_px = float((turn + drive) * cfg.effects_strength)
    fx.motion_angle = float(rng.normal(0.0, 0.1))  # a yawing robot smears the image sideways
    gain_db = rng.uniform(*rc.gain_db)  # teams run high gain with short exposures: noisier images
    fx.read_noise = float(0.003 * 10 ** (gain_db / 20) * cfg.effects_strength)
    fx.exposure_ms, fx.gain_db = float(exposure_ms), float(gain_db)
    return fx


def _match_record(match, match_id: int, sim_seed: int, tiers: list[str], cfg: DatasetConfig) -> dict:
    return {"id": match_id, "sim_seed": sim_seed, "t": float(match.t), "period": period_at(match.t).name,
            "physics": "high_fidelity" if cfg.hifi else "strategy", "tiers": tiers,
            "hub_active": {a.name.lower(): bool(match.hub_active(a)) for a in Alliance},
            "score": [s.total for s in match.scores]}


def _camera_record(rig: Rig, cam: Camera, k: int) -> dict:
    intr = rig.intrinsics
    m = rig.mount
    return {
        "preset": rig.preset.name, "role": rig.role, "mono": rig.preset.mono, "robot_index": rig.robot,
        "K": intr.matrix.tolist(), "dist": list(intr.dist),  # OpenCV k1 k2 p1 p2 k3
        "lens_tries": rig.lens_tries, "supersample": k,
        "mount": {"x": m.x, "y": m.y, "z": m.z, "yaw_deg": math.degrees(m.yaw),
                  "pitch_up_deg": math.degrees(m.pitch_up), "roll_deg": math.degrees(m.roll)},
        "robot_to_camera_wpilib": m.to_wpilib(),
        "pose": cam.pose(),  # the true camera pose, BUMP tilt and climbing included
    }


def _robot_record(match, index: int, look) -> dict:
    from .robot_model import robot_frame

    r = match.robots[index]
    base, rot = robot_frame(r)
    rec = {"index": index, "alliance": r.alliance.name.lower(), "tier": r.spec.name,
           "team_number": look.styles[index].number,
           "pose": {"x": float(base[0]), "y": float(base[1]), "z": float(base[2]), "heading_deg": math.degrees(r.heading),
                    "pitch_up_deg": math.degrees(math.asin(max(-1.0, min(1.0, rot[2, 0]))))},
           "twist": {"vx": r.vx, "vy": r.vy, "omega": r.omega}, "pose_estimate": None}
    if match.sensors is not None:  # what the robot's own software believes (high fidelity)
        x, y, heading, _, _ = match.perceived(index)
        rec["pose_estimate"] = {"x": x, "y": y, "heading_deg": math.degrees(heading)}
    return rec


def _index_row(rec: dict) -> dict:
    seen = [o for o in rec["objects"] if o["bbox"] is not None]
    vis = [o["visibility"] for o in seen]
    return {"image": rec["image"], "split": rec["split"], "match_id": rec["match"]["id"], "t": round(rec["match"]["t"], 2),
            "period": rec["match"]["period"], "preset": rec["camera"]["preset"], "role": rec["camera"]["role"],
            "robot_index": rec["camera"]["robot_index"],
            **{f"n_{c}": sum(o["category"] == c for o in seen) for c in ("fuel", "robot", "apriltag")},
            "n_visible": len(seen), "mean_visibility": round(float(np.mean(vis)), 3) if vis else ""}


def _clean(value):
    """JSON-safe: plain floats (6 decimals), None for NaN, no numpy types."""
    if isinstance(value, dict):
        return {k: _clean(x) for k, x in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _save(root: Path, split: str, stem: str, image: np.ndarray, effects: fx_mod.Effects, frame, k: int,
          cfg: DatasetConfig) -> dict:
    from PIL import Image

    rel = Path("images") / split / f"{stem}.{'png' if cfg.lossless else 'jpg'}"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(image[..., 0] if effects.grayscale else image)
    if cfg.lossless:
        if effects.jpeg_quality:  # the compression effect, then stored losslessly
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(effects.jpeg_quality))
            img = Image.open(io.BytesIO(buf.getvalue()))
        img.save(root / rel)
    else:
        img.save(root / rel, quality=int(effects.jpeg_quality or 95))
    rec = {"image": rel.as_posix(), "masks": None, "depth": None}
    if cfg.masks:  # the object id at each output pixel's center sample (+1, so 0 = nothing)
        ids = frame.obj[k // 2::k, k // 2::k].astype(np.int64) + 1
        path = Path("masks") / split / f"{stem}.png"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ids.astype(np.uint16)).save(root / path)
        rec["masks"] = path.as_posix()
    if cfg.depth:
        d = frame.depth[k // 2::k, k // 2::k]
        mm = np.where(np.isfinite(d), np.clip(d * 1000.0, 0, 65535), 0).astype(np.uint16)
        path = Path("depth") / split / f"{stem}.png"
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mm).save(root / path)
        rec["depth"] = path.as_posix()
    return rec


# ---------------------------------------------------------------------------- the whole dataset
def load_records(root, split: str | None = None):
    """Every image's record, in a fixed order (by split, then match, then frame)."""
    for shard in sorted((Path(root) / "records").glob(f"{split or '*'}/m*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _job(args):
    match_id, cfg = args
    return match_id, generate_match(match_id, cfg)


def _existing_rows(root: Path, match_id: int, cfg: DatasetConfig) -> list[dict] | None:
    shard = root / "records" / split_of(match_id, cfg) / f"m{match_id:05d}.jsonl.gz"
    if not shard.exists():
        return None
    with gzip.open(shard, "rt", encoding="utf-8") as f:
        return [_index_row(json.loads(line)) for line in f]


def _jsonable(cfg: DatasetConfig) -> dict:
    return json.loads(json.dumps(asdict(cfg)))


def generate(cfg: DatasetConfig, progress=print, overwrite: bool = False) -> dict:
    """Generate the whole dataset, in parallel with ``cfg.workers`` processes. Matches already on
    disk are kept, so an interrupted run resumes where it stopped (unless ``overwrite``)."""
    root = Path(cfg.out_dir) / cfg.name
    root.mkdir(parents=True, exist_ok=True)
    cfg_file = root / "config.json"
    same = {k: v for k, v in _jsonable(cfg).items() if k not in ("matches", "workers")}
    if cfg_file.exists() and not overwrite:
        old = {k: v for k, v in json.loads(cfg_file.read_text(encoding="utf-8")).items() if k not in ("matches", "workers")}
        if old != same:
            raise ValueError(f"{root} was made with a different config: pick another --name, or --overwrite")
    cfg_file.write_text(json.dumps(_jsonable(cfg), indent=2), encoding="utf-8")
    start = time.perf_counter()
    rows: dict[int, list[dict]] = {}
    todo = []
    for m in range(cfg.matches):
        old_rows = None if overwrite else _existing_rows(root, m, cfg)
        if old_rows is None:
            todo.append(m)
        else:
            rows[m] = old_rows
    if rows:
        progress(f"{len(rows)} matches already on disk, {len(todo)} to go")

    def report():
        progress(f"match {len(rows)}/{cfg.matches}: {sum(map(len, rows.values()))} images, "
                 f"{time.perf_counter() - start:.0f} s")

    if cfg.workers > 1 and len(todo) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(cfg.workers, len(todo))) as pool:
            for m, r in pool.imap_unordered(_job, [(m, cfg) for m in todo]):
                rows[m] = r
                report()
    else:
        for m in todo:
            rows[m] = generate_match(m, cfg)
            report()
    index = [row for m in sorted(rows) for row in rows[m]]
    with open(root / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index[0]) if index else ["image"])
        writer.writeheader()
        writer.writerows(index)
    from .card import dataset_card, stats, write_manifest

    st = stats(root)
    write_manifest(root, cfg, st, time.perf_counter() - start)
    (root / "dataset_card.md").write_text(dataset_card(cfg, st), encoding="utf-8")
    return st


def dry_run(cfg: DatasetConfig) -> dict:
    """Rough time, disk and memory needs, before generating anything. Timings were measured on
    one core of the development container; a desktop i7 core is likely somewhat faster."""
    images = cfg.matches * cfg.frames_per_match * cfg.cameras_per_frame
    pixels = float(np.mean([PRESETS[p].width * PRESETS[p].height for p in cfg.presets])) * cfg.scale ** 2
    fine = pixels * cfg.supersample ** 2
    s_render = 1.35 * fine / 1.0e6 + 0.1  # measured: about 1.35 s per million rendered pixels
    s_sim = (12.0 if cfg.hifi else 4.0) * (cfg.time_range[1] - cfg.time_range[0]) / C.MATCH_END
    seconds = (images * s_render + cfg.matches * s_sim) / max(cfg.workers, 1)
    kb = pixels / 256_000 * (170 if cfg.lossless else 35) + 12  # image + gzipped record
    kb += pixels / 256_000 * ((15 if cfg.masks else 0) + (30 if cfg.depth else 0))
    return {"images": images, "hours": round(seconds / 3600, 2), "disk_gb": round(images * kb / 1e6, 2),
            "ram_gb": round(cfg.workers * (0.25 + 3e-7 * fine), 1)}
