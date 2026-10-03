"""Render one camera view of a simulated match, with its labels drawn on top (to look at what the
dataset generator makes, and check the labels).

    python scripts/render_view.py                                  # robot 0's camera, 30 s into a match
    python scripts/render_view.py --t 75 --robot 3 --preset ov9782 --pitch -15
    python scripts/render_view.py --overview --out field.png       # a view over the whole field

Writes view.png (the camera image) and view_labels.png (with boxes and tag corners).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t", type=float, default=30.0, help="match time to render (s)")
    ap.add_argument("--robot", type=int, default=0, help="whose camera (0-2 blue, 3-5 red)")
    ap.add_argument("--preset", default="limelight4")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--height", type=float, default=0.45, help="camera height (m)")
    ap.add_argument("--pitch", type=float, default=0.0, help="camera tilt, degrees up")
    ap.add_argument("--yaw", type=float, default=0.0, help="camera yaw on the robot, degrees left")
    ap.add_argument("--overview", action="store_true", help="a high view from behind the blue wall instead")
    ap.add_argument("--effects", type=float, default=0.0, help="camera effects strength (0 = clean)")
    ap.add_argument("--strategy-physics", action="store_true")
    ap.add_argument("--out", default="view.png")
    args = ap.parse_args()

    from PIL import Image

    from rebuilt_sim.vision.overlay import draw_labels

    out = Path(args.out)
    from rebuilt_sim.bots import ScriptedPolicy
    from rebuilt_sim.sim import HiFiConfig, MatchConfig, make_match
    from rebuilt_sim.vision import effects as fx
    from rebuilt_sim.vision.camera import Camera, Intrinsics, Mount
    from rebuilt_sim.vision.randomize import PRESETS
    from rebuilt_sim.vision.labels import annotate
    from rebuilt_sim.vision.render import Renderer
    from rebuilt_sim.vision.scene import build_scene, match_look, robot_camera

    config = MatchConfig(hifi=None if args.strategy_physics else HiFiConfig())
    m = make_match(["elite", "strong", "mid"], ["strong", "mid", "elite_climber"], seed=args.seed, config=config)
    pol = ScriptedPolicy(m, seed=args.seed)
    while m.t < args.t and not m.done:
        m.step(pol())
    rng = np.random.default_rng(args.seed)
    look = match_look(m, rng, randomize=False)
    p = PRESETS[args.preset]
    intr = Intrinsics.from_fov(round(p.width * args.scale), round(p.height * args.scale), p.hfov_deg)
    if args.overview:
        cam = Camera.looking(intr, (-1.2, 0.5, 3.0), math.radians(22), math.radians(-20))
    else:
        r = m.robots[args.robot]
        mount = Mount(r.spec.length / 2 - 0.08, 0.0, args.height, math.radians(args.yaw), math.radians(args.pitch))
        cam = robot_camera(m, args.robot, intr, mount)
    k = 2
    scene, objects = build_scene(m, look)
    frame = Renderer(intr.scaled(k)).render(scene, Camera(intr.scaled(k), cam.position, cam.rotation))
    labels = annotate(frame, cam, objects, k)
    effects = fx.random_effects(rng, args.effects) if args.effects > 0 else fx.Effects()
    image = fx.apply(fx.downsample(frame.color, k), effects, rng)
    Image.fromarray(image).save(out)
    draw_labels(image, labels).save(out.with_name(out.stem + "_labels.png"))
    seen = [o for o in labels if o["bbox"] is not None]
    tags = sorted(o["tag_id"] for o in seen if o["category"] == "apriltag")
    print(f"t = {m.t:.1f} s, {len(seen)} objects in view ({sum(o['category'] == 'fuel' for o in seen)} FUEL), tags {tags}")
    print(f"wrote {out} and {out.with_name(out.stem + '_labels.png')}")


if __name__ == "__main__":
    main()
