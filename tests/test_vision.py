"""The vision track (step 4): cameras, the renderer, the 3D field, labels and the dataset generator.

The strongest checks close the loop with tools that share no code with the renderer: an
independent AprilTag detector must read the rendered tags with the labeled IDs and corners, and
the fast renderer must match a brute-force version of itself pixel for pixel.
"""

import json
import math

import numpy as np
import pytest

from rebuilt_sim import constants as C
from rebuilt_sim.apriltags import TAGS
from rebuilt_sim.sim import MatchConfig, make_match
from rebuilt_sim.vision import effects as fx
from rebuilt_sim.vision import ids
from rebuilt_sim.vision import tags as T
from rebuilt_sim.vision.camera import Camera, Intrinsics, Mount
from rebuilt_sim.vision.check import validate
from rebuilt_sim.vision.dataset import DatasetConfig, generate, load_records, split_of
from rebuilt_sim.vision.export import ExportConfig, export
from rebuilt_sim.vision.field_model import HUB_X, bump_height
from rebuilt_sim.vision.labels import ObjectInfo, annotate
from rebuilt_sim.vision.randomize import PRESETS, RandomizationConfig, random_intrinsics
from rebuilt_sim.vision.render import Box, Cylinder, Polygon, Renderer, Scene, Sphere
from rebuilt_sim.vision.robot_model import robot_frame
from rebuilt_sim.vision.scene import build_scene, fuel_positions, hub_light, match_look, robot_camera

LENS = (-0.1, 0.02, 0.001, -0.001, 0.0)


def _match(seed=0):
    return make_match(["mid"] * 3, ["mid"] * 3, seed=seed)


def _look(m):
    return match_look(m, np.random.default_rng(0), randomize=False)


def _render(scene, cam, k=1, full=False):
    fine = cam.intrinsics.scaled(k)
    return Renderer(fine, full_windows=full).render(scene, Camera(fine, cam.position, cam.rotation))


# ------------------------------------------------------------------------------ cameras
def test_projection_follows_opencv_conventions():
    intr = Intrinsics.from_fov(640, 480, 90.0)
    cam = Camera.looking(intr, (0.0, 0.0, 1.0), yaw=0.0)  # looking down the field's +x axis
    uv, depth = cam.project(np.array([[5.0, 0.0, 1.0], [5.0, 1.0, 2.0], [-1.0, 0.0, 1.0]]))
    assert np.allclose(uv[0], (intr.cx, intr.cy)) and depth[0] == pytest.approx(5.0)
    assert uv[1, 0] < intr.cx and uv[1, 1] < intr.cy  # up and to the left (+y is left in the field)
    assert np.isnan(uv[2]).all()  # behind the camera
    up = Camera.looking(intr, (0.0, 0.0, 1.0), yaw=0.0, pitch_up=math.radians(20))
    assert up.project(np.array([[5.0, 0.0, 1.0]]))[0][0, 1] > intr.cy  # tilted up: straight ahead is low


def test_camera_pose_round_trip():
    intr = Intrinsics.from_fov(64, 48, 70.0)
    for yaw, pitch, roll in ((0.3, 0.2, 0.0), (-2.5, -0.4, 0.1), (math.pi / 2, 0.0, -0.2)):
        p = Camera.looking(intr, (1.0, 2.0, 0.5), yaw, pitch, roll).pose()
        assert p["yaw_deg"] == pytest.approx(math.degrees(yaw), abs=1e-6)
        assert p["pitch_up_deg"] == pytest.approx(math.degrees(pitch), abs=1e-6)
        assert p["roll_deg"] == pytest.approx(math.degrees(roll), abs=1e-6)


def test_wpilib_and_opencv_pose_conventions():
    """A camera tilted up by 20 degrees is WPILib pitch -20 (WPILib's pitch is nose-down), and the
    OpenCV extrinsics in a pose reproduce the camera's own projection."""
    from rebuilt_sim.vision.camera import quaternion, rodrigues

    w, x, y, z = Mount(0.2, 0.0, 0.3, 0.0, math.radians(20)).to_wpilib()["quaternion"]
    assert (w, x, z) == pytest.approx((math.cos(math.radians(10)), 0.0, 0.0)) and y == pytest.approx(-math.sin(math.radians(10)))
    rz = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 degrees about z
    assert np.allclose(rodrigues(rz), (0, 0, math.pi / 2)) and np.allclose(quaternion(rz), (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
    intr = Intrinsics(64, 48, 50.0, 50.0, 31.5, 23.5, LENS)
    cam = Camera.looking(intr, (1.0, 2.0, 0.5), 0.7, 0.2, 0.05)
    cv = cam.pose()["opencv"]
    pts = np.array([[4.0, 3.5, 1.0], [3.0, 2.0, 0.2]])
    pc = pts @ np.array(cv["R"]).T + np.array(cv["t"])
    u, v = intr.distort(pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2])
    assert np.allclose(np.column_stack([u, v]), cam.project(pts)[0])
    wp = cam.pose()["wpilib"]
    assert np.allclose(wp["translation"], cam.position)
    qw, qx, qy, qz = wp["quaternion"]  # the camera's forward axis is its x axis in WPILib
    fwd = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)])
    assert np.allclose(fwd, cam.forward)


def test_fuel_truncation_at_the_image_edge():
    intr = Intrinsics(80, 60, 60.0, 60.0, 39.5, 29.5)
    cam = Camera.looking(intr, (0.0, 0.0, 0.0), 0.0)
    edge_y = (39.5 + 0.5) / 60.0 * 2.0  # at 2 m, the image's left edge (+y is left)
    labels = {}
    for i, y in enumerate((0.0, edge_y, edge_y + 0.2)):
        c = np.array([2.0, y, 0.0])
        info = ObjectInfo("fuel", {}, c + np.vstack([np.eye(3), -np.eye(3)]) * 0.1, sphere=(c, 0.1))
        f = _render(Scene(spheres=[Sphere(tuple(c), 0.1, (1, 1, 0), i)]), cam)
        labels.update({o["id"]: o for o in annotate(f, cam, {i: info})})
    assert labels[0]["truncation"] == 0.0
    assert labels[1]["truncation"] == pytest.approx(0.5, abs=0.05)
    assert 2 not in labels  # entirely outside the image: not in the frame at all


def test_lens_model_round_trip_and_fold_over():
    intr = Intrinsics(320, 200, 250.0, 250.0, 159.5, 99.5, LENS)
    u, v = np.meshgrid(np.arange(0, 320, 7.0), np.arange(0, 200, 7.0))
    x, y = intr.undistort(u, v, iterations=20)
    u2, v2 = intr.distort(x, y)
    assert np.abs(u2 - u).max() < 1e-6 and np.abs(v2 - v).max() < 1e-6
    folds = Intrinsics(640, 400, 368.0, 368.0, 319.5, 199.5, (-0.25, 0.0, 0.0, 0.0, 0.0))  # wide lens, strong barrel
    assert folds.lens_error() == float("inf")
    with pytest.raises(ValueError):
        Renderer(folds)
    rng = np.random.default_rng(3)
    tries = []
    for _ in range(25):  # the dataset's lens sampler only returns valid lenses
        k, n = random_intrinsics(PRESETS[str(rng.choice(list(PRESETS)))], 0.25, rng, RandomizationConfig())
        assert k.lens_error(step=2) < 1e-6
        tries.append(n)
    assert max(tries) > 1  # some draws were rejected and redrawn, so the check matters


def test_robot_camera_is_the_mount_on_flat_ground_and_tilts_on_a_bump():
    m = _match()
    r = m.robots[0]
    intr = Intrinsics.from_fov(64, 48, 70.0)
    mount = Mount(0.3, 0.1, 0.5, 0.2, 0.1, 0.02)
    r.x, r.y, r.heading = 2.0, 2.5, 0.3
    a, b = robot_camera(m, 0, intr, mount), Camera.on_robot(intr, r.x, r.y, r.heading, mount)
    assert np.allclose(a.position, b.position) and np.allclose(a.rotation, b.rotation)
    r.x, r.heading = HUB_X - 0.5, 0.0  # front wheels up on the blue BUMP's near ramp
    base, rot = robot_frame(r)
    assert base[2] > 0.03 and math.degrees(math.asin(rot[2, 0])) > 5.0  # nose up


# ------------------------------------------------------------------------------ AprilTags
def test_tag_patterns():
    fixture = ["..........", ".########.", ".#..#..##.", ".##.#...#.", ".#....###.",
               ".##..####.", ".#.#..#.#.", ".###.##.#.", ".########.", ".........."]
    assert ["".join("." if c else "#" for c in row) for row in T.tag_cells(1)] == fixture

    def bits(cells):
        return cells[2:8, 2:8].astype(bool).ravel()

    codes = {t: T.tag_cells(t) for t in T.IDS}
    for a in T.IDS:
        for b in T.IDS:
            for rot in range(4):
                if a == b and rot == 0:
                    continue
                # 36h11: any two codes, in any rotation, differ in at least 11 bits
                assert (bits(np.rot90(codes[a], rot)) != bits(codes[b])).sum() >= 11


def test_tag_geometry_matches_the_layout():
    for tid, x, y, z, yaw in TAGS:
        bl, br, tr, tl = T.tag_corners(tid)
        for p, q in ((bl, br), (br, tr), (tr, tl), (tl, bl)):
            assert np.linalg.norm(q - p) == pytest.approx(T.TAG_SIZE)
        assert np.allclose((bl + tr) / 2, (x, y, z))
        normal = np.cross(br - bl, tl - bl)
        assert np.allclose(normal / np.linalg.norm(normal), (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0))
        assert tl[2] > bl[2] and tr[2] > br[2]


# ------------------------------------------------------------------------------ renderer
def _random_scene(rng):
    scene = Scene()
    for _ in range(6):
        c = rng.uniform((0.5, -1.5, 0.0), (5.0, 1.5, 1.5))
        a = rng.uniform(0, math.pi)
        rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        scene.boxes.append(Box(tuple(c), tuple(rng.uniform(0.05, 0.6, 3)), (0.5, 0.2, 0.2), rot, int(rng.integers(100))))
        scene.spheres.append(Sphere(tuple(rng.uniform((0.3, -1.5, 0.0), (5.0, 1.5, 1.5))), 0.075, (0.9, 0.8, 0.1),
                                    int(rng.integers(100, 200)), shiny=True))
        scene.cylinders.append(Cylinder(tuple(rng.uniform((0.5, -1.5, 0.0), (5.0, 1.5, 1.5))), 0.05, 0.4, (0.2, 0.2, 0.8),
                                        rot, int(rng.integers(200, 300))))
    # long shapes that pass behind the camera, where the window clipping matters most
    scene.boxes.append(Box((0.0, 1.2, 0.3), (6.0, 0.02, 0.3), (0.6, 0.6, 0.6), obj=300))
    scene.polygons.append(Polygon(np.array([[-4.0, -1.0, 0.8], [6.0, -1.0, 0.8], [6.0, -1.0, 0.1], [-4.0, -1.0, 0.1]]),
                                  obj=301))
    scene.floor = lambda x, y: np.full((x.size, 3), 0.3, dtype=np.float32)
    return scene


def test_renderer_matches_its_brute_force_reference():
    """Screen windows and near-plane clipping are pure speed-ups: rendering every shape over every
    pixel must give the same image, ids and depths."""
    rng = np.random.default_rng(0)
    for trial in range(4):
        intr = Intrinsics(80, 60, 50.0, 50.0, 39.5, 29.5, LENS if trial % 2 else (0.0,) * 5)
        scene = _random_scene(rng)
        cam = Camera.looking(intr, (0.0, 0.0, 0.6), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3))
        fast, slow = _render(scene, cam), _render(scene, cam, full=True)
        assert (fast.obj == slow.obj).all()
        assert np.array_equal(fast.depth, slow.depth) and np.array_equal(fast.color, slow.color)
        assert fast.coverage == slow.coverage and fast.extent == slow.extent


def test_pixel_centers_follow_the_opencv_convention():
    """A square seen head-on: its rendered centroid lands on its projected center, and its area
    matches, to a small fraction of a pixel. A half-pixel slip would be obvious."""
    intr = Intrinsics(64, 48, 60.0, 60.0, 31.5, 23.5)
    cam = Camera.looking(intr, (0.0, 0.0, 0.0), 0.0)
    c = np.array([2.0, -0.1234, 0.0567])
    h = 0.25
    sq = Polygon(np.array([c + (0, h, h), c + (0, -h, h), c + (0, -h, -h), c + (0, h, -h)]), obj=1)
    k = 16
    f = _render(Scene(polygons=[sq]), cam, k)
    v, u = np.nonzero(f.obj == 1)
    centroid = np.array([(u.mean() - (k - 1) / 2) / k, (v.mean() - (k - 1) / 2) / k])
    uv, _ = cam.project(c[None])
    assert np.abs(centroid - uv[0]).max() < 0.02
    assert u.size / k ** 2 == pytest.approx((2 * h * intr.fx / c[0]) ** 2, rel=0.005)


def test_sphere_extent_is_the_analytic_silhouette():
    intr = Intrinsics(80, 60, 50.0, 50.0, 39.5, 29.5)
    cam = Camera.looking(intr, (0.0, 0.0, 0.0), 0.0)
    k = 8
    for center in ((1.5, 0.3, 0.2), (2.5, -0.6, -0.4), (0.9, 0.1, 0.25)):
        f = _render(Scene(spheres=[Sphere(center, 0.1, (1, 1, 1), 7)]), cam, k)
        X, Y, Z = cam.to_camera(np.array([center]))[0]
        r = 0.1
        lo_hi = []
        for P in (X, Y):  # tangent planes through the lens: s = (PZ +- r sqrt(P^2 + Z^2 - r^2)) / (Z^2 - r^2)
            root = r * math.sqrt(P * P + Z * Z - r * r)
            lo_hi.append(sorted(((P * Z - root) / (Z * Z - r * r), (P * Z + root) / (Z * Z - r * r))))
        (x0, x1), (y0, y1) = lo_hi
        u0, u1, v0, v1 = f.extent[7]
        # the extent is in fine pixels; the silhouette edge lies within one fine pixel of it
        assert abs((u0 - (k - 1) / 2) / k - (intr.fx * x0 + intr.cx)) < 1.0 / k + 1e-3
        assert abs((u1 - (k - 1) / 2) / k - (intr.fx * x1 + intr.cx)) < 1.0 / k + 1e-3
        assert abs((v0 - (k - 1) / 2) / k - (intr.fy * y0 + intr.cy)) < 1.0 / k + 1e-3
        assert abs((v1 - (k - 1) / 2) / k - (intr.fy * y1 + intr.cy)) < 1.0 / k + 1e-3


def test_occlusion_visibility_and_boxes():
    intr = Intrinsics(80, 60, 60.0, 60.0, 39.5, 29.5)
    cam = Camera.looking(intr, (0.0, 0.0, 0.5), 0.0)
    scene = Scene(spheres=[Sphere((3.0, 0.0, 0.5), 0.1, (1, 1, 0), 1), Sphere((3.0, 0.38, 0.5), 0.1, (1, 1, 0), 2),
                           Sphere((1.0, 0.62, 0.5), 0.1, (1, 1, 0), 3)],
                  boxes=[Box((2.0, 0.08, 0.5), (0.05, 0.15, 0.3), (0.5, 0.5, 0.5), obj=4)])
    axes = np.vstack([np.eye(3), -np.eye(3)]) * 0.1
    objects = {i: ObjectInfo("fuel", {}, np.array(s.center) + axes) for i, s in zip((1, 2, 3), scene.spheres)}
    labels = {o["id"]: o for o in annotate(_render(scene, cam, 2), cam, objects, 2)}
    assert labels[1]["visibility"] == 0.0 and labels[1]["bbox"] is None  # right behind the box
    assert 0.0 < labels[2]["visibility"] < 1.0  # half behind it
    assert 0.0 < labels[3]["truncation"] < 1.0  # cut by the image edge
    x, y, w, h = labels[2]["bbox"]
    ax, ay, aw, ah = labels[2]["bbox_amodal"]
    assert ax <= x and ay <= y and x + w <= ax + aw and y + h <= ay + ah


def test_back_facing_tags_are_not_drawn():
    m = _match()
    scene, _ = build_scene(m, _look(m))
    intr = Intrinsics.from_fov(96, 64, 60.0)
    c, n, _, _ = T.tag_frame(17)  # a TRENCH tag, printed on one side of its bracket
    front = _render(scene, Camera.looking(intr, c + 1.0 * n, math.atan2(-n[1], -n[0])))
    assert front.coverage.get(ids.TAG + 17, 0) > 0
    behind = _render(scene, Camera.looking(intr, c - 1.0 * n, math.atan2(n[1], n[0])))
    assert ids.TAG + 17 not in behind.coverage and behind.coverage.get(ids.TAG + 28, 0) > 0  # its twin on the back


def test_every_tag_is_fully_visible_head_on():
    """Each of the 32 tags, seen from 1 m straight in front: nothing of the field hides any of it."""
    m = _match()
    look = _look(m)
    scene, objects = build_scene(m, look, hide=set(range(6)))
    intr = Intrinsics.from_fov(120, 90, 60.0)
    for tid in T.IDS:
        c, n, _, _ = T.tag_frame(tid)
        cam = Camera.looking(intr, c + 1.0 * n, math.atan2(-n[1], -n[0]))
        labels = {o["id"]: o for o in annotate(_render(scene, cam, 2), cam, objects, 2)}
        tag = labels[ids.TAG + tid]
        assert tag["visibility"] > 0.999 and all(tag["corners_visible"]), tid
        uv, _ = cam.project(T.tag_corners(tid))
        assert np.allclose(tag["corners"], uv)


def test_rendered_tag_edges_match_the_labels():
    """Just inside a tag's printed edge the id buffer shows the tag; just outside it doesn't, even
    through a distorting lens."""
    m = _match()
    scene, _ = build_scene(m, _look(m))
    intr = Intrinsics(320, 200, 250.0, 250.0, 159.5, 99.5, LENS)
    cam = Camera.looking(intr, (2.6, 4.2, 1.0), math.radians(-12), math.radians(6))
    k = 4
    f = _render(scene, cam, k)
    for tid in (25, 26):
        sq = T.square(tid, T.TAG_OUTER, 0.002)
        center = sq.mean(axis=0)
        for i in range(4):
            a, b = sq[i], sq[(i + 1) % 4]
            for s in np.linspace(0.05, 0.95, 25):
                p = a + s * (b - a)
                inward = (center - p) / np.linalg.norm(center - p)
                for d, inside in ((0.006, True), (-0.006, False)):  # about 0.9 px either side
                    uv, _ = cam.project((p + d * inward)[None])
                    fu, fv = np.round(uv[0] * k + (k - 1) / 2).astype(int)
                    assert (f.obj[fv, fu] == ids.TAG + tid) == inside


def test_tag_pose_labels_reproject_to_the_corners():
    m = _match()
    scene, objects = build_scene(m, _look(m))
    intr = Intrinsics(320, 200, 250.0, 250.0, 159.5, 99.5, LENS)
    cam = Camera.looking(intr, (2.4, 3.6, 0.8), math.radians(10), math.radians(10))
    tags = [o for o in annotate(_render(scene, cam), cam, objects) if o["category"] == "apriltag" and o["bbox"]]
    assert len(tags) >= 2
    for o in tags:
        rot = np.array(o["pose_in_camera"]["rotation"])
        t = np.array(o["pose_in_camera"]["translation"])
        pc = T.tag_object_points() @ rot.T + t
        u, v = intr.distort(pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2])
        assert np.allclose(np.column_stack([u, v]), o["corners"], atol=1e-6)


def test_an_independent_detector_reads_the_rendered_tags():
    """The AprilTag C library (the one WPILib and PhotonVision use) must decode every clearly visible
    rendered tag with the labeled ID, and find its corners where the labels say. The library puts
    pixel centers at +0.5, so its corners are the labels + 0.5."""
    apriltags = pytest.importorskip("pupil_apriltags")
    det = apriltags.Detector(families="tag36h11")
    m = _match()
    scene, objects = build_scene(m, _look(m))
    intr = Intrinsics(320, 240, 330.0, 330.0, 159.5, 119.5, (-0.05, 0.01, 0.0, 0.0, 0.0))
    k = 2
    renderer = Renderer(intr.scaled(k))
    checked = set()
    # one view per kind of mount: HUB faces, TRENCH brackets, TOWER walls, OUTPOSTs; all from off to one
    # side. (From higher up the TOWER's low rung hides the top of its tags, and from far to the side
    # its uprights do: the real field has the same blind spots.)
    for tid, side, dist, dz in ((26, 25, 1.4, -0.3), (10, -30, 1.3, -0.4), (17, 20, 1.2, 0.3), (1, -25, 1.3, 0.2),
                                (31, 12, 1.2, -0.2), (15, -10, 1.3, -0.25), (29, 30, 1.1, 0.2), (13, -10, 1.3, 0.4)):
        c, n, _, _ = T.tag_frame(tid)
        a = math.atan2(n[1], n[0]) + math.radians(side)
        pos = c + dist * np.array([math.cos(a), math.sin(a), 0.0]) + (0.0, 0.0, dz)
        look_at = c - pos
        cam = Camera.looking(intr, pos, math.atan2(look_at[1], look_at[0]),
                             math.atan2(look_at[2], math.hypot(look_at[0], look_at[1])), math.radians(side / 5))
        frame = renderer.render(scene, Camera(intr.scaled(k), cam.position, cam.rotation))
        labels = {o["tag_id"]: o for o in annotate(frame, cam, objects, k) if o["category"] == "apriltag"}
        image = fx.apply(fx.downsample(frame.color, k), fx.Effects(), np.random.default_rng(0))
        gray = (image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)
        found = {d.tag_id: d for d in det.detect(gray)}
        for fid in found:
            assert fid in labels and labels[fid]["bbox"] is not None, fid  # no ghost or wrong IDs
        o = labels[tid]
        assert o["visibility"] > 0.999 and o["bbox"][2] > 25 and tid in found, tid
        err = np.abs(np.asarray(found[tid].corners) - (np.asarray(o["corners"]) + 0.5)).max()
        assert err < 0.8, (tid, err)
        checked.add(tid)
    assert len(checked) == 8


# ------------------------------------------------------------------------------ the scene
def test_scene_draws_every_loose_fuel_and_the_hub_lights():
    m = make_match(["strong"] * 3, ["strong"] * 3, seed=2)
    from rebuilt_sim.bots import ScriptedPolicy

    pol = ScriptedPolicy(m, seed=2)
    while m.t < 25.0:
        m.step(pol())
    look = _look(m)
    idx, pts, states = fuel_positions(m)
    census = m.fuel_census()
    assert len(idx) == census["ground"] + census["flight"] + census["chute"]
    assert (pts[:, 2] >= C.FUEL_RADIUS - 1e-9).all()
    on_bump = bump_height(pts[:, 0], pts[:, 1]) > 0
    assert (pts[on_bump, 2] > C.FUEL_RADIUS).all()
    scene, objects = build_scene(m, look)
    assert len(scene.spheres) == len(idx)
    assert {o.category for o in objects.values()} >= {"fuel", "robot", "apriltag", "hub", "trench", "bump"}
    for a in (0, 1):
        assert (hub_light(m, a, look) is not None) == m.hub_active(a)


def test_camera_effects():
    rng = np.random.default_rng(0)
    color = np.random.default_rng(1).uniform(0.0, 0.8, (40, 60, 3)).astype(np.float32)
    clean = fx.apply(color, fx.Effects(), rng)
    assert clean.dtype == np.uint8 and clean.shape == (40, 60, 3)
    assert np.array_equal(clean, fx.apply(color, fx.Effects(), np.random.default_rng(5)))  # no randomness when clean
    blurred = fx.apply(color, fx.Effects(motion_px=6.0, gamma=1.0), rng)
    assert abs(blurred.mean() - fx.apply(color, fx.Effects(gamma=1.0), rng).mean()) < 2.0
    mono = fx.apply(color, fx.Effects(grayscale=True), rng)
    assert (mono[..., 0] == mono[..., 1]).all() and (mono[..., 1] == mono[..., 2]).all()
    assert fx.downsample(np.ones((8, 6, 3)), 2).shape == (4, 3, 3)


# ------------------------------------------------------------------------------ the dataset
def _tiny(tmp_path, name, **kw):
    return DatasetConfig(name=name, out_dir=str(tmp_path), matches=3, frames_per_match=2, cameras_per_frame=2,
                         time_range=(1.0, 8.0), scale=0.12, supersample=1, hifi=False, masks=True, depth=True, **kw)


def test_dataset_generation_end_to_end(tmp_path):
    cfg = _tiny(tmp_path, "a")
    st = generate(cfg, progress=lambda s: None)
    root = tmp_path / "a"
    records = list(load_records(root))
    assert len(records) == 12 == sum(st["images"].values())
    for r in records:
        assert (root / r["image"]).exists() and (root / r["masks"]).exists() and (root / r["depth"]).exists()
        assert r["split"] == split_of(r["match"]["id"], cfg)
        cam = r["camera"]
        assert len(cam["pose"]["wpilib"]["quaternion"]) == 4 and len(cam["robot_to_camera_wpilib"]["quaternion"]) == 4
    assert validate(root, progress=lambda s: None)["problems"] == []
    for name in ("manifest.json", "index.csv", "dataset_card.md", "config.json"):
        assert (root / name).exists()
    assert "Tags are chiral" in (root / "dataset_card.md").read_text(encoding="utf-8")
    # exporting: the classes and filters are the caller's choice
    counts = export(root, ExportConfig("coco", "coco", ("fuel", "robot_blue", "robot_red", "apriltag")), progress=lambda s: None)
    for split in counts["images"]:
        coco = json.loads((root / "exports" / "coco" / "annotations" / f"{split}.json").read_text())
        assert len(coco["images"]) == counts["images"][split]
        assert all(len(a["bbox"]) == 4 and a["category_id"] in (1, 2, 3, 4) for a in coco["annotations"])
    export(root, ExportConfig("yolo", "yolo-pose", ("fuel", "apriltag"), min_box_side_px=2.0), progress=lambda s: None)
    out = root / "exports" / "yolo"
    assert "kpt_shape" in (out / "data.yaml").read_text()
    labels = list((out / "labels").rglob("*.txt"))
    assert len(labels) == len(list((out / "images").rglob("*.*"))) == 12
    for txt in labels:
        for line in txt.read_text().splitlines():
            cls, *vals = line.split()
            assert cls in ("0", "1") and len(vals) == 16 and all(0.0 <= float(v) <= 2.0 for v in vals)
    # the same config and seed give the same dataset; a second run resumes instead of redoing
    generate(_tiny(tmp_path, "b"), progress=lambda s: None)
    for shard in sorted((root / "records").rglob("*.gz")):
        assert shard.read_bytes() == (tmp_path / "b" / shard.relative_to(root)).read_bytes()
    for img in sorted((root / "images").rglob("*.*")):
        assert img.read_bytes() == (tmp_path / "b" / img.relative_to(root)).read_bytes()
    said = []
    generate(_tiny(tmp_path, "b"), progress=said.append)
    assert "3 matches already on disk" in said[0]
    with pytest.raises(ValueError):
        generate(_tiny(tmp_path, "b", seed=1), progress=lambda s: None)


def test_the_strategy_physics_is_untouched_by_rendering():
    """Rendering reads the match; it must never change it."""
    a, b = make_match(["mid"] * 3, ["mid"] * 3, seed=5), make_match(["mid"] * 3, ["mid"] * 3, seed=5)
    look = _look(a)
    intr = Intrinsics.from_fov(48, 32, 70.0)
    for _ in range(60):
        a.step([None] * 6)
        b.step([None] * 6)
        build_scene(a, look)
        _render(build_scene(a, look)[0], robot_camera(a, 0, intr, Mount()))
    assert np.array_equal(a.f_pos, b.f_pos) and [(r.x, r.y) for r in a.robots] == [(r.x, r.y) for r in b.robots]
    assert MatchConfig().hifi is None
