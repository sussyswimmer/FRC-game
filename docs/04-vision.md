# Step 4: the vision track (simulator side)

Robots see the field through cameras. Two kinds of vision matter in REBUILT:

- **Detection:** find FUEL, robots (and whose they are) in an image.
- **Localization:** find the AprilTags, read their IDs and corners, and work out where the robot is.

Both are learned from labeled images. Labeling real images by hand is slow, so step 4 makes
**synthetic** ones. It renders the simulator's matches in 3D from cameras on the robots, and writes
exact labels for every image: boxes, tag corners, poses, visibility.

| | What step 4 gives you |
|---|---|
| **Renderer** | A small CPU ray tracer in NumPy (`vision/render.py`): exact per-pixel geometry, lens distortion, anti-aliasing, simple lighting |
| **3D field** | The 2026 field from the official drawings: HUBs (funnel, cap lights, net), BUMPs, TRENCHes, TOWERs, DEPOTs, OUTPOSTs, driver stations, guardrails, carpet and tape, and all 32 AprilTags |
| **Match state** | Robots (bumpers in alliance color with team numbers) and every FUEL on the carpet, in the air and in the OUTPOST chutes, from a running match |
| **Cameras** | Real FRC cameras (Limelight 4/3G/3, Arducam OV9281/OV9782, OV2311), mounted like teams mount them |
| **Randomization** | Colors, lighting, lenses, mounts, exposure, motion blur, noise, JPEG |
| **Labels** | Visible and whole-object boxes, visibility, truncation, tag IDs, corners and poses, camera and robot poses, optional masks and depth |
| **Tools** | `make_dataset.py`, `export_dataset.py` (COCO/YOLO), `check_dataset.py`, `render_view.py` |

**What step 4 does not do: train anything.** Choosing classes, box style and filters, choosing a model,
and training and evaluating it are yours (HANDOFF, rule at the top). Section 9 lists those decisions
with their trade-offs.

## 1. Quick start

```powershell
uv sync --all-extras                                          # adds Pillow and pupil-apriltags (the "vision" extra)
.venv\Scripts\python.exe scripts\render_view.py --t 30        # one camera view, and the same with its labels drawn on
.venv\Scripts\python.exe scripts\render_view.py --overview    # a high view over the field
.venv\Scripts\python.exe scripts\make_dataset.py --name smoke --matches 2 --frames 4
.venv\Scripts\python.exe scripts\check_dataset.py datasets\smoke --sheet sheet.png --decode-tags 20
.venv\Scripts\python.exe scripts\export_dataset.py datasets\smoke --name yolo_fuel --format yolo --classes fuel --box visible --camera-robot keep
```

A real dataset uses all your cores. First ask for an estimate with `--dry-run`:

```powershell
.venv\Scripts\python.exe scripts\make_dataset.py --name v1 --matches 250 --frames 40 --workers 12 --dry-run
# about 10000 images, 0.36 h with 12 worker(s), 0.41 GB of disk, 6.0 GB of RAM
```

- **Disk:** about 45 KB per 640 x 400 image with its labels. `--png` stores lossless images at about
  four times that (twice for monochrome cameras, seven times for color). Your C: drive is tight, so `--out D:\datasets` puts the data on another drive.
- **RAM:** each worker needs about 0.5 GB. On 16 GB, 12 workers is safe.
- **Resuming:** run the same command again and it continues from where it stopped. `--overwrite`
  starts over: it deletes the old images, records and exports first.

## 2. How the renderer works

Every pixel of a camera image looks along one line of sight, a **ray**. The renderer finds the
nearest thing each ray hits, exactly:

- a sphere for FUEL, by solving the ray-sphere equation;
- a box, by the "slab" test (the ray enters and leaves each pair of parallel faces);
- a flat polygon, by where the ray meets its plane and whether that point is inside;
- a cylinder (TOWER rungs, pipes, LED stacks).

It keeps the nearest hit per pixel (the **z-buffer**) and which object it belongs to (the **id
buffer**). The id buffer is what makes the labels exact: a FUEL's visible box is simply the box
around the pixels that show it. The renderer also counts the pixels each object would cover with
nothing in front of it, and that gives how much of the object is hidden.

Testing every pixel against every shape would be slow. So each shape is first projected onto the
image, and only the pixels inside its screen window are tested. Shapes that pass behind the camera
(a guardrail along the field, say) are clipped at a near plane first. A test renders random scenes
both ways and checks that the images, ids and depths are identical.

- **Lens distortion is exact:** each pixel's ray comes from the camera's lens model. Edges bend like
  in a real wide-angle image, and labels bend with them. Nothing is warped after rendering.
- **Anti-aliasing:** rendering at k x k samples per output pixel (`--supersample 2`) and averaging
  smooths the jagged edges a real camera never shows.
- **Lighting:**
  - drawing records each pixel's surface color, direction (normal) and material;
  - one pass lights them all: ambient light, a sky term that brightens upward faces, one main light, a
    highlight on FUEL, the HUB's light bars glowing in alliance color while it counts, and soft
    shadows under FUEL and robots.
- **Speed:** about 1.35 s per million rendered pixels on one core. A 640 x 400 image at 2 x 2
  supersampling is about 1.1-1.5 s, plus the match simulation (about 12 s per high-fidelity match).

Why not a real 3D engine?

- **Game and film engines** need a GPU context, and this development container has none; the
  project would become untestable here.
- **Blender** is about 1 GB and needs Python 3.13.
- **Unity** needs 5-10 GB of disk that isn't free.
- **Isaac Sim** needs more RAM than 16 GB.

The NumPy renderer runs identically on Windows and Linux, needs no GPU, and is exact and
deterministic. If rendering ever becomes the bottleneck, the renderer can be swapped behind the same
`Scene`/`Camera`/labels interfaces.

## 3. The 3D field

Dimensions come from the official **2026 Field Dimension Drawings** (FE-2026 rev B, with the element
sheets GE-26000 OUTPOST, GE-26100 BUMP, GE-26200 TRENCH, GE-26300 HUB, GE-26500 TOWER, GE-26600 DEPOT,
GE-26900 FUEL), the Game Manual section 5 and the Field Manual. All numbers are in
`vision/field_model.py` as the drawings' inches.

| Element | Modeled |
|---|---|
| HUB | 47 in square body, 49.75 in tall, steel base. The black band carries the tags. Black side panels have the teal "REBUILT" graphic. The aluminum frame and X braces show behind the clear lower panels. There are four FUEL exits. The white diffuser cap glows in the alliance color while the HUB counts (pulsing in the last 3 s), with an alliance-colored roof. Then the clear hexagonal funnel frame (72 in rim, 41.9 in across the flats, **corners toward the alliance walls**) and the net on the neutral side (120.4 in tall) |
| BUMP | 15° peaked ridge, 6.51 in apex, 0.61 in lips, 73 x 44.4 in, alliance-colored with rows of bolt heads. FUEL resting on it sits on the ramp. Robots crossing it rise and tilt (and so do their cameras) |
| TRENCH | 50.34 in wide opening, arm underside at 22.25 in, trapezoid column. Tags sit back to back on a bracket at 35 in. The scoring-table side (low y) arm is fixed, overhangs the guardrail and rests on a leg. The audience side (high y) arm is hinged |
| TOWER | Open frame: floor plate, two alliance-colored uprights 32.25 in apart, three 1.66 in rungs at 27/45/63 in, and grey supports to the wall. The TOWER wall has the teal graphic and tags 31/32 (15/16 for red) |
| DEPOT | U of 3 in bars, 42 x 27 in, open toward the wall |
| OUTPOST | Aluminum frame of the clear wall, CORRAL and CHUTE openings, the black CHUTE door, and the sloped chute with its FUEL behind the wall |
| Walls | Driver stations (diamond plate, sponsor panel, clear upper panels, top rail), team number signs with LED stacks, guardrails with posts and yellow trip guards |
| Floor and arena | Gray carpet with dark flecks, CENTER LINE and ROBOT STARTING LINE tape; stands, crowd, banners and ceiling lamps in the backdrop |
| AprilTags | All 32 tags, 36h11, 6.5 in black square, on 10.5 in white panels, at the official layout poses. Printed on one side only: seen from behind, a tag is not drawn |

Estimated rather than measured (APPROX in the code): the TRENCH arm's section and the column's top,
the HUB cap and the funnel's bottom width, the net's shape, the HUB exits, and all colors. Colors are
randomized anyway.

**Clear polycarbonate is invisible** here: no tint and no reflections. It is the biggest missing
effect.

**AndyMark fields.** The Türkiye regionals run on AndyMark fields (16.518 x 8.043 m), where the tags
sit 1-3 cm away from the welded layout this project uses (WPILib also ships
`2026-rebuilt-andymark.json`). The render uses the welded layout.

## 4. Simulator vs drawings (a decision for you)

The drawings disagree with a few simulator constants that were estimated in step 2 from the broadcast
and the tag positions:

| Item | Simulator (`constants.py`, `field.py`, `ballistics.py`) | Official drawings | Off by |
|---|---|---|---|
| TOWER center y (blue) | 3.9616 m (between tags 31 and 32) | 3.7457 m (tag 31 is the centered tag) | 0.22 m |
| OUTPOST center y (blue) | 0.8819 m | 0.666 m (tag 29) | 0.22 m |
| DEPOT center y (blue) | 7.03 m | 5.965 m, next to the TOWER | 1.07 m |
| Climb positions | 0 and ±0.95 m from the TOWER center | the rungs only span ±0.60 m | past the rung ends |
| HUB hexagon | flats toward the alliance walls | corners toward the alliance walls | rotated 30° |
| TOWER collision | solid 45 x 49.25 in box | open frame robots can drive into | shape |

The render follows the drawings, because real cameras will see the real field, and leaves the
simulator alone, so its calibrated physics is unchanged. The cost: DEPOT FUEL starts about 1 m from the
drawn DEPOT, and climbing robots hang beside the drawn rungs. Rendered robots are never moved to hide
this, because that would break the pose labels.

Options:

- **(A)** Fix the simulator constants in their own small step, then re-run the tests and
  `calibrate.py` (and `--hifi`) and update docs 02 and 03. Recommended.
- **(B)** Keep things as they are and live with the offsets (they are in every dataset card).
- **(C)** Render the simulator's geometry instead of the drawings.

## 5. Robots, FUEL and the HUB lights

- **Robots** (`vision/robot_model.py`). The simulator knows a robot's footprint and height, so those are
  exact. The rest is random per match, so that a detector learns "robot" and not one robot:
  - bumpers in alliance color with the team number on all four sides (numbers also show on the
    driver-station signs);
  - a chassis, a hopper, a shooter (it turns with the turret on turret robots), uprights and an
    over-the-bumper intake, in random colors.

  Robots rise and tilt on BUMPs, and ride up when climbing.
- **FUEL** (`vision/scene.py`):
  - on the carpet or on a BUMP ramp;
  - in flight: at the height the high-fidelity ballistics computes (with the strategy physics, on a
    made-up arc, because that physics has no heights);
  - in the OUTPOST chutes, 5 across.

  FUEL held inside robots or being processed inside a HUB is not drawn. Each ball's color varies a
  little (wear).
- **HUB lights** follow the Game Manual (table 5-3): alliance color while the HUB counts, pulsing in
  the last 3 s before it turns off, dark otherwise.

## 6. Cameras

| Preset | Resolution | Horizontal FOV | Sensor | Used for |
|---|---|---|---|---|
| `limelight4`, `limelight3g` | 1280 x 800 | 82° | OV9281 mono, global shutter | AprilTags, and Limelight's mono FUEL models |
| `ov9281` | 1280 x 800 | 70° | Arducam mono, global shutter (PhotonVision's AprilTag camera) | AprilTags |
| `ov9782` | 1280 x 800 | 70° | Arducam color, global shutter (PhotonVision's object camera) | FUEL and robots |
| `limelight3` | 640 x 480 | 62.5° | OV5647 color, rolling shutter (the skew isn't modeled) | older robots |
| `ov2311` | 1600 x 1300 | about 78° | mono, global shutter (ThriftyCam class) | AprilTags |

- **Lens model:** a pinhole plus OpenCV's distortion coefficients (`k1 k2 p1 p2 k3`), the same model
  Limelight and PhotonVision calibrate.
  - Each camera gets a random lens per match: field of view ±5%, principal point off-center by about
    1%, and random distortion.
  - Some random distortion values describe impossible lenses: the image "folds over" near the corners,
    so some pixels have no ray and others have two. The sampler redraws those, and the renderer
    refuses them (`Intrinsics.lens_error`).
  - Rays come from undistorting each pixel by Newton's method, exact to about 1e-12 px.
- **Mounts** follow what teams run. AprilTag cameras sit low (0.17-0.40 m) and tilt **up** 10-35°.
  Game-piece cameras sit high (0.35-0.70 m) and tilt **down** 8-30°. A camera faces forward (most
  often), 25° or 70° to a side, or backward. It sits on the frame's edge on that side, with a couple
  of degrees of mounting error (recorded as the truth).
- **Motion blur is physical:** a robot turning at ω rad/s during an exposure of t seconds smears the
  image by about ω · t · f pixels (f = focal length in pixels). Monochrome cameras get short exposures
  (1-5 ms, as teams run their tag cameras), color cameras longer ones (3-10 ms).
- **Sign conventions:**
  - `Mount.pitch_up` is positive for a camera tilted up. WPILib's `Rotation3d` pitch is positive
    nose-down, so the WPILib mount is `Transform3d(x, y, z, Rotation3d(roll, -pitch_up, yaw))`.
    Every record carries it ready-made as `robot_to_camera_wpilib`.

## 7. Domain randomization

A model trained on perfect renders learns the renderer's quirks (flat colors, razor edges, no noise)
and fails on real images: the **sim-to-real gap**. **Domain randomization** narrows it: vary
everything that shouldn't matter, so widely that the only thing left in common is what the model should
learn. All ranges are fields of `RandomizationConfig` (`vision/randomize.py`). Override any of them
with a JSON file:

```powershell
'{"k1": [-0.1, 0.05], "apriltag_role": 0.8}' | Out-File ranges.json -Encoding ascii
.venv\Scripts\python.exe scripts\make_dataset.py --randomization ranges.json ...
```

| When | What varies |
|---|---|
| Per match | robot mix (Day 1 İstanbul mix 70%, stronger later-season mix 30%); carpet, paint, alliance colors, tag paper and ink; FUEL color; lighting (ambient, sky, main light direction and strength, color temperature); HUB light brightness; the crowd and ceiling backdrop; each robot's look and team number |
| Per camera (for the match) | preset; lens (FOV, principal point, distortion); role; mount position, height, yaw, pitch, roll |
| Per frame | exposure time, gain (more gain = more noise), overall brightness, white balance, vignetting, defocus, motion blur from the robot's own motion, shot and read noise, JPEG quality; monochrome sensors mix red, green and blue with random, red-leaning weights (no infrared filter) |

Every value drawn is stored in the image's record (`camera`, `render.light`, `render.effects`, and the
per-match look in `render.appearance`), so you can study or filter by it. `--effects 0 --no-randomize` gives clean, default renders.

Each purpose has its own random stream (`dataset.streams`). Changing one setting, like frames per
match, doesn't reshuffle the rest of the match.

## 8. The dataset: files and labels

```
datasets/<name>/
  images/<split>/m00012_f003_c0.jpg      match 12, moment 3, camera 0
  records/<split>/m00012.jsonl.gz        one JSON line per image: every label (gzipped, to save disk)
  index.csv                              one row per image: time, camera, counts (open it in a spreadsheet)
  manifest.json                          config, git commit, versions, checksums: how to make it again
  dataset_card.md                        what's in it and how to read it
  config.json
  masks/, depth/                         only with --masks / --depth (16-bit PNGs; each pixel's most common
                                         object id among its k x k samples, and that sample's depth)
  exports/<export name>/                 what export_dataset.py writes
```

- Read the records in Python with `rebuilt_sim.vision.dataset.load_records("datasets/v1")`.
- **Splits are by match** (80/10/10 by default, from a hash of the seed and the match; set them with
  `make_dataset.py --splits train=0.7 val=0.15 test=0.15`). They are fixed when the dataset is
  generated. Frames from one match are near-copies of each other, so splitting frames at random
  would put nearly the same image in train and in validation, and the validation score would lie.
  This is called **data leakage**.

**A record** holds:

- `image`, `split`, `width`, `height`, `channels`.
- `match`: id, `t`, period, which HUBs count, the score, the robot tiers.
- `camera`: preset, role, `K` (3 x 3 intrinsics), `dist`, the mount (readable and as WPILib
  `Transform3d`), and `pose`. The pose comes three ways: plain numbers, a WPILib `Pose3d`, and OpenCV
  `R`/`t`/`rvec`, which map field points into the camera.
- `robot`: the camera's robot. Its true 3D pose (with BUMP tilt), its own pose estimate (high
  fidelity), and its speeds.
- `render`: the light and every camera effect.
- `objects`: every object that covers at least one pixel, **including completely hidden ones**:

| Field | Meaning |
|---|---|
| `category` | fuel, robot, apriltag, hub, tower, trench, bump, outpost, depot |
| `bbox` | `[x, y, w, h]` around the **visible** pixels (None if fully hidden) |
| `bbox_amodal` | around everything it would cover with nothing in front of it ("amodal" = including the hidden parts), clipped to the image |
| `visibility` | visible pixels / covered pixels: 1 in full view, 0 hidden |
| `truncation` | the fraction outside the image (cut off by the edge, or behind the camera). FUEL: the share of its round silhouette; the rest: the share of points spread through its 3D box |
| `distance` | meters from the camera |
| FUEL | `state` (ground, flight, chute), `center` in the field |
| robot | `alliance`, `robot_index`, `team_number`, `pose`, `climbing`, `fuel_held`, `camera_robot` (it carries this camera: often its own bumper or intake at the image's edge) |
| structures | `alliance`; `side` (low_y / high_y) for BUMPs and TRENCHes. A structure's own tags count as part of it for its visibility |
| apriltag | `tag_id`, `corners` (pixels), `corners_visible`, `corners_field`, `min_edge_px`, `view_angle_deg` (0 = head-on), `pose_in_camera` (`rotation`, `translation`, `rvec`) |

**Pixel conventions** (they bite everyone once):

- **Boxes** use pixel *edges*, like COCO: the image spans `[0, width] x [0, height]`.
- **Points** (tag corners) use OpenCV's pixel *centers*: the top-left pixel's center is (0, 0), so
  edge = center + 0.5. That is the convention of `cv2.solvePnP` and of the `K` matrix here.
- **The AprilTag C library** (inside WPILib, PhotonVision and pupil-apriltags) reports corners
  **+0.5 px** in both directions: its pixel centers are at +0.5. It lists them in the same order.
  Measured on clean renders of all 32 tags: +0.51 / +0.47 px head-on. At 30° off-axis its corners
  also drift about 0.3 px sideways. That is the detector's own bias: the rendered tags match their
  labels to 0.01 px at any angle (a test checks this).
- **Corner order:** counter-clockwise from the tag's printed bottom-left (bottom-left, bottom-right,
  top-right, top-left), like WPILib. **OpenCV's ArUco module** returns AprilTag corners rotated 180°:
  indices `[1, 0, 3, 2]` of these. Feeding ArUco's order straight into a pose solver flips the tag
  180° about its face.
- **Tag pose:** `pose_in_camera` is what `cv2.solvePnP(tags.tag_object_points(), corners, K, dist)`
  returns, with the tag frame x right, y down, z into the tag. A test checks that it reprojects onto
  the labeled corners exactly.

## 9. Exporting for training: your decisions

`export_dataset.py` makes COCO or YOLO files from the records. **Every choice there is a training
decision**, and none is made for you: `--format`, `--classes`, `--box` and `--camera-robot` are
required, and the filters default to keeping everything. Some you'll meet:

| Decision | Options and trade-offs |
|---|---|
| Classes | `fuel`, `robot`, `robot_blue`/`robot_red`, `apriltag`, `hub_blue`/`hub_red`, field elements. More classes means harder learning and more confusion. One team found that mixing FUEL and bumper datasets hurt FUEL accuracy. Limelight's advice is "as few classes as possible" |
| Box style | `--box visible` (only what's seen; Limelight's advice) or `--box amodal` (the whole object; one team's all-synthetic FUEL model used this and did very well). They teach different things with piles of FUEL |
| Filters | `--min-visible-px`, `--min-box-side`, `--min-visibility`, `--max-truncation`. A 2-pixel sliver of FUEL is a label a model can't possibly learn from, but cutting too much hides the hard cases it must handle |
| FUEL states | ground, flight, chute |
| The camera's own robot | Its bumper or intake often shows at the edge of its own images. Keep it as a "robot" or leave it out (`--camera-robot keep` or `drop`) |
| Flips | **Mirroring corrupts AprilTags** (a mirrored tag is another, invalid code, and its corners swap). Ultralytics' default `fliplr=0.5` does this |
| Letterboxing | PhotonVision and most YOLO pipelines squeeze frames into 640 x 640 with black bars. Small far FUEL gets 2x smaller in that step |
| Mono vs color | Most tag cameras (and Limelight's 2026 FUEL models) are monochrome. You can train on the mix, or on one kind |
| Size and splits | How many matches; the split fractions (`make_dataset.py --splits`, chosen when generating) |
| A real test set | The only honest measure of the sim-to-real gap is a few hundred **real, hand-labeled** images, never trained on. The 171 full-resolution İstanbul broadcast frames from step 1 could be a start |

YOLO exports hard-link the images (no extra disk), write `labels/<split>/*.txt` and a `data.yaml` that
only describes paths and class names. `yolo-pose` adds the 4 tag corners as keypoints, normalized
like the boxes (pixel edges). COCO exports carry visibility, truncation and, for tags, the corners as
keypoints in pixel centers, the way detectron2 and mmpose read COCO keypoints. An export warns when a
split has no images: small datasets (a few matches) may have no validation match at all.

## 10. Checking the data

```powershell
.venv\Scripts\python.exe scripts\check_dataset.py datasets\v1 --stats
.venv\Scripts\python.exe scripts\check_dataset.py datasets\v1 --decode-tags 100
.venv\Scripts\python.exe scripts\check_dataset.py datasets\v1 --sheet sheet.png --count 24
```

- `--stats`: counts, box sizes and how often things are hidden.
- `--decode-tags`: runs the real AprilTag detector on clearly visible tags and compares IDs and corners
  with the labels.
- `--sheet`: draws the labels on a grid of random images. Look at it before training on anything.

## 11. The GPU, for when you train

Training a detector wants the RTX 3070. The project has CPU PyTorch because of disk space:

1. Free about 8 GB on C: (the CUDA build of PyTorch is about 5 GB, plus room for datasets and runs).
2. In `pyproject.toml`, change the `pytorch-cpu` index URL to `https://download.pytorch.org/whl/cu126`.
3. Run `uv sync --all-extras`.
4. Check it: `python -c "import torch; print(torch.cuda.is_available())"`.

The training packages (a detector library, if you pick one) are your choice too; nothing is installed
for it.

## 12. What the tests prove (`tests/test_vision.py`, 23 tests)

- **An independent detector reads the rendered tags.** For 8 tags on every kind of mount, the AprilTag
  C library (pupil-apriltags) decodes the labeled ID and finds the corners within 0.8 px of the labels
  (after its +0.5 px convention), through a distorting lens. This proves the codes, the bit order, the
  orientation, the corner order and the geometry.
- **Every tag is fully visible head-on** from 1 m: nothing of the field hides any tag. The black
  squares are 0.1651 m and face the right way.
- **Rendering matches the labels:** just inside a tag's printed edge the id buffer shows the tag, just
  outside it doesn't, through a distorting lens. Seen head-on and at up to 50°, a tag covers exactly
  its projected quadrilateral: its centroid lands within 0.02 px and its area within 0.2% (a half-pixel
  slip would be obvious). A sphere's rendered extent matches its exact silhouette.
- **The fast renderer equals its brute-force self** on random scenes, including shapes behind the camera.
- **Conventions:** OpenCV projection directions, WPILib pitch sign and quaternions, OpenCV extrinsics,
  pose round trips.
- **The lens model:** undistort and distort invert each other to 1e-6 px, folding lenses are refused,
  and the sampler redraws them.
- **Occlusion and visibility:** hidden, half-hidden and off-image objects get the right labels. Tags
  are invisible from behind. FUEL truncation is exact at the image edge.
- **The dataset:**
  - end to end: files, splits, validation, COCO/YOLO/YOLO-pose exports;
  - byte-identical reruns;
  - resuming;
  - refusing to mix configs.
- **Rendering never changes the match.**
- **Regressions** for the bugs an independent review found: truncation of objects reaching behind the
  camera, structures hidden by their own tags, a ball behind the lens, a clean `--overwrite`.

## 13. Limits and possible follow-ups

- **Invisible polycarbonate.** A tinted, reflective "glass" overlay would add the reflections real
  cameras see on the HUB, the driver stations and the guardrails.
- **Shadows.** Only soft blobs under FUEL and robots.
- **Details not modeled:** the FUEL logos, the tags' "ID n" text, rolling-shutter skew, and
  distractors outside the field (people, carts).
- **Robots:** they are boxes. Held FUEL in open hoppers is not drawn.
- **Speed:** batching many small shapes into one NumPy pass could make rendering 2-3x faster, if
  generation time ever matters.
- **Tag layouts:** an option to render the AndyMark tag layout.
- **The simulator conflicts** (section 4).
