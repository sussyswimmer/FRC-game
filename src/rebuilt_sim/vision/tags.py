"""The field's AprilTags as they are printed: the 36h11 pattern of each ID, its size, and where its
corners are in the field.

An AprilTag is a 10 x 10 grid of square cells: a white outer ring, a black ring inside it, and a
6 x 6 block of black or white data cells that encodes the ID. The 2026 field uses the ``tag36h11``
family (36 data bits, any two codes differ in at least 11 bits) with IDs 1-32. The printed black
square is 6.5 in (0.1651 m) on a side: that is the "tag size" pose solvers such as OpenCV's
``solvePnP``, WPILib and PhotonVision expect. With the white ring the printed square is 8.125 in.

The codes below are IDs 1-32 of ``tag36h11.c`` from AprilRobotics' apriltag library (Copyright
(C) 2013-2016, The Regents of The University of Michigan, BSD 2-Clause license). The rendered
patterns were checked against the official apriltag-imgs PNGs and the 2026 FRC AprilTag PDF.

Corner order follows the AprilTag library and WPILib: counter-clockwise as seen facing the tag,
starting at its printed bottom-left corner (bottom-left, bottom-right, top-right, top-left).
OpenCV's ArUco module lists the same corners clockwise from the top-left instead.
"""

from __future__ import annotations

import math

import numpy as np

from ..apriltags import TAGS

TAG_SIZE = 0.1651  # m, edge of the black square
TAG_OUTER = 0.206375  # m, edge of the printed square including the white ring (10 cells)
CELL = TAG_OUTER / 10
PANEL = 0.2667  # m, edge of the white panel the tag is printed on (APPROX: the backing's exact size varies)

# tag36h11 codes for IDs 1-32 (bit i = (code >> (35 - i)) & 1, 1 = white)
_CODES = (
    0xdda664ca7, 0xdc4a1c821, 0xe17b470e9, 0xef91d01b1, 0xf429cdd73, 0x05da29225, 0x1106cba43, 0x223bed79d,
    0x21f51213c, 0x33eb19ca6, 0x3f76eb0f8, 0x469a97414, 0x45dcfe0b0, 0x4a6465f72, 0x51801db96, 0x5eb946b4e,
    0x68a7cc2ec, 0x6f0ba2652, 0x78765559d, 0x87b83d129, 0x86cc4a5c5, 0x8b64df90f, 0x9c577b611, 0xa3810f2f5,
    0xaf4d75b83, 0xb59a03fef, 0xbb1096f85, 0xd1b92fc76, 0xd0dd509d2, 0xe2cfda160, 0x2ff497c63, 0x47240671b,
)
# column and row of data bit i inside the black ring (add 1 for the white ring)
_BIT_X = (1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4, 1, 1, 1, 1, 1, 2, 2, 2, 3)
_BIT_Y = (1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4)

IDS = tuple(range(1, len(_CODES) + 1))
_POSE = {tid: (np.array([x, y, z]), math.radians(yaw)) for tid, x, y, z, yaw in TAGS}


def tag_cells(tag_id: int) -> np.ndarray:
    """The 10 x 10 pattern (1 = white, 0 = black), row 0 at the top as seen facing the tag."""
    code = _CODES[tag_id - 1]
    im = np.zeros((10, 10), dtype=np.float32)
    im[0, :] = im[-1, :] = im[:, 0] = im[:, -1] = 1.0
    for i in range(36):
        if (code >> (35 - i)) & 1:
            im[_BIT_Y[i] + 1, _BIT_X[i] + 1] = 1.0
    return im


def tag_texture(tag_id: int, white: float = 0.85, black: float = 0.04) -> np.ndarray:
    """(10, 10, 3) albedo texture: matte paper white and ink black."""
    cells = tag_cells(tag_id)
    return np.repeat((black + (white - black) * cells)[..., None], 3, axis=-1)


def tag_frame(tag_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center, outward normal, and the viewer's right and up directions (field frame) of a tag.
    WPILib's tag pose has +x pointing out of the printed face."""
    center, yaw = _POSE[tag_id]
    normal = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])  # to the right of someone facing the tag
    return center, normal, right, np.array([0.0, 0.0, 1.0])


def square(tag_id: int, edge: float, offset: float = 0.0) -> np.ndarray:
    """(4, 3) corners of a square of ``edge`` centered on a tag, ``offset`` meters out from its
    face, ordered top-left, top-right, bottom-right, bottom-left as seen facing it."""
    c, n, r, u = tag_frame(tag_id)
    h = edge / 2
    c = c + offset * n
    return np.array([c - h * r + h * u, c + h * r + h * u, c + h * r - h * u, c - h * r - h * u])


def tag_corners(tag_id: int) -> np.ndarray:
    """(4, 3) field-frame corners of the black square in AprilTag/WPILib order: bottom-left,
    bottom-right, top-right, top-left as printed."""
    tl, tr, br, bl = square(tag_id, TAG_SIZE)
    return np.array([bl, br, tr, tl])


def tag_object_points() -> np.ndarray:
    """(4, 3) corners of the black square in the tag's own frame for ``cv2.solvePnP`` (x right,
    y down, z into the tag, so the camera looks along +z), in ``tag_corners`` order."""
    h = TAG_SIZE / 2
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
