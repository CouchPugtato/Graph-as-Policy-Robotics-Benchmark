"""Bounded frame recording using existing simulator camera observations."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path, pixels):
    height, width, channels = pixels.shape
    if channels != 3 or pixels.dtype != np.uint8 or width * height > 640 * 480:
        raise ValueError("recording requires bounded rgb byte images")

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))

    rows = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    with Path(path).open("xb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        handle.write(chunk(b"IDAT", zlib.compress(rows, level=1)))
        handle.write(chunk(b"IEND", b""))


class FrameRecorder:
    def __init__(self, directory, camera="agentview", fps=20, max_frames=1001):
        if type(fps) is not int or not 1 <= fps <= 60:
            raise ValueError("recording rate must be an integer from one to sixty")
        if type(max_frames) is not int or not 1 <= max_frames <= 1001:
            raise ValueError("recording frame budget must be from one to one thousand and one")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.camera, self.fps, self.max_frames = camera, fps, max_frames
        self.frames = []
        self.closed = False

    def record(self, observation, step):
        if self.closed or len(self.frames) >= self.max_frames:
            raise RuntimeError("recording frame budget exhausted or recorder closed")
        if self.frames and step <= self.frames[-1]["step"]:
            raise ValueError("recording steps must increase")
        if type(step) is not int or step < 0:
            raise ValueError("recording steps must be nonnegative integers")
        # mujoco camera observations have the opposite vertical image origin
        pixels = np.asarray(observation[f"{self.camera}_image"])[::-1]
        filename = f"frame-{len(self.frames):05d}.png"
        write_png(self.directory / filename, pixels)
        self.frames.append({"file": filename, "step": step, "time_s": step / self.fps})

    def close(self):
        if not self.closed:
            self.closed = True
            with (self.directory / "frames.json").open("x", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, "fps": self.fps,
                           "camera": self.camera, "frames": self.frames}, handle, indent=2)
