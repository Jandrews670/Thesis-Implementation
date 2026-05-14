from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Color = Tuple[int, int, int]


def write_line_plot_png(path: Path, values: Sequence[float], width: int = 640, height: int = 360) -> None:
    canvas = _blank(width, height)
    _draw_axes(canvas)
    if values:
        ys = np.asarray(values, dtype=float)
        xs = np.arange(len(ys), dtype=float)
        _draw_polyline(canvas, xs, ys, (30, 90, 180))
    _write_png(path, canvas)


def write_histogram_png(path: Path, values: Sequence[float], width: int = 640, height: int = 360) -> None:
    canvas = _blank(width, height)
    _draw_axes(canvas)
    if values:
        counts, bin_edges = np.histogram(np.asarray(values, dtype=float), bins=min(20, max(5, len(values))))
        xs = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        if counts.max() > 0:
            _draw_bars(canvas, xs, counts.astype(float), (30, 140, 90))
    _write_png(path, canvas)


def _blank(width: int, height: int) -> List[List[Color]]:
    return [[(255, 255, 255) for _ in range(width)] for _ in range(height)]


def _plot_bounds(canvas: List[List[Color]]) -> Tuple[int, int, int, int]:
    height = len(canvas)
    width = len(canvas[0])
    return 48, width - 24, 24, height - 42


def _draw_axes(canvas: List[List[Color]]) -> None:
    left, right, top, bottom = _plot_bounds(canvas)
    for x in range(left, right + 1):
        _set(canvas, x, bottom, (80, 80, 80))
    for y in range(top, bottom + 1):
        _set(canvas, left, y, (80, 80, 80))


def _scale_points(
    canvas: List[List[Color]],
    xs: np.ndarray,
    ys: np.ndarray,
) -> List[Tuple[int, int]]:
    left, right, top, bottom = _plot_bounds(canvas)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0
    points = []
    for x_value, y_value in zip(xs, ys):
        x = int(left + (float(x_value) - x_min) / (x_max - x_min) * (right - left))
        y = int(bottom - (float(y_value) - y_min) / (y_max - y_min) * (bottom - top))
        points.append((x, y))
    return points


def _draw_polyline(
    canvas: List[List[Color]],
    xs: np.ndarray,
    ys: np.ndarray,
    color: Color,
) -> None:
    points = _scale_points(canvas, xs, ys)
    for start, end in zip(points[:-1], points[1:]):
        _draw_line(canvas, start[0], start[1], end[0], end[1], color)
    for x, y in points:
        _draw_square(canvas, x, y, color)


def _draw_bars(
    canvas: List[List[Color]],
    xs: np.ndarray,
    ys: np.ndarray,
    color: Color,
) -> None:
    points = _scale_points(canvas, xs, ys)
    left, right, _top, bottom = _plot_bounds(canvas)
    bar_width = max(2, int((right - left) / max(1, len(points)) * 0.7))
    for x, y in points:
        for px in range(x - bar_width // 2, x + bar_width // 2 + 1):
            for py in range(y, bottom):
                _set(canvas, px, py, color)


def _draw_line(canvas: List[List[Color]], x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        _set(canvas, x, y, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _draw_square(canvas: List[List[Color]], x: int, y: int, color: Color) -> None:
    for px in range(x - 2, x + 3):
        for py in range(y - 2, y + 3):
            _set(canvas, px, py, color)


def _set(canvas: List[List[Color]], x: int, y: int, color: Color) -> None:
    if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
        canvas[y][x] = color


def _write_png(path: Path, canvas: List[List[Color]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(canvas)
    width = len(canvas[0])
    raw_rows = []
    for row in canvas:
        raw_rows.append(b"\x00" + bytes(channel for pixel in row for channel in pixel))
    raw = b"".join(raw_rows)
    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        _chunk(handle, b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0))
        _chunk(handle, b"IDAT", zlib.compress(raw, 9))
        _chunk(handle, b"IEND", b"")


def _chunk(handle, chunk_type: bytes, data: bytes) -> None:
    handle.write(struct.pack("!I", len(data)))
    handle.write(chunk_type)
    handle.write(data)
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    handle.write(struct.pack("!I", crc & 0xFFFFFFFF))
