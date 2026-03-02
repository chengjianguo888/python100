#!/usr/bin/env python3
"""Tests for LaserCornerExtractor corner detection algorithms."""

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ── inline copy of LaserCornerExtractor (avoids tkinter dependency) ──────
class LaserCornerExtractor:
    """Algorithms for 2-D laser-scan corner extraction."""

    @staticmethod
    def extract(points, method="curvature", sigma=1.5, threshold=0.05,
                min_distance=5):
        if len(points) < 10:
            return np.empty((0, 2))
        x, y = points[:, 0], points[:, 1]
        if method == "curvature":
            idx = LaserCornerExtractor._curvature(x, y, sigma, threshold, min_distance)
        elif method == "harris":
            idx = LaserCornerExtractor._harris(x, y, sigma, threshold, min_distance)
        elif method == "bearing_angle":
            idx = LaserCornerExtractor._bearing_angle(x, y, sigma, threshold, min_distance)
        elif method == "line_segment":
            idx = LaserCornerExtractor._line_segment(x, y, threshold, min_distance)
        else:
            idx = np.array([], dtype=int)
        return points[idx]

    @staticmethod
    def _curvature(x, y, sigma, threshold, min_dist):
        xs = gaussian_filter1d(x.astype(float), sigma)
        ys = gaussian_filter1d(y.astype(float), sigma)
        dx = np.gradient(xs);  dy = np.gradient(ys)
        ddx = np.gradient(dx); ddy = np.gradient(dy)
        denom = (dx**2 + dy**2)**1.5
        with np.errstate(divide='ignore', invalid='ignore'):
            kappa = np.abs(dx*ddy - dy*ddx) / np.where(denom < 1e-10, 1e-10, denom)
        kappa = gaussian_filter1d(kappa, sigma)
        peak_thresh = threshold * kappa.max() if kappa.max() > 0 else threshold
        return LaserCornerExtractor._find_peaks(kappa, peak_thresh, min_dist)

    @staticmethod
    def _harris(x, y, sigma, threshold, min_dist):
        xs = gaussian_filter1d(x.astype(float), sigma)
        ys = gaussian_filter1d(y.astype(float), sigma)
        dx = np.gradient(xs); dy = np.gradient(ys)
        Ixx = gaussian_filter1d(dx**2, sigma)
        Iyy = gaussian_filter1d(dy**2, sigma)
        Ixy = gaussian_filter1d(dx * dy, sigma)
        k = 0.04
        det = Ixx*Iyy - Ixy**2
        trace = Ixx + Iyy
        R = det - k * trace**2
        thresh = threshold * R.max() if R.max() > 0 else threshold
        return LaserCornerExtractor._find_peaks(np.maximum(R, 0), thresh, min_dist)

    @staticmethod
    def _bearing_angle(x, y, sigma, threshold, min_dist):
        xs = gaussian_filter1d(x.astype(float), sigma)
        ys = gaussian_filter1d(y.astype(float), sigma)
        deviation = np.zeros(len(xs))
        for i in range(1, len(xs) - 1):
            v1 = np.array([xs[i-1]-xs[i], ys[i-1]-ys[i]])
            v2 = np.array([xs[i+1]-xs[i], ys[i+1]-ys[i]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.clip(np.dot(v1, v2)/(n1*n2), -1, 1)
                deviation[i] = 180.0 - np.degrees(np.arccos(cos_a))
        deviation = gaussian_filter1d(deviation, sigma)
        thresh = threshold * 180.0
        return LaserCornerExtractor._find_peaks(deviation, thresh, min_dist)

    @staticmethod
    def _line_segment(x, y, threshold, min_dist):
        n = len(x)
        if n < 6:
            return np.array([], dtype=int)
        extent = max(x.max() - x.min(), y.max() - y.min())
        tol = threshold * extent if extent > 0 else threshold
        segments = LaserCornerExtractor._recursive_split(x, y, 0, n-1, tol)
        seg_endpoints = set()
        for s, e in segments:
            seg_endpoints.add(s)
            seg_endpoints.add(e)
        boundaries = sorted(seg_endpoints)
        return np.array([b for b in boundaries[1:-1]], dtype=int)

    @staticmethod
    def _recursive_split(x, y, start, end, tol):
        if end - start < 3:
            return [(start, end)]
        dx, dy = x[end]-x[start], y[end]-y[start]
        L = np.hypot(dx, dy)
        if L < 1e-10:
            dists = np.hypot(x[start:end+1]-x[start], y[start:end+1]-y[start])
        else:
            dists = np.abs(dy*x[start:end+1] - dx*y[start:end+1]
                           + x[end]*y[start] - y[end]*x[start]) / L
        max_idx = np.argmax(dists) + start
        if dists[max_idx - start] > tol:
            left = LaserCornerExtractor._recursive_split(x, y, start, max_idx, tol)
            right = LaserCornerExtractor._recursive_split(x, y, max_idx, end, tol)
            return left + right
        return [(start, end)]

    @staticmethod
    def _find_peaks(signal, threshold, min_dist):
        indices = []
        i = 0
        n = len(signal)
        while i < n:
            if signal[i] >= threshold:
                j = i
                while j < n and signal[j] >= threshold:
                    j += 1
                peak = i + np.argmax(signal[i:j])
                if not indices or (peak - indices[-1]) >= min_dist:
                    indices.append(peak)
                i = j
            else:
                i += 1
        return np.array(indices, dtype=int)


# ── helpers ──────────────────────────────────────────────────────────────
METHODS = ["curvature", "harris", "bearing_angle", "line_segment"]


def _make_sample_room():
    """Reproduce the synthetic room scan from MainApp._generate_sample."""
    rng = np.random.default_rng(42)
    angles = np.linspace(-np.pi * 0.85, np.pi * 0.85, 720)
    ranges = np.zeros(len(angles))
    for i, a in enumerate(angles):
        ca, sa = np.cos(a), np.sin(a)
        tx = (4.0 / abs(ca)) if abs(ca) > 1e-6 else 1e9
        ty = (3.0 / abs(sa)) if abs(sa) > 1e-6 else 1e9
        ranges[i] = min(tx, ty)
    for i, a in enumerate(angles):
        ca, sa = np.cos(a), np.sin(a)
        B = -2 * (ca * 2.5 + sa * 1.0)
        C = 2.5**2 + 1.0**2 - 0.09
        disc = B**2 - 4 * C
        if disc >= 0:
            t = (-B - np.sqrt(disc)) / 2.0
            if 0 < t < ranges[i]:
                ranges[i] = t
    ranges += rng.normal(0, 0.015, len(ranges))
    ranges = np.clip(ranges, 0.1, 20)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.column_stack([x, y])


def _has_corner_near(corners, target_x, target_y, max_dist=0.5):
    """Check if any detected corner is within max_dist of a target."""
    if len(corners) == 0:
        return False
    dists = np.sqrt((corners[:, 0] - target_x)**2 +
                    (corners[:, 1] - target_y)**2)
    return float(dists.min()) < max_dist


# ── tests ────────────────────────────────────────────────────────────────
def test_few_points_returns_empty():
    pts = np.random.randn(5, 2)
    for m in METHODS:
        r = LaserCornerExtractor.extract(pts, method=m)
        assert r.shape == (0, 2), f"{m} should return empty for < 10 points"
    print("PASS: test_few_points_returns_empty")


def test_straight_line_no_corners():
    t = np.linspace(0, 10, 100)
    pts = np.column_stack([t, t * 2.0])
    for m in METHODS:
        r = LaserCornerExtractor.extract(pts, method=m)
        assert len(r) == 0, f"{m} found {len(r)} corners on a straight line"
    print("PASS: test_straight_line_no_corners")


def test_l_shape_single_corner():
    t1 = np.linspace(0, 5, 50)
    pts1 = np.column_stack([t1, np.zeros(50)])
    t2 = np.linspace(0, 5, 50)
    pts2 = np.column_stack([np.full(50, 5.0), t2])
    pts = np.vstack([pts1, pts2])
    for m in METHODS:
        r = LaserCornerExtractor.extract(pts, method=m)
        assert len(r) >= 1, f"{m} found no corners on L-shape"
        assert _has_corner_near(r, 5.0, 0.0), \
            f"{m} missed corner at (5, 0)"
    print("PASS: test_l_shape_single_corner")


def test_sample_room_finds_all_corners():
    pts = _make_sample_room()
    expected = [(-4, -3), (4, -3), (4, 3), (-4, 3)]
    for m in METHODS:
        r = LaserCornerExtractor.extract(pts, method=m)
        assert len(r) >= 4, f"{m} found only {len(r)} corners (expected >=4)"
        for ex, ey in expected:
            assert _has_corner_near(r, ex, ey), \
                f"{m} missed corner near ({ex},{ey})"
    print("PASS: test_sample_room_finds_all_corners")


def test_bearing_angle_not_degenerate():
    """Bearing-angle must detect more than 1 corner on realistic data."""
    pts = _make_sample_room()
    r = LaserCornerExtractor.extract(pts, method="bearing_angle")
    assert len(r) > 1, (
        f"Bearing-angle found only {len(r)} corners – likely still inverted")
    print("PASS: test_bearing_angle_not_degenerate")


def test_line_segment_reasonable_count():
    """Line-segment should not produce hundreds of false corners."""
    pts = _make_sample_room()
    r = LaserCornerExtractor.extract(pts, method="line_segment")
    assert len(r) < 50, (
        f"Line-segment found {len(r)} corners – tolerance likely too small")
    print("PASS: test_line_segment_reasonable_count")


if __name__ == "__main__":
    test_few_points_returns_empty()
    test_straight_line_no_corners()
    test_l_shape_single_corner()
    test_sample_room_finds_all_corners()
    test_bearing_angle_not_degenerate()
    test_line_segment_reasonable_count()
    print("\nAll tests passed!")
