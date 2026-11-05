"""Turn a binary change mask into georeferenced features and summary statistics.

Geometry is the convex hull of each connected component's pixel corners. It is a valid
polygon but over-approximates concave shapes; the pixel area and the area computed from the
affine transform are reported separately so exact areas never depend on the hull.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

# 8-connectivity
_STRUCTURE = np.ones((3, 3), dtype=int)


def components(mask: np.ndarray, min_pixels: int = 4) -> list[dict]:
    """Connected components with at least ``min_pixels`` pixels."""
    labels, n = ndimage.label(mask.astype(bool), structure=_STRUCTURE)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        region = labels[sl] == i
        pixels = int(region.sum())
        if pixels < min_pixels:
            continue
        rows, cols = np.nonzero(region)
        rows, cols = rows + sl[0].start, cols + sl[1].start
        out.append({"id": len(out) + 1, "pixels": pixels, "rows": rows, "cols": cols})
    return out


def _hull(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Corner points of all component pixels, reduced to their convex hull (closed ring)."""
    corners = np.concatenate([np.stack([cols + dx, rows + dy], axis=1)
                              for dx in (0, 1) for dy in (0, 1)]).astype(float)
    corners = np.unique(corners, axis=0)
    try:
        hull = corners[ConvexHull(corners).vertices]
    except QhullError:  # degenerate (a line); fall back to the bounding box
        x0, y0 = corners.min(0)
        x1, y1 = corners.max(0)
        hull = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return np.vstack([hull, hull[:1]])


def to_geojson(mask: np.ndarray, origin: tuple[float, float] = (0.0, 0.0),
               pixel_size: tuple[float, float] = (1.0, 1.0), min_pixels: int = 4,
               prob: np.ndarray | None = None) -> dict:
    """GeoJSON FeatureCollection with one polygon per change region.

    ``origin`` is the (x, y) coordinate of the scene's top-left corner and ``pixel_size`` the
    (width, height) of a pixel in map units. Rows increase downwards, so y decreases.
    """
    x0, y0 = origin
    sx, sy = pixel_size
    features = []
    for comp in components(mask, min_pixels):
        ring = _hull(comp["rows"], comp["cols"])
        coords = [[x0 + c * sx, y0 - r * sy] for c, r in ring]
        props = {"id": comp["id"], "pixels": comp["pixels"], "area": comp["pixels"] * sx * sy,
                 "centroid_x": x0 + (comp["cols"].mean() + 0.5) * sx,
                 "centroid_y": y0 - (comp["rows"].mean() + 0.5) * sy}
        if prob is not None:
            props["mean_probability"] = float(prob[comp["rows"], comp["cols"]].mean())
        geometry = {"type": "Polygon", "coordinates": [coords]}
        features.append({"type": "Feature", "geometry": geometry, "properties": props})
    return {"type": "FeatureCollection", "features": features}


def change_statistics(mask: np.ndarray, pixel_area: float = 1.0, min_pixels: int = 4) -> dict:
    comps = components(mask, min_pixels)
    sizes = np.array([c["pixels"] for c in comps], dtype=float)
    return {
        "changed_fraction": float(mask.mean()),
        "changed_area": float(mask.sum() * pixel_area),
        "num_regions": len(comps),
        "region_area_median": float(np.median(sizes) * pixel_area) if len(sizes) else 0.0,
        "region_area_max": float(sizes.max() * pixel_area) if len(sizes) else 0.0,
    }
