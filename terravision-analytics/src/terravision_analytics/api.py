"""HTTP API for small scenes. Large rasters should use the ``terravision-detect`` CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from terravision_analytics.inference import detect, load_checkpoint
from terravision_analytics.vectorize import change_statistics, to_geojson

MAX_PIXELS = 1024 * 1024


class DetectRequest(BaseModel):
    before: list[list[list[float]]] = Field(description="C x H x W")
    after: list[list[list[float]]] = Field(description="C x H x W")
    origin: tuple[float, float] = (0.0, 0.0)
    pixel_size: tuple[float, float] = (1.0, 1.0)
    threshold: float | None = None
    min_pixels: int = 4


def create_app(checkpoint_path: str | Path) -> FastAPI:
    model, meta = load_checkpoint(checkpoint_path)
    app = FastAPI(title="TerraVision", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **meta}

    @app.post("/detect")
    def detect_changes(req: DetectRequest) -> dict:
        try:
            b, a = np.asarray(req.before, np.float32), np.asarray(req.after, np.float32)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        if b.ndim != 3 or b.shape != a.shape:
            raise HTTPException(status_code=422, detail="before/after must both be C x H x W")
        if b.shape[0] != meta["bands"]:
            raise HTTPException(status_code=422, detail=f"expected {meta['bands']} bands")
        if min(b.shape[1:]) < meta["tile"]:
            raise HTTPException(status_code=422, detail=f"scene smaller than tile {meta['tile']}")
        if b.shape[1] * b.shape[2] > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="scene too large for the HTTP API")
        prob = detect(model, b, a, tile=meta["tile"])
        mask = prob >= (req.threshold if req.threshold is not None else meta["threshold"])
        return {"statistics": change_statistics(mask, req.pixel_size[0] * req.pixel_size[1],
                                                req.min_pixels),
                "geojson": to_geojson(mask, req.origin, req.pixel_size, req.min_pixels, prob)}

    return app
