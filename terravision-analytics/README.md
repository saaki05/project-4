# terravision-analytics

Scene-level change detection on top of `terravision-models`: whole-scene inference,
polygon extraction to GeoJSON, change statistics and a small HTTP API.

## What it does

- `save_checkpoint` / `load_checkpoint`: versioned model checkpoints (architecture, bands,
  widths, decision threshold, tile size).
- `detect`: sliding-window inference over a `(C, H, W)` before/after pair with overlapping
  tiles blended by a Hann window (from `terravision-datasets`), which suppresses tile seams.
- `to_geojson` / `change_statistics`: 8-connected components above a minimum size become
  GeoJSON polygons (convex hull of the component's pixel corners) with pixel count, area from
  the affine pixel size, centroid and mean change probability. The hull over-approximates
  concave regions; reported areas come from pixel counts, not from the hull.
- `terravision-detect` CLI for large arrays and a FastAPI app (`/health`, `/detect`) for small
  scenes (limited to 1 Mpixel; wrong band count, mismatched shapes or scenes smaller than a
  tile return 422).

## Install, test, run

```bash
git clone https://github.com/saaki05/project-4 && cd project-4
pip install -e terravision-datasets -e terravision-models
cd terravision-analytics && pip install -e ".[dev]" && pytest

terravision-detect --checkpoint model.pt --before before.npy --after after.npy \
    --origin 500000 4650000 --pixel-size 10 10 --out changes.geojson
```

## Limitations

- GeoTIFF reading and CRS handling are not included: the inputs are arrays, and the caller
  provides the origin and pixel size. Add a raster reader once real scenes are in hand.
- The checkpoint is loaded with `torch.load(weights_only=False)`; only load your own files.
- No model trained on real imagery ships with this repository.

## License

MIT.
