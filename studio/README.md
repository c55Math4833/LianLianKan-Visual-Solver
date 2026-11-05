# LianLianKan (連連看) Python Visual Studio

The Visual Calibration, Slicing, Feature Extraction, Clustering, and Template Exporter Studio for the LianLianKan Visual Solver Engine.

## Overview & Architecture

- **`grid_calibrator.py`**:
  - **ROI Determination**: Supports interactive `cv2.selectROI`, manual `--roi x y w h`, or auto-detection.
  - **Adaptive Grid Pitch Detection**: Uses 1D signal autocorrelation on horizontal and vertical Sobel gradient projections ($P_x, P_y$) within lag range $[25, 100]$ px. Automatically determines grid dimensions (`rows` and `cols`) and precise floating-point cell boundaries without hardcoding board size.
  - Slices all board cells cleanly and outputs `debug_grid_on_roi.png`.

- **`feature_extractor.py`**:
  - Samples tile body color (outer frame ring) to eliminate central icon interference.
  - Distinguishes:
    - **Empty Cells** (`id: 0`): Identifies flat tan/khaki cleared background (`BGR ≈ [141, 168, 189]`, variance near zero).
    - **Obstacle Stones** (`id: -1`): Identifies the two solid grey stone blocks (at row 4, col 3 and row 4, col 11).
    - **Matchable Tiles** (`id: 1..N`): Extracts HSV color family (`GREY`, `YELLOW`, `CYAN_BLUE`, `VIOLET`, `MAGENTA`), HSV color histogram, and central icon shape/contour features (Hu moments).

- **`clusterer.py`**:
  - Implements shift-tolerant template matching (allowing translation tolerance for grid rasterization offsets).
  - Uses `scikit-learn` `AgglomerativeClustering` (complete linkage) with an adaptive threshold sweep.
  - **Strict Parity Constraint Enforcement**: Validates that 100% of matchable tile clusters contain an **even** number of tiles ($\ge 2$).
  - Computes the medoid tile of each cluster to generate representative templates.

- **`exporter.py`**:
  - Exports normalized template images into `templates/` (`templates/empty.png`, `templates/obstacle.png`, `templates/tile_01.png` .. `templates/tile_19.png`).
  - Exports `game_profile.json` strictly matching the Rust solver engine schema.
  - Generates a local, self-contained HTML inspection dashboard (`preview.html`).

- **`main.py`**:
  - End-to-end studio CLI orchestrator.

## Quick Start

Run the end-to-end pipeline using `uv`:

```bash
uv run python main.py
```

### Options

```bash
uv run python main.py --help
# Options:
#   --image PATH          Path to input board image (default: ..\data\未命名.jpg)
#   --roi X Y W H         Specify manual ROI bounding box
#   --interactive         Open OpenCV window for interactive cv2.selectROI
#   --open                Automatically open preview.html in default web browser
#   --templates DIR       Output directory for template images (default: templates)
#   --profile PATH        Output path for game_profile.json (default: game_profile.json)
#   --preview PATH        Output path for preview.html (default: preview.html)
```

## Outputs

- `game_profile.json`: Ready for direct consumption by the Rust solver engine.
- `templates/`: Normalized 45x45 PNG template library.
- `debug_grid_on_roi.png`: ROI bounding box and grid overlay verification image.
- `preview.html`: Visual inspection gallery and parity verification dashboard.
