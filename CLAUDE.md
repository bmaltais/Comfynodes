# CLAUDE.md — Comfynodes

## NEVER COMMIT DIRECTLY TO MAIN

Always create a feature branch before making any changes:

```bash
git checkout -b <branch-name>
```

Commit and push on the branch. Open a PR to merge into main.

---

## Project Structure

```
Comfynodes/
├── nodes.py          # ALL node classes live here (single file)
├── __init__.py       # Re-exports from nodes.py only — do not modify
├── requirements.txt  # pip dependencies (opencv-python, numpy, mediapipe)
└── CLAUDE.md
```

**Important**: All nodes are consolidated in `nodes.py`. Do NOT create per-folder node
files. The old subdirectory stubs (AnalogFilmNoiseNode/, ImageMergeNode/, etc.) are
legacy — ignore them.

## Adding a New Node

1. Add the class to `nodes.py`
2. Add an entry to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` at the bottom
3. If the node needs a JS frontend, add `web/<NodeName>.js` — files are loaded automatically
4. Add any new pip dependencies to `requirements.txt`

### Node class skeleton

```python
class MyNewNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "run"
    CATEGORY = "image/transform"   # use lowercase/slash convention

    def run(self, image):
        # image tensor shape: (B, H, W, C), float32, range [0, 1]
        return (image,)
```

## Standard Imports (already in nodes.py)

```python
import torch
import numpy as np
import cv2
import comfy.model_management
import comfy.utils
```

## Image Tensor Convention

ComfyUI IMAGE tensors are `(B, H, W, C)`, `torch.float32`, values in `[0, 1]`.

Convert to numpy for OpenCV:
```python
np_img = (image[0].cpu().numpy() * 255).astype(np.uint8)  # H×W×C, uint8
```

Convert back to tensor:
```python
out = torch.from_numpy(np_img.astype(np.float32) / 255.0).unsqueeze(0)  # 1×H×W×C
```

## JavaScript Frontend

To add a JS frontend to a node:

1. Create `web/` directory and add `WEB_DIRECTORY = "./web"` to `nodes.py`
2. Export `WEB_DIRECTORY` from `__init__.py`
3. Add `web/<NodeName>.js` — all `.js` files in `web/` are loaded automatically

```javascript
import { app } from "../../scripts/app.js";
app.registerExtension({ name: "comfynodes.<extension_name>", ... });
```

## Before Starting Any Work

**Step 0 — always create a feature branch first. Never work on main.**

```bash
git checkout -b <branch-name>
```

Check you are not on main:

```bash
git branch --show-current   # must NOT be "main"
```

## Before Committing

```bash
uv run black nodes.py
```

Install black if needed:

```bash
uv pip install black
```

## Testing a Node

Do this **before committing** any new or changed node — do not skip:

1. Restart ComfyUI (or use the Manager's "Reload Custom Nodes" if available)
2. Open browser DevTools console — check for JS errors on load
3. Search for the node by display name in the ComfyUI node menu
4. Connect inputs, run the workflow, check the ComfyUI terminal for Python errors
5. Confirm the node produces correct output before committing

## Dependencies

Install with:
```bash
uv pip install -r requirements.txt
```
