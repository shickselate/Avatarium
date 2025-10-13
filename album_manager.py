
# album_manager.py
# Minimal album utility for saving frames + metadata per character.
# Dependencies: Pillow

from __future__ import annotations
import os, json, time, shutil
from pathlib import Path
from typing import Dict, List, Tuple

ALBUM_ROOT = Path("album")

def _ts():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def ensure_album(char_name: str) -> Path:
    p = ALBUM_ROOT / char_name
    p.mkdir(parents=True, exist_ok=True)
    (p / "thumbs").mkdir(exist_ok=True)
    return p

def save_to_album(char_name: str, image_path: str, meta: Dict) -> Path:
    """
    Copies the image into album/<char> with a timestamped filename and writes a matching .json.
    Also creates a 512px thumbnail in album/<char>/thumbs.
    Returns the saved image path.
    """
    album = ensure_album(char_name)
    ts = _ts()
    ext = Path(image_path).suffix or ".png"
    dst_img = album / f"{ts}{ext}"
    dst_json = album / f"{ts}.json"

    shutil.copy2(image_path, dst_img)
    meta = dict(meta or {})
    meta["ts"] = ts
    meta["filename"] = dst_img.name

    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # thumbnail
    try:
        from PIL import Image
        im = Image.open(dst_img)
        im.thumbnail((512, 512))
        im.save(album / "thumbs" / f"{ts}.jpg", quality=92)
    except Exception as e:
        # non-fatal
        pass

    return dst_img

def list_album(char_name: str, limit: int = 20) -> List[Dict]:
    album = ensure_album(char_name)
    items = []
    for j in sorted(album.glob("*.json"), reverse=True):
        try:
            meta = json.loads(j.read_text(encoding="utf-8"))
            meta["image_path"] = str(album / meta.get("filename", ""))
            meta["thumb_path"] = str(album / "thumbs" / (j.stem + ".jpg"))
            items.append(meta)
        except Exception:
            continue
    return items[:limit]

def latest_image(char_name: str) -> str | None:
    items = list_album(char_name, limit=1)
    return items[0]["image_path"] if items else None
