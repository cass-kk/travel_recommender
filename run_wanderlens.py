#!/usr/bin/env python3
"""
WanderLens portfolio runner — one script to label your photos, discover album
themes (LDA), and get Texas travel recommendations (TF-IDF).

Usage:
  # Paid OpenAI (your existing key)
  set OPENAI_API_KEY=sk-...
  python run_wanderlens.py --photos "C:/path/to/my/album"

  # Free GitHub Models (PAT with models:read — NOT the same as OPENAI_API_KEY)
  set GITHUB_MODELS_TOKEN=ghp_...
  python run_wanderlens.py --provider github --photos "C:/path/to/my/album"
  python run_wanderlens.py --photos ./PicturesAlbum --only topics
  python run_wanderlens.py --photos ./vacation.jpg --album-name "Summer 2024"
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# HEIC/HEIF: use pillow_heif.read_heif() directly (register_heif_opener alone is easy to miss)
HEIF_AVAILABLE = False
try:
    import pillow_heif as _pillow_heif

    HEIF_AVAILABLE = True
    try:
        _pillow_heif.register_heif_opener()
    except Exception:
        pass  # explicit read_heif path still works
except ImportError:
    _pillow_heif = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ITEMS_CSV = SCRIPT_DIR / "things_to_do_all.csv"

# GitHub Models (free tier) — OpenAI-compatible API, different auth/URL/model names.
# Docs: https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
GITHUB_DEFAULT_VISION_MODEL = "openai/gpt-4o"
OPENAI_DEFAULT_VISION_MODEL = "gpt-4o"

IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".tif", ".tiff", ".gif", ".bmp",
}

# User-facing album id in CSVs
ALBUM_COL = "album"
FILENAME_COL = "filename"

# Define categories & schema. From Photo Tagger.ipynb
CATEGORIES = [
    "Outdoor Activities",
    "Food & Drink",
    "Cultural",
    "Night Life",
    "Concerts & Shows",
    "Casino & Gambling",
    "Attractions",
    "Sporting Events",
]

LABEL_SCHEMA = {
    "name": "photo_labels",
    "schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "description": "5–12 concise tags capturing scene, activity, objects, vibe.",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 12,
            },
            "category_distribution": {
                "type": "object",
                "description": "Probability mass across the EXACT 8 categories; MUST sum to 1.0.",
                "properties": {cat: {"type": "number", "minimum": 0, "maximum": 1} for cat in CATEGORIES},
                "required": CATEGORIES,
                "additionalProperties": False,
            },
            "attributes": {
                "type": "object",
                "properties": {
                    "indoor_outdoor": {"type": "string", "enum": ["indoor", "outdoor", "mixed", "unknown"]},
                    "day_night": {"type": "string", "enum": ["day", "night", "dusk/dawn", "unknown"]},
                    "season_hint": {"type": "string", "enum": ["winter", "spring", "summer", "autumn", "unknown"]},
                },
                "required": ["indoor_outdoor", "day_night", "season_hint"],
                "additionalProperties": False,
            },
        },
        "required": ["labels", "category_distribution", "attributes"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You are labeling personal photos for travel preference modeling.\n"
    "Return STRICT JSON matching the provided schema. Never include prose. \n"
    "Prefer compact, generalizable tags (e.g., 'hiking trail', 'alpine lake', 'sushi', 'museum', 'beach', 'friends hangout').\n"
    "Distribute probability mass across EXACTLY these 8 categories so they SUM TO 1.0:\n"
    f"{', '.join(CATEGORIES)}.\n"
    "If uncertain, distribute softly but still sum to 1.00. Prefer compact, generizable tags."
)


def to_safe_id(s: str) -> str:
    """Lowercase alphanumeric + underscores (safe for column names and file prefixes)."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s\-._]", "", str(name))
    name = re.sub(r"\s+", "_", name.strip())
    return name or "album"


def album_file_prefix(display_name: str) -> str:
    """Filesystem-safe id for output folder name and file prefixes (e.g. sample_album)."""
    return to_safe_id(display_name) or safe_filename(display_name).lower() or "album"


def album_output_paths(base_out: Path, display_name: str) -> Dict[str, Path]:
    """Outputs under output/<folder_name>/ with <album_file_prefix>_*.csv/xlsx names."""
    prefix = album_file_prefix(display_name)
    folder = base_out / prefix
    return {
        "folder": folder,
        "folder_name": prefix,
        "album_file_prefix": prefix,
        "display_name": display_name,
        "labels_per_photo": folder / f"{prefix}_labels_per_photo.csv",
        "label_counts": folder / f"{prefix}_label_counts.csv",
        "category_means_long": folder / f"{prefix}_category_means_long.csv",
        "topics": folder / f"{prefix}_topics.csv",
        "recommendations": folder / f"{prefix}_recommendations.xlsx",
        "summary": folder / f"{prefix}_summary.xlsx",
    }


def ensure_album_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if ALBUM_COL in df.columns:
        return df
    for legacy in ("person", "owner"):
        if legacy in df.columns:
            return df.rename(columns={legacy: ALBUM_COL})
    raise ValueError(f"Expected '{ALBUM_COL}' column.")


def add_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add filename from path if missing (album → filename → path column order)."""
    df = df.copy()
    if "path" in df.columns:
        df[FILENAME_COL] = df["path"].astype(str).map(lambda p: Path(p).name)
    elif FILENAME_COL not in df.columns:
        df[FILENAME_COL] = ""
    return df


def order_labels_per_photo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Stable column order for labels_per_photo exports."""
    df = add_filename_column(ensure_album_column(df))
    cat_cols = [f"cat_{to_safe_id(c)}" for c in CATEGORIES]
    base_cols = [
        ALBUM_COL,
        FILENAME_COL,
        "path",
        "labels",
        "indoor_outdoor",
        "day_night",
        "season_hint",
    ]
    other_cols = [c for c in df.columns if c not in base_cols + cat_cols]
    ordered = [c for c in base_cols + cat_cols + other_cols if c in df.columns]
    return df[ordered]


def detect_album_col(cols) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for key in (ALBUM_COL, "person", "owner"):
        if key in lower:
            return lower[key]
    for c in cols:
        if "album" in c.lower() or "person" in c.lower() or "owner" in c.lower():
            return c
    return None


def parse_labels_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "labels" in df.columns:
        df["labels"] = df["labels"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and str(x).strip().startswith("[") else x
        )
    return df


def find_existing_labels_csv(out_dir: Path, display_name: Optional[str]) -> Optional[Path]:
    """Locate labels CSV from a prior run (new layout or group project flat files)."""
    if display_name:
        p = album_output_paths(out_dir, display_name)["labels_per_photo"]
        if p.is_file():
            return p
    for sub in sorted(out_dir.iterdir()) if out_dir.is_dir() else []:
        if sub.is_dir():
            for f in sub.glob("*_labels_per_photo.csv"):
                return f
    legacy = out_dir / "labels_per_photo.csv"
    return legacy if legacy.is_file() else None


def resolve_display_name(cli_name: Optional[str], images: List[Dict[str, str]], multi_album: bool) -> Optional[str]:
    if cli_name and multi_album:
        print(
            "Note: --name is ignored when --photos contains multiple album subfolders; "
            "each subfolder name is used instead.",
        )
        return None
    return cli_name


def discover_images(
    photos_path: Path, display_name: Optional[str]
) -> tuple[List[Dict[str, str]], bool]:
    """local folder or file discovery. From Photo Tagger.ipynb"""
    photos_path = photos_path.resolve()

    if photos_path.is_file():
        if photos_path.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Unsupported image type: {photos_path.suffix}")
        if photos_path.stat().st_size == 0:
            raise ValueError(f"Empty file: {photos_path}")
        name = display_name or photos_path.parent.name or "My Album"
        return [{"album": name, "path": str(photos_path)}], False

    if not photos_path.is_dir():
        raise FileNotFoundError(f"Path not found: {photos_path}")

    subdirs = [p for p in photos_path.iterdir() if p.is_dir()]

    def folder_has_images(folder: Path) -> bool:
        return any(
            p.is_file() and p.suffix.lower() in IMG_EXTS and p.stat().st_size > 0
            for p in folder.rglob("*")
        )

    multi_album = bool(subdirs) and any(folder_has_images(sd) for sd in subdirs)
    images: List[Dict[str, str]] = []

    if multi_album:
        for sd in sorted(subdirs):
            if not folder_has_images(sd):
                continue
            for p in sorted(sd.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stat().st_size > 0:
                    images.append({"album": sd.name, "path": str(p)})
    else:
        name = display_name or photos_path.name or "My Album"
        for p in sorted(photos_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stat().st_size > 0:
                images.append({"album": name, "path": str(p)})

    if not images:
        raise ValueError(
            f"No images found under {photos_path}. "
            f"Supported extensions: {', '.join(sorted(IMG_EXTS))}"
        )
    if display_name and not multi_album:
        for item in images:
            item["album"] = display_name
    return images, multi_album


def load_image_rgb(img_path: str) -> tuple[Optional[Image.Image], Optional[str]]:
    """
    Load any supported photo as RGB PIL Image.
    Returns (image, None) or (None, error_code_message).
    HEIC uses pillow_heif.read_heif — does not rely on register_heif_opener timing.
    """
    path = Path(img_path)
    if not path.is_file():
        return None, "file_not_found"
    try:
        size = path.stat().st_size
    except OSError as e:
        return None, f"stat_error:{e}"
    if size == 0:
        return None, "empty_file"

    ext = path.suffix.lower()
    if ext in (".heic", ".heif"):
        if not HEIF_AVAILABLE:
            return None, (
                "heic_requires_pillow_heif — install on THIS python: "
                f'"{sys.executable}" -m pip install pillow-heif'
            )
        try:
            heif_file = _pillow_heif.read_heif(str(path))
            # Primary image (iPhone HEIC is usually one frame)
            if hasattr(heif_file, "to_pillow"):
                im = heif_file.to_pillow()
            else:
                frame = heif_file[0] if len(heif_file) else heif_file
                im = frame.to_pillow() if hasattr(frame, "to_pillow") else Image.open(path)
            return im.convert("RGB"), None
        except Exception as e:
            return None, f"heic_decode_error:{type(e).__name__}:{e}"

    try:
        with Image.open(path) as opened:
            return opened.convert("RGB"), None
    except Exception as e:
        hint = ""
        if ext in (".heic", ".heif"):
            hint = " (try: pip install pillow-heif)"
        return None, f"image_open_error:{type(e).__name__}:{e}{hint}"


def to_data_url(img_path: str, max_side: int = 768) -> tuple[Optional[str], Optional[str]]:
    """Returns (data_url, error_message). From Photo Tagger.ipynb"""
    im, err = load_image_rgb(img_path)
    if err or im is None:
        return None, err or "unreadable_or_unsupported"
    ratio = max(im.size) / max_side if max(im.size) > max_side else 1.0
    if ratio > 1.0:
        new_size = (int(im.size[0] / ratio), int(im.size[1] / ratio))
        im = im.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", None


def _check_heic_support(images: List[Dict[str, str]]) -> None:
    """Warn early if batch has HEIC but pillow-heif is missing on this interpreter."""
    heic_paths = [i["path"] for i in images if Path(i["path"]).suffix.lower() in (".heic", ".heif")]
    if not heic_paths:
        return
    if HEIF_AVAILABLE:
        print(f"HEIC support: enabled (pillow-heif) — {len(heic_paths)} HEIC/HEIF file(s) in batch")
        return
    print(
        f"\nWARNING: {len(heic_paths)} HEIC/HEIF photo(s) in this run but pillow-heif is NOT installed "
        f"for:\n  {sys.executable}\n"
        f'  Fix: "{sys.executable}" -m pip install pillow-heif\n',
        file=sys.stderr,
    )


def _github_pat_from_env() -> str:
    """GitHub PAT with models:read (prefer GITHUB_MODELS_TOKEN to avoid clashing with git)."""
    return (
        os.environ.get("GITHUB_MODELS_TOKEN", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )


def resolve_vision_llm(provider: str = "auto", model: Optional[str] = None):
    """
    Return (OpenAI SDK client, model id, provider label).

    - openai: OPENAI_API_KEY → api.openai.com, model e.g. gpt-4o
    - github: GITHUB_MODELS_TOKEN or GITHUB_TOKEN → models.github.ai, model e.g. openai/gpt-4o
    - auto: github if only GitHub token is set; else OpenAI
    """
    from openai import OpenAI

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    github_pat = _github_pat_from_env()
    chosen = provider.lower()

    if chosen == "auto":
        if github_pat and not openai_key:
            chosen = "github"
        else:
            chosen = "openai"

    if chosen == "github":
        if not github_pat:
            github_pat = input(
                "Paste your GitHub PAT (models:read scope). "
                "Save it as GITHUB_MODELS_TOKEN for next time: "
            ).strip()
            os.environ["GITHUB_MODELS_TOKEN"] = github_pat
        client = OpenAI(base_url=GITHUB_MODELS_BASE_URL, api_key=github_pat)
        model_id = model or GITHUB_DEFAULT_VISION_MODEL
        if model_id.startswith("gpt-") and "/" not in model_id:
            model_id = f"openai/{model_id}"
        print(f"Using GitHub Models (free tier) at {GITHUB_MODELS_BASE_URL}")
        print(f"  Model: {model_id}")
        print("  Note: daily rate limits apply (see GitHub Models docs).")
        return client, model_id, "github"

    if not openai_key:
        openai_key = input("Paste your OpenAI API key (or set OPENAI_API_KEY): ").strip()
        os.environ["OPENAI_API_KEY"] = openai_key
    client = OpenAI(api_key=openai_key)
    model_id = model or OPENAI_DEFAULT_VISION_MODEL
    print(f"Using OpenAI API, model: {model_id}")
    return client, model_id, "openai"


def label_photo(client, model: str, data_url: str) -> Dict[str, Any]:
    """From Photo Tagger.ipynb."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Label this image following the schema."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        response_format={"type": "json_schema", "json_schema": LABEL_SCHEMA},
        max_tokens=300,
    )
    return json.loads(completion.choices[0].message.content)


def run_photo_labeling(
    images: List[Dict[str, str]],
    sleep_s: float = 0.15,
    provider: str = "auto",
    model: Optional[str] = None,
) -> pd.DataFrame:
    """From Photo Tagger.ipynb."""
    _check_heic_support(images)
    client, model_id, _ = resolve_vision_llm(provider=provider, model=model)
    rows = []
    for item in tqdm(images, desc="Labeling photos"):
        data_url, load_err = to_data_url(item["path"])
        img_path = item["path"]
        img_name = Path(img_path).name
        if data_url is None:
            rows.append({
                ALBUM_COL: item[ALBUM_COL],
                FILENAME_COL: img_name,
                "path": img_path,
                "error": load_err or "unreadable_or_unsupported",
            })
            continue
        try:
            out = label_photo(client, model_id, data_url)
            row = {
                ALBUM_COL: item[ALBUM_COL],
                FILENAME_COL: img_name,
                "path": img_path,
                "labels": out["labels"],
                "indoor_outdoor": out["attributes"]["indoor_outdoor"],
                "day_night": out["attributes"]["day_night"],
                "season_hint": out["attributes"]["season_hint"],
            }
            for cat in CATEGORIES:
                val = out.get("category_distribution", {}).get(cat, 0.0)
                row[f"cat_{to_safe_id(cat)}"] = float(val)
            rows.append(row)
            time.sleep(sleep_s)
        except Exception as e:
            rows.append({
                ALBUM_COL: item[ALBUM_COL],
                FILENAME_COL: img_name,
                "path": img_path,
                "error": str(e),
            })

    return order_labels_per_photo_columns(pd.DataFrame(rows))


def aggregate_labels(df: pd.DataFrame, out_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Aggregations, one output set per album name. From Photo Tagger.ipynb"""
    df = ensure_album_column(df)
    cat_cols = [f"cat_{to_safe_id(c)}" for c in CATEGORIES]
    category_key_to_label = {f"cat_{to_safe_id(c)}": c for c in CATEGORIES}
    by_album: Dict[str, Dict[str, pd.DataFrame]] = {}

    for album_name, album_df in df.groupby(ALBUM_COL):
        album_name = str(album_name)
        paths = album_output_paths(out_dir, album_name)
        paths["folder"].mkdir(parents=True, exist_ok=True)

        df_ok = album_df.dropna(subset=["labels"]).copy().explode("labels")
        label_counts = (
            df_ok.groupby([ALBUM_COL, "labels"])
            .size()
            .reset_index(name="count")
            .sort_values([ALBUM_COL, "count"], ascending=[True, False])
        )
        cat_means_long = (
            album_df[[ALBUM_COL] + cat_cols]
            .melt(ALBUM_COL, var_name="category_column_key", value_name="prob")
            .assign(category=lambda d: d["category_column_key"].map(category_key_to_label))
            .groupby([ALBUM_COL, "category"], as_index=False)["prob"]
            .mean()
            .rename(columns={"prob": "avg_prob"})
            .sort_values([ALBUM_COL, "avg_prob"], ascending=[True, False])
        )

        album_df = order_labels_per_photo_columns(album_df)
        album_df.to_csv(paths["labels_per_photo"], index=False)
        label_counts.to_csv(paths["label_counts"], index=False)
        cat_means_long.to_csv(paths["category_means_long"], index=False)
        print(f"Saved album outputs under {paths['folder']} ({paths['album_file_prefix']}_*)")

        by_album[album_name] = {
            "paths": paths,
            "labels_per_photo": album_df,
            "label_counts": label_counts,
            "category_means_long": cat_means_long,
        }
    return by_album


def _require_lda_packages():
    """LDA needs nltk + gensim packages loaded in the same Python location as this script."""
    missing = []
    for pkg in ("nltk", "gensim"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        exe = sys.executable
        print(
            f"\nMissing packages for album topics (LDA): {', '.join(missing)}\n"
            f"Install with the SAME python that runs this script:\n\n"
            f'  "{exe}" -m pip install nltk gensim\n'
            f'  "{exe}" -m pip install -r requirements.txt\n',
            file=sys.stderr,
        )
        sys.exit(1)


def run_lda(df_labels: pd.DataFrame, out_dir: Path) -> Dict[str, pd.DataFrame]:
    """From Final_Project_LDA.ipynb"""
    _require_lda_packages()
    import nltk
    from gensim import corpora
    from gensim.models import LdaModel
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def clean_labels(label_list):
        if not isinstance(label_list, list):
            try:
                label_list = ast.literal_eval(label_list)
            except Exception:
                return []
        cleaned = []
        for lbl in label_list:
            if not isinstance(lbl, str):
                continue
            lbl = re.sub(r"[^a-zA-Z\s]", "", lbl).lower()
            words = [lemmatizer.lemmatize(w) for w in lbl.split() if w not in stop_words and len(w) > 2]
            if words:
                cleaned.append("_".join(words))
        return cleaned

    df = ensure_album_column(df_labels.copy())
    df["tokens"] = df["labels"].apply(clean_labels)

    all_topics_frames = []
    for album_name, subdf in df.groupby(ALBUM_COL):
        album_name = str(album_name)
        print(f"\n=== LDA for {album_name} ({len(subdf)} photos) ===")
        if len(subdf) < 3:
            print("  Skipped (need at least 3 labeled photos).")
            continue

        dictionary = corpora.Dictionary(subdf["tokens"])
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(tokens) for tokens in subdf["tokens"]]
        num_topics = min(5, max(3, len(subdf) // 5))

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
        )

        topics_data = []
        for i, topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
            words = [word for word, _ in topic]
            topics_data.append({
                ALBUM_COL: album_name,
                "topic_number": i,
                "top_words": ", ".join(words),
            })
            print(f"  Topic {i}: {', '.join(words)}")

        topics_df = pd.DataFrame(topics_data)
        out_file = album_output_paths(out_dir, album_name)["topics"]
        out_file.parent.mkdir(parents=True, exist_ok=True)
        topics_df.to_csv(out_file, index=False)
        print(f"  Saved {out_file}")
        all_topics_frames.append(topics_df)

    combined = pd.concat(all_topics_frames, ignore_index=True) if all_topics_frames else pd.DataFrame()
    return {"lda_topics": combined}


# From travel_recommender_code.ipynb (TF-IDF recommender)

def build_listing_text(df_items: pd.DataFrame) -> pd.Series:
    """
    Build one text field per listing from CSV columns (title, category, description, etc.).
    No image download or vision API. Hero URL filename tokens are optional light cues.
    """
    if "title" not in df_items.columns:
        raise ValueError("Listings CSV must include a 'title' column.")

    parts: List[pd.Series] = [df_items["title"].astype(str).map(normalize)]

    for col in ("category", "meta_description", "description", "body", "summary", "snippet"):
        if col in df_items.columns:
            parts.append(df_items[col].fillna("").astype(str).map(normalize))

    for col in [c for c in df_items.columns if "label" in c.lower()]:
        filled = df_items[col].fillna("").astype(str).str.strip()
        if filled.ne("").any():
            parts.append(filled.apply(parse_maybe_list).map(normalize))

    if "hero_image_url" in df_items.columns:

        def _hero_filename_tokens(url: Any) -> str:
            if not isinstance(url, str) or not url.strip():
                return ""
            fname = url.split("/")[-1].split(".")[0]
            tokens = re.split(r"[_\-\./]+", fname)
            return normalize(" ".join(t for t in tokens if len(t) > 2))

        parts.append(df_items["hero_image_url"].map(_hero_filename_tokens))

    text = parts[0]
    for p in parts[1:]:
        text = (text + " " + p).str.strip()
    return text


def normalize(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def parse_maybe_list(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, (list, tuple)):
                return " ".join(map(str, arr))
        except Exception:
            pass
    return s


def expand_counts_to_text(series_terms, series_counts):
    tokens = []
    for term, cnt in zip(series_terms, series_counts):
        term = normalize(term)
        try:
            n = int(round(float(cnt)))
        except Exception:
            n = 1
        tokens.extend([term] * max(n, 1))
    return " ".join(tokens)


def detect_url_col(cols):
    lower = {c.lower(): c for c in cols}
    for key in ["detail_url", "url", "link", "website", "details_url", "detailurl", "detailURL"]:
        if key in lower:
            return lower[key]
    for c in cols:
        cl = c.lower()
        if "url" in cl or "link" in cl or ("detail" in cl and "id" not in cl):
            return c
    return None


def run_recommendations(
    label_counts_csv: Path,
    items_csv: Path,
    recs_xlsx: Path,
    album_display_name: str,
    top_n: int = 5,
) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    df_items = pd.read_csv(items_csv)
    df_items["_text"] = build_listing_text(df_items)

    url_col = detect_url_col(df_items.columns)
    item_urls = df_items[url_col].astype(str).tolist() if url_col else [""] * len(df_items)
    item_titles = df_items["title"].astype(str).tolist()

    df_users = pd.read_csv(label_counts_csv)
    album_col = detect_album_col(df_users.columns)
    lower = {c.lower(): c for c in df_users.columns}
    label_col = lower.get("label") or lower.get("labels") or lower.get("tag") or next(
        (c for c in df_users.columns if "label" in c.lower() or "tag" in c.lower()), None
    )
    count_col = next(
        (c for c in df_users.columns if any(k in c.lower() for k in ["count", "freq", "weight", "score", "frequency"])),
        None,
    )
    if not label_col:
        raise ValueError(f"Need a 'label' column in {label_counts_csv.name}.")

    if count_col is None:
        df_users["_count"] = 1
        count_col = "_count"

    if album_col:
        sub = df_users[df_users[album_col].astype(str) == str(album_display_name)]
        if sub.empty:
            sub = df_users
    else:
        sub = df_users

    text = expand_counts_to_text(sub[label_col].astype(str).tolist(), sub[count_col].tolist())
    text = normalize(text)
    if not text:
        raise ValueError("No label profile generated. Run labeling/aggregate first.")

    df_profiles = pd.DataFrame([{ALBUM_COL: str(album_display_name), "text": text}])

    combined_text = pd.concat([df_profiles["text"], df_items["_text"]], ignore_index=True)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(combined_text)
    n_people = df_profiles.shape[0]
    people_mat = X[:n_people]
    items_mat = X[n_people:]
    sim = cosine_similarity(people_mat, items_mat)
    feat = np.array(vectorizer.get_feature_names_out())

    def why_terms(profile_idx, item_idx, top_k=5):
        p_vec = people_mat[profile_idx]
        i_vec = items_mat[item_idx]
        p_idx = set(p_vec.indices.tolist())
        i_idx = set(i_vec.indices.tolist())
        inter = np.array(sorted(p_idx & i_idx))
        if inter.size == 0:
            return ""
        contrib = p_vec[:, inter].toarray().ravel() * i_vec[:, inter].toarray().ravel()
        order = np.argsort(-contrib)[:top_k]
        return ", ".join(feat[inter[order]])

    recs_xlsx.parent.mkdir(parents=True, exist_ok=True)
    album_name = df_profiles[ALBUM_COL].iloc[0]
    row = sim[0]
    top_idx = np.argsort(-row)[:top_n]
    print(f"\n=== Recommendations (TF-IDF) for {album_name} — Top {top_n} ===")
    rows = []
    for rank, j in enumerate(top_idx, start=1):
        title = item_titles[j]
        score = float(row[j])
        url = item_urls[j] if j < len(item_urls) else ""
        why = why_terms(0, j)
        print(f"{rank:>2}. {title}  |  score={score:.4f}  |  url: {url}\n    why: {why}")
        rows.append({
            ALBUM_COL: album_name,
            "rank": rank,
            "title": title,
            "score": round(score, 4),
            "url": url,
            "why": why,
        })
    out_df = pd.DataFrame(rows)
    try:
        with pd.ExcelWriter(recs_xlsx, engine="xlsxwriter") as writer:
            out_df.to_excel(writer, index=False, sheet_name="recommendations")
            wb = writer.book
            ws = writer.sheets["recommendations"]
            for r in range(len(out_df)):
                url_val = out_df.at[r, "url"]
                if isinstance(url_val, str) and url_val.startswith(("http://", "https://")):
                    ws.write_url(r + 1, 3, url_val, string=url_val)
    except ModuleNotFoundError:
        with pd.ExcelWriter(recs_xlsx, engine="openpyxl") as writer:
            out_df.to_excel(writer, index=False, sheet_name="recommendations")
    print(f"Saved {recs_xlsx}")
    return out_df


def run_recommendations_all_albums(
    out_dir: Path,
    items_csv: Path,
    top_n: int = 5,
    album_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Run TF-IDF recs for each album folder that has label_counts CSV."""
    all_recs = []
    for sub in sorted(out_dir.iterdir()) if out_dir.is_dir() else []:
        if not sub.is_dir():
            continue
        counts_files = list(sub.glob("*_label_counts.csv"))
        if not counts_files:
            continue
        counts_csv = counts_files[0]
        display_name = None
        if ALBUM_COL in pd.read_csv(counts_csv, nrows=1).columns:
            sample = pd.read_csv(counts_csv)
            if not sample.empty:
                display_name = str(sample[ALBUM_COL].iloc[0])
        if display_name is None:
            display_name = counts_csv.stem.replace("_label_counts", "").replace("_", " ")
        if album_filter and display_name != album_filter:
            continue
        paths = album_output_paths(out_dir, display_name)
        recs = run_recommendations(
            counts_csv, items_csv, paths["recommendations"], display_name, top_n=top_n
        )
        all_recs.append(recs)
    if not all_recs:
        raise FileNotFoundError(
            f"No *_label_counts.csv found under {out_dir}. Run aggregate step first."
        )
    return pd.concat(all_recs, ignore_index=True)


def print_photo_labels(df: pd.DataFrame, album: Optional[str] = None, limit: int = 15):
    df = ensure_album_column(df)
    sub = df if album is None else df[df[ALBUM_COL] == album]
    if sub.empty:
        print("No labeled photos to show.")
        return
    print("\n--- Photo labels (sample) ---")
    for _, r in sub.head(limit).iterrows():
        fname = r.get(FILENAME_COL, Path(str(r["path"])).name)
        labels = r.get("labels", "")
        print(f"  {r[ALBUM_COL]} | {fname}\n    {labels}")


def top10_from_label_counts(label_counts: pd.DataFrame) -> pd.DataFrame:
    """Top 10 tags per album from full label_counts table."""
    label_counts = ensure_album_column(label_counts)
    if ALBUM_COL in label_counts.columns:
        return label_counts.groupby(ALBUM_COL, group_keys=False).head(10)
    return label_counts.head(10)


def print_top_labels(label_counts: pd.DataFrame, album: Optional[str] = None):
    top10_df = top10_from_label_counts(label_counts)
    sub = top10_df if album is None else top10_df[top10_df[ALBUM_COL] == album]
    print("\n--- Top labels per album ---")
    group_col = ALBUM_COL if ALBUM_COL in sub.columns else None
    if group_col:
        for name, g in sub.groupby(group_col):
            tags = ", ".join(f"{row['labels']} ({row['count']})" for _, row in g.iterrows())
            print(f"  {name}: {tags}")
    else:
        tags = ", ".join(f"{row['labels']} ({row['count']})" for _, row in sub.iterrows())
        print(f"  {tags}")


def print_topics(topics_df: pd.DataFrame, album: Optional[str] = None):
    if topics_df.empty:
        print("\n--- Album topics ---\n  (none — need at least 3 labeled photos per album for LDA)")
        return
    topics_df = ensure_album_column(topics_df)
    sub = topics_df if album is None else topics_df[topics_df[ALBUM_COL] == album]
    print("\n--- Album topics (LDA) ---")
    for name, g in sub.groupby(ALBUM_COL):
        print(f"  {name}:")
        for _, row in g.iterrows():
            print(f"    Topic {row['topic_number']}: {row['top_words']}")


def write_summary_workbook(summary_path: Path, tables: Dict[str, pd.DataFrame]):
    """Excel workbook for one album (named <album_file_prefix>_summary.xlsx)."""
    sheet_order = (
        "labels_per_photo",
        "all_labels",
        "top10_labels",
        "album_topics",
        "recommendations",
    )
    summary = summary_path
    try:
        with pd.ExcelWriter(summary, engine="openpyxl") as writer:
            for sheet in sheet_order:
                frame = tables.get(sheet)
                if frame is None or frame.empty:
                    continue
                if sheet == "labels_per_photo" and "error" in frame.columns:
                    ok = frame["error"].isna() | (frame["error"].astype(str).str.strip() == "")
                    frame = frame.loc[ok]
                if frame.empty:
                    continue
                frame.to_excel(writer, index=False, sheet_name=sheet[:31])
        print(f"\nSummary workbook: {summary}")
    except Exception as e:
        print(f"Could not write summary workbook: {e}")


def parse_args():
    p = argparse.ArgumentParser(
        description="WanderLens: label your travel photos and get Texas activity recommendations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--photos",
        default=None,
        help="Path to a photo file, a folder of photos, or a folder of album subfolders.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use bundled sample CSVs in this repo (no OpenAI calls). Shows topics + recommendations.",
    )
    p.add_argument(
        "--name",
        "--album-name",
        dest="name",
        default=None,
        metavar="ALBUM_NAME",
        help="Your album name (labels CSVs, topics, recs use this in filenames). "
        "Default: photo folder name. Ignored if --photos has multiple album subfolders.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write CSV/Excel outputs (default: ./output next to this script).",
    )
    p.add_argument(
        "--items-csv",
        default=str(DEFAULT_ITEMS_CSV),
        help="Listings CSV for recommendations (default: things_to_do_all.csv). "
        "Uses title, category, and any description/meta/body columns for matching.",
    )
    p.add_argument("--top-n", type=int, default=5, help="Number of recommendations per album.")
    p.add_argument("--max-photos", type=int, default=None, help="Limit photos to label (testing).")
    p.add_argument(
        "--skip-labeling",
        action="store_true",
        help="Reuse existing <album_file_prefix>_labels_per_photo.csv under --output-dir (use with --name if set).",
    )
    p.add_argument(
        "--only",
        choices=["all", "label", "aggregate", "topics", "recs"],
        default="all",
        help="Run a subset of the pipeline.",
    )
    p.add_argument(
        "--album",
        "--person",
        dest="album",
        default=None,
        metavar="ALBUM_NAME",
        help="Only print/process this album name (must match --name or folder name).",
    )
    p.add_argument(
        "--provider",
        choices=["auto", "openai", "github"],
        default="auto",
        help="Vision API: openai (OPENAI_API_KEY), github (GITHUB_MODELS_TOKEN), or auto.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override vision model (e.g. gpt-4o or openai/gpt-4o). See github.com/marketplace/models.",
    )
    return p.parse_args()


def load_bundled_sample(out_dir: Path) -> pd.DataFrame:
    """Copy teammate sample outputs shipped with the repo (portfolio demo)."""
    src = SCRIPT_DIR / "labels_per_photo.csv"
    if not src.is_file():
        raise FileNotFoundError(
            f"Bundled sample not found: {src}. Run with --photos to label your own album."
        )
    df = parse_labels_column(ensure_album_column(pd.read_csv(src)))
    aggregate_labels(df, out_dir)
    print(f"Loaded bundled sample ({len(df)} labeled photos, multiple albums) into {out_dir}")
    return df


def main():
    args = parse_args()
    if not args.demo and not args.photos:
        print("Provide --photos <path> or use --demo for bundled sample data.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_labels: Optional[pd.DataFrame] = None
    agg_by_album: Dict[str, Dict[str, pd.DataFrame]] = {}
    topics_df = pd.DataFrame()
    recs_df = pd.DataFrame()
    primary_album: Optional[str] = args.name or args.album

    run_label = args.only in ("all", "label", "aggregate", "topics")
    run_agg = args.only in ("all", "aggregate", "topics", "recs")
    run_topics = args.only in ("all", "topics")
    run_recs = args.only in ("all", "recs")

    if args.demo:
        df_labels = load_bundled_sample(out_dir)
        run_label = False
        primary_album = primary_album or str(df_labels[ALBUM_COL].iloc[0])
    elif args.skip_labeling or (not run_label and (run_agg or run_topics or run_recs)):
        labels_path = find_existing_labels_csv(out_dir, args.name)
        if not labels_path:
            print(
                "No labels file found. Run labeling first, or use --name to match your album folder.\n"
                f"  Looked under: {out_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loading existing labels from {labels_path}")
        df_labels = order_labels_per_photo_columns(
            parse_labels_column(pd.read_csv(labels_path))
        )
        if not primary_album and ALBUM_COL in df_labels.columns:
            primary_album = str(df_labels[ALBUM_COL].iloc[0])
    elif run_label:
        photos_path = Path(args.photos).expanduser()
        images, multi_album = discover_images(photos_path, args.name)
        resolve_display_name(args.name, images, multi_album)
        if args.max_photos:
            images = images[: args.max_photos]
        albums = sorted({i[ALBUM_COL] for i in images})
        print(f"Found {len(images)} photo(s) across {len(albums)} album(s): {', '.join(albums)}")
        if args.name and not multi_album:
            print(f"Album name for outputs: {args.name}")
        df_labels = run_photo_labeling(images, provider=args.provider, model=args.model)
        primary_album = primary_album or str(df_labels[ALBUM_COL].iloc[0])
        if run_agg or args.only == "label":
            agg_by_album = aggregate_labels(df_labels, out_dir)

    if df_labels is not None and run_agg and not agg_by_album:
        agg_by_album = aggregate_labels(df_labels, out_dir)

    if df_labels is not None and run_topics:
        if args.album:
            sub = df_labels[df_labels[ALBUM_COL] == args.album]
            lda_out = run_lda(sub, out_dir)
        else:
            lda_out = run_lda(df_labels, out_dir)
        topics_df = lda_out.get("lda_topics", pd.DataFrame())

    if run_recs:
        items_path = Path(args.items_csv)
        if not items_path.is_file():
            print(f"Items file not found: {items_path}", file=sys.stderr)
            sys.exit(1)
        recs_df = run_recommendations_all_albums(
            out_dir, items_path, top_n=args.top_n, album_filter=args.album
        )

    album_filter = args.album
    if df_labels is not None:
        print_photo_labels(df_labels, album=album_filter)
    if primary_album and primary_album in agg_by_album:
        print_top_labels(agg_by_album[primary_album]["label_counts"], album=album_filter)
    elif agg_by_album:
        for data in agg_by_album.values():
            print_top_labels(data["label_counts"], album=album_filter)
    elif primary_album:
        p = album_output_paths(out_dir, primary_album)["label_counts"]
        if p.exists():
            print_top_labels(pd.read_csv(p), album=album_filter)

    if not topics_df.empty:
        print_topics(topics_df, album=album_filter)
    elif primary_album:
        tp = album_output_paths(out_dir, primary_album)["topics"]
        if tp.exists() and run_topics:
            print_topics(ensure_album_column(pd.read_csv(tp)), album=album_filter)

    for album_name, data in agg_by_album.items():
        if album_filter and album_name != album_filter:
            continue
        tables = {}
        if df_labels is not None:
            sub = df_labels[df_labels[ALBUM_COL] == album_name]
            if not sub.empty:
                tables["labels_per_photo"] = sub
        if "label_counts" in data:
            lc = data["label_counts"]
            tables["all_labels"] = lc
            tables["top10_labels"] = top10_from_label_counts(lc)
        if not topics_df.empty:
            tsub = topics_df[topics_df[ALBUM_COL] == album_name]
            if not tsub.empty:
                tables["album_topics"] = tsub
        if not recs_df.empty and ALBUM_COL in recs_df.columns:
            rsub = recs_df[recs_df[ALBUM_COL] == album_name]
            if not rsub.empty:
                tables["recommendations"] = rsub
        if tables:
            write_summary_workbook(album_output_paths(out_dir, album_name)["summary"], tables)

    if primary_album:
        paths = album_output_paths(out_dir, primary_album)
        print(f"\nDone. Main album folder: {paths['folder'].resolve()}")
        print(
            f"  Files: {paths['album_file_prefix']}_labels_per_photo.csv, {paths['album_file_prefix']}_label_counts.csv, "
            f"{paths['album_file_prefix']}_category_means_long.csv, {paths['album_file_prefix']}_topics.csv, "
            f"{paths['album_file_prefix']}_recommendations.xlsx, {paths['album_file_prefix']}_summary.xlsx"
        )
    else:
        print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
