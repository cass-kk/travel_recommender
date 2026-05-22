# WanderLens — Travel Photo Recommender

## Background

Travel planning often starts from what someone already enjoys. Using photos from past trips, weekends, and everyday moments can capture personal preferences for food, nightlife, culture, and the outdoors in a way that can be difficult to express. **WanderLens** uses vision AI to evaluate a personal photo album, extract visual interests, and match them to travel articles previously scraped from selected travel websites.

The project entry point is [`run_wanderlens.py`](run_wanderlens.py), which brings together the original Jupyter notebooks from group work [`group_files/`](group_files/).

## Objective

Turn a folder of travel photos into **ranked Texas travel article recommendations** by:

1. Labeling each photo with tags, travel-category scores, and scene attributes (vision API).
2. Aggregating tags and category means per album.
3. Discovering album-level themes with LDA.
4. Matching the album’s tag profile to listing text via TF-IDF and cosine similarity.

Outputs are CSV and Excel files under [`output/`](output/) (see committed example in [`output/sample_album/`](output/sample_album/) which uses 10 sample photos found in [`sample_photos/`](sample_photos/)).

## How to run (OpenAI or GitHub Models)

**Requirements:** Python 3.10+, dependencies from [`requirements.txt`](requirements.txt). See [`.env.example`](.env.example) for environment variable names.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### OpenAI (paid)

1. Create an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
2. In the same terminal session:

```powershell
$env:OPENAI_API_KEY = "sk-your-openai-key"
python run_wanderlens.py --provider openai --photos .\sample_images --name "Sample Album"
```

Default vision model: `gpt-4o`. Override with `--model gpt-4o`.

### GitHub Models (free, rate-limited)

1. Create a [Personal Access Token](https://github.com/settings/tokens) with **`models`** read access.
2. Pick a model id from the [GitHub Models marketplace](https://github.com/marketplace/models) (e.g. `openai/gpt-4o`).

```powershell
$env:GITHUB_MODELS_TOKEN = "ghp_your_pat"
python run_wanderlens.py --provider github --photos .\sample_images --name "Sample Album"
```

Uses the same `openai` Python SDK with base URL `https://models.github.ai/inference`. Do **not** put a GitHub PAT in `OPENAI_API_KEY`.

While testing on the free tier, limit labeling cost with `--max-photos 5` before labeling a large photo album.

### Auto-detect provider

If only `GITHUB_MODELS_TOKEN` is set, GitHub is used; otherwise OpenAI when `OPENAI_API_KEY` is set:

```powershell
python run_wanderlens.py --photos .\sample_images --name "Sample Album"
```

### Photo input

WanderLens can run with as few as one photo, but output quality improves with larger and more varied sets.
- 1-2 photos: limits to only labeling and recommendations, where results reflect individual scenes rather than an overall preference.
- 3-9 photos: the full pipeline runs, including topic extraction; however, topics are broad and recommendations depend heavily on a limited number of images.
- 10-20+ photos: the most stable results are produced as labels repeat across more images which strengthens topic modeling and representative recommendations.


| Layout | Example |
|--------|---------|
| Flat album | `--photos .\sample_images --name "My Album"` |
| Single image | `--photos .\trip.jpg --name "Summer 2024"` |
| Multiple albums (subfolders) | `--photos .\PicturesAlbum` (each subfolder = album name; `--name` ignored) |

Supported formats include `.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`, `.heif` (via `pillow-heif` in [`requirements.txt`](requirements.txt)).

### Useful flags

| Flag | Purpose |
|------|---------|
| `--only label` | Vision labeling only |
| `--only aggregate` | Tag counts and category means only |
| `--only topics` | LDA topics only |
| `--only recs` | Recommendations only |
| `--skip-labeling` | Reuse existing `*_labels_per_photo.csv` under `--output-dir` |
| `--items-csv` | Listings file (default: [`things_to_do_all.csv`](things_to_do_all.csv)) |
| `--top-n` | Number of recommendations per album (default: 5) |
| `--output-dir` | Output root (default: `./output`) |


## Dataset

### Travel Listings (recommendation catalog)

| File | Role |
|------|------|
| [`things_to_do_all.csv`](things_to_do_all.csv) | **Default** — Texas travel articles: `title`, `category`, image URLs, `detail_url` |
| [`things_to_do_all_world.csv`](things_to_do_all_world.csv) | Larger catalog; pass with `--items-csv` if desired |
| [`things_to_do_all_world_with_local_with_labels.csv`](things_to_do_all_world_with_local_with_labels.csv) | Extended world catalog with label columns |

Matching uses **text from the CSV** (title, category, optional description columns, light tokens from hero-image filenames). Image URLs are not downloaded at recommendation time.

### Group Files

[`group_files/`](group_files/) holds intermediate CSVs and per-person topic files from the original pipeline with data from groupmates' photo albums (e.g. [`group_files/labels_per_photo.csv`](group_files/labels_per_photo.csv), [`group_files/category_means_per_person_long.csv`](group_files/category_means_per_person_long.csv), [`group_files/lda_results_per_person.csv`](group_files/lda_results_per_person.csv)).

### Example run outputs

[`output/sample_album/`](output/sample_album/) contains a full results for the 10 sample photos in [`sample_images/`](sample_images/) album (labels, counts, topics, recommendations, summary workbook) using the Texas travel recommendations dataset.

## Folder structure

```
travel_recommender/
├── run_wanderlens.py              # Main
├── requirements.txt
├── .env.example
├── README.md
│
├── things_to_do_all.csv           # Default travel listings
├── things_to_do_all_world.csv
├── things_to_do_all_world_with_local_with_labels.csv
│
├── sample_images/                 # Example photos
├── output/
│   └── sample_album/              # Sample photo outputs
│
├── group_files/                   # Original notebooks + outputs
│   ├── Photo Tagger_orig.ipynb
│   ├── Final_Project_LDA_orig.ipynb
│   ├── travel_recommender_code_orig.ipynb
│   ├── Recommender Analysis Code_orig.ipynb
│   └── … (labels, topics, listings variants)
│
├── WanderLens_pptx.pdf            # Project presentation
└── WanderLens PPT.pptx
```

## Tools and packages

| Layer | Packages | Role |
|-------|----------|------|
| Core | `pandas`, `numpy`, `tqdm` | Data handling and progress |
| Vision | `Pillow`, `pillow-heif`, `openai`, `httpx` | Image load + OpenAI / GitHub Models API |
| Topics | `nltk`, `gensim` | Lemmatization, LDA per album |
| Recommendations | `scikit-learn`, `openpyxl`, `xlsxwriter` | TF-IDF, cosine similarity, Excel export |

Install once: `python -m pip install -r requirements.txt`.

**API keys:** labeling requires `OPENAI_API_KEY` or `GITHUB_MODELS_TOKEN`. LDA and recommendations run offline after labels exist.

## Modeling notes

Pipeline stages in [`run_wanderlens.py`](run_wanderlens.py) map to the notebooks found in [`group_files/`](group_files/):

| Stage | Method | Prototype notebook |
|-------|--------|-------------------|
| **1. Labeling** | GPT-4o vision with a fixed JSON schema: 5–12 tags, eight travel categories (probabilities), indoor/outdoor, day/night, season | [`group_files/Photo Tagger_orig.ipynb`](group_files/Photo%20Tagger_orig.ipynb) |
| **2. Aggregate** | Tag frequency counts; mean category probabilities per album | [`group_files/Photo Tagger_orig.ipynb`](group_files/Photo%20Tagger_orig.ipynb) |
| **3. LDA topics** | Tokenize/lemmatize labels; `gensim` LDA per album (3–5 topics, `random_state=42`, 10 passes); needs ≥3 labeled photos | [`group_files/Final_Project_LDA_orig.ipynb`](group_files/Final_Project_LDA_orig.ipynb) |
| **4. Recommendations** | User profile = tag counts expanded to weighted text; listings = concatenated title/category/(optional) body; **TF-IDF** (unigrams + bigrams) + **cosine similarity**; top-N with overlap “why” terms | [`group_files/travel_recommender_code_orig.ipynb`](group_files/travel_recommender_code_orig.ipynb) |

**Travel categories** (vision schema): Outdoor Activities, Food & Drink, Cultural, Night Life, Concerts & Shows, Casino & Gambling, Attractions, Sporting Events.

**Design choices:** recommendations are intentionally lightweight (bag-of-words + TF-IDF) for interpretability and fast iteration; LDA topics are exploratory context and do not directly score listings. For analysis of group-level results, see [`group_files/Recommender Analysis Code_orig.ipynb`](group_files/Recommender%20Analysis%20Code_orig.ipynb).
