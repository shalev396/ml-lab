# 👗 Fashion Image Search

> Upload a photo of a clothing item, shoe or accessory and get the most visually similar products from an 8,000-item fashion catalog.

<p>
  <a href="https://huggingface.co/spaces/shalev396/fashion-image-search"><img alt="Space" src="https://img.shields.io/badge/🤗%20Space-live%20demo-blue"></a>
  <a href="https://huggingface.co/shalev396/fashion-image-search"><img alt="Model" src="https://img.shields.io/badge/🤗%20Model-shalev396%2Ffashion--image--search-yellow"></a>
  <a href="https://colab.research.google.com/github/shalev396/ml-lab/blob/main/fashion-image-search/training/notebook.ipynb"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

| | |
|---|---|
| **Task** | Content-based image retrieval (visual product search) |
| **Framework** | TensorFlow / Keras 3 (`backbone.keras`) + NumPy search |
| **Architecture** | Frozen ImageNet MobileNetV2, global average pool (1280-d) -> PCA 256-d -> exact cosine k-NN over 8,000 products (2,257,984 parameters, none trained) |
| **Dataset** | [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), 44k Myntra photos: 8,000 catalog / 500 val / 500 test queries |
| **Result** | Precision@5 (article type) = 0.728 · Precision@5 (master category) = 0.970 (test) |
| **Runs on** | Space: CPU basic · Training: CPU, GPU or Colab |

## The problem
An online shop wants a "find similar items" button: given a photo of a product, show the catalog
products that look most like it. There are no pairs of "similar" images to train on, so the project
uses a pretrained image network as a general-purpose visual descriptor and measures how often the
returned products are really of the same kind.

## The data
- Kaggle [`paramaggarwal/fashion-product-images-small`](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small):
  44,419 usable product photos from Myntra (60x80 px, white background) with `styles.csv` labels:
  gender, master category, sub category, article type, colour, product name.
- Master categories with < 20 products are dropped (only *Home*, 1 product), leaving 6.
- Stratified split by master category (seed 42): **8,000 catalog** products (Apparel 3,853, Accessories 2,031,
  Footwear 1,660, Personal Care 433, Free Items 19, Sporting Goods 4; 119 article types),
  **500 validation** and **500 test** query photos. Queries are never in the catalog.
- Downloaded anonymously with kagglehub and cached; CIFAR-10 is the coded fallback if Kaggle is unreachable.
  This dataset substitutes the original project brief's iMaterialist data, whose image URLs are dead.

## Architecture
`model/model.py` is the single source of truth:
1. **Preprocessing**: RGB, bilinear resize to 224x224; a `Rescaling(1/127.5, offset=-1)` layer inside
   `backbone.keras` does MobileNetV2's `[-1, 1]` scaling, so the saved model carries its own preprocessing.
2. **Embedder**: `keras.applications.MobileNetV2(include_top=False, pooling="avg", weights="imagenet")`,
   frozen, 2,257,984 parameters -> 1280-d features. Saved as `backbone.keras`, so the Space never
   downloads ImageNet weights.
3. **Projection**: PCA to 256-d, fit on the catalog embeddings without labels (`projection.npz`), then L2 normalization.
4. **Search**: exact cosine k-NN (dot product over L2-normalized float16 catalog embeddings, `argpartition`).
   The notebook cross-checks it against `sklearn.neighbors.NearestNeighbors(metric="cosine")` (max difference 5e-7).
5. **Output**: `{"matches": [{"id", "name", "master_category", "article_type", "score", "thumbnail"}]}`,
   thumbnails are JPEG data URIs stored in `catalog.parquet` (one file, not thousands of loose images).

## Training & experiments
No weights are trained: building the model = embedding the catalog + fitting PCA. Five representations
are compared with the same search. Selection rule: best **validation** Precision@10 (article type);
the test split is reported once.

| variant | dim | val P@10 (article type) | test P@5 (article type) | test P@10 (article type) | test P@5 (master cat.) |
|---|---|---|---|---|---|
| **mobilenetv2_pca256** (deployed) | 256 | 0.7090 | 0.7284 | 0.7022 | 0.9704 |
| mobilenetv2_pca128 | 128 | 0.7084 | 0.7252 | 0.7026 | 0.9720 |
| mobilenetv2 | 1,280 | 0.7038 | 0.7264 | 0.7012 | 0.9680 |
| pixels_32 (raw 32x32 RGB) | 3,072 | 0.5988 | 0.6392 | 0.6066 | 0.9384 |
| color_hist_8 (RGB histogram) | 512 | 0.2988 | 0.2868 | 0.2614 | 0.6904 |
| random ranking | – | – | 0.0537 | 0.0537 | 0.3423 |

The three MobileNetV2 variants are within noise of each other (500 queries); PCA-256 won the rule and makes
the index 5x smaller. The CNN features clearly beat the no-learning baselines, most of all on the
fine-grained article type. Graphs: [Precision@K curves](model/assets/precision_curves.png) ·
[variant comparison](model/assets/variant_comparison.png).

## Results
Test split (500 held-out photos), deployed `mobilenetv2_pca256` (from `model/metrics.json`):

| | P@1 | P@5 | P@10 |
|---|---|---|---|
| same master category | 0.978 | 0.970 | 0.969 |
| same article type | 0.752 | 0.728 | 0.702 |

<!-- per_category:start -->
Per master category (test P@5, article type; `metrics.json` -> `per_category`, n = test queries): Apparel 0.74 (n=241), Accessories 0.82 (n=127), Footwear 0.63 (n=103), Personal Care 0.57 (n=27), Free Items 1.00 (n=1), Sporting Goods 0.40 (n=1).
<!-- per_category:end -->
Footwear confuses visually close types (heels / flats / sandals). An earlier run of the legacy code
(full 1280-d embedding, different 500-query sample) reported P@5 0.977 / 0.718, consistent with these numbers.
See the [t-SNE of the catalog](model/assets/tsne_embeddings.png) and the [example queries](model/assets/query_examples.png).

Build time: 879 s on a desktop CPU that was shared with other training jobs (embedding 9,000 photos: 767 s);
the legacy run of the same pipeline on an idle machine took 256 s.

## Deployment
- **Model repo**: https://huggingface.co/shalev396/fashion-image-search (`backbone.keras`, `embeddings.npy`,
  `catalog.parquet`, `projection.npz`, `model.py`, `handler.py`)
- **Space / API**: https://huggingface.co/spaces/shalev396/fashion-image-search — `POST /gradio_api/call/predict`
  with `[image, k]` -> `[{"matches": [...]}, seconds, device]`; the gallery is rendered by a private follow-up event.
- **Inference Endpoint**: deploy the model repo from its page (Deploy → Inference Endpoints);
  `{"inputs": "<base64 image>", "parameters": {"k": 6}}`.

## Project structure
```
fashion-image-search/
├── README.md               this write-up
├── model/                  HF model repo: model.py, handler.py, backbone.keras, embeddings.npy, catalog.parquet,
│                           projection.npz, config.json, metrics.json, assets/, README.md (model card)
├── space/                  HF Space: app.py, space_utils.py, requirements.txt, examples/, README.md (API docs)
└── training/               notebook.ipynb (controller) + src/ (config, data_setup, model_builder, engine, export, utils)
```

## Reproduce
```bash
pip install -r requirements.txt                         # repo root
cd fashion-image-search/training
jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # full build, rewrites ../model
SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebook.ipynb --output-dir /tmp   # smoke run
cd ../space && python app.py                            # local Space, uses ../model
```
Or open the notebook in [Colab](https://colab.research.google.com/github/shalev396/ml-lab/blob/main/fashion-image-search/training/notebook.ipynb).
Details: [training/README.md](training/README.md).

## Limitations
- Fixed catalog of 8,000 studio product shots from one shop; street photos, several items in one picture or
  unusual angles match less well. New products need re-embedding (no incremental index yet).
- ImageNet features capture shape, texture and colour, not brand, style or price: "similar" means *visually* similar.
- Packaged products match packaging: the held-out watch-in-a-box example retrieves fragrance gift sets.
- Labels are the ground truth for Precision@K; near-duplicates with different labels (*Tshirts* vs *Tops*)
  count as misses, so the article-type number is a conservative estimate.
- Rare categories (Free Items, Sporting Goods) have almost no catalog items.
