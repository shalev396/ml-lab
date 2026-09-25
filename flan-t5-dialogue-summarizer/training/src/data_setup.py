"""DialogSum: download -> cache in training/data -> seeded subsets -> tokenized DataLoaders.

Primary source: the Hub dataset `knkarthick/dialogsum`, cached in `training/data/hf_datasets`.
Fallback: the authors' JSONL files on GitHub (github.com/cylnlp/dialogsum), same dialogues.
The test split has 500 dialogues with 3 human summaries each; they are grouped per dialogue so
every prediction is scored against all three.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

import model as M  # ../model/model.py (put on sys.path by src/__init__.py)

from . import utils
from .config import Config

DATASET_ID = "knkarthick/dialogsum"
CACHE_DIR = utils.DATA_DIR / "hf_datasets"
MIRROR = "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.{split}.jsonl"
KB_PATH = utils.MODEL_DIR / M.KB_FILE   # the knowledge base ships with the model repo

# Held-out questions for the retriever: question -> id of the knowledge-base doc that answers it.
RAG_EVAL = [
    ("How long does standard shipping take and what does it cost?", "ship-01"),
    ("Can I get my package tomorrow?", "ship-02"),
    ("Do you deliver to other countries?", "ship-03"),
    ("Where can I see where my order is?", "ship-04"),
    ("How many days do I have to send something back?", "ret-01"),
    ("When will I get my money back after a return?", "ret-03"),
    ("Can I return earbuds after opening them?", "ret-05"),
    ("My laptop arrived with a cracked screen, what do I do?", "ret-07"),
    ("What does the NovaCare+ plan cover?", "war-02"),
    ("How much RAM does the NovaBook Air have?", "prod-01"),
    ("Is the NovaPhone X2 waterproof?", "prod-03"),
    ("How long do the NovaBuds Pro last on a charge?", "prod-04"),
    ("Can I pay in installments?", "pay-02"),
    ("Do gift cards expire?", "pay-03"),
    ("Can I cancel an order I just placed?", "acc-01"),
    ("Do students get a discount?", "acc-03"),
    ("When is phone support open?", "sup-01"),
    ("Can I trade in my old phone?", "sup-03"),
]


@dataclass
class Splits:
    train: Dataset   # columns: dialogue, summary
    val: Dataset     # columns: dialogue, summary
    test: Dataset    # columns: dialogue, references (list of 3 summaries)

    def sizes(self) -> dict[str, int]:
        return {"n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test)}


# --------------------------------------------------------------------------- download
def load_dialogsum() -> DatasetDict:
    """train / validation / test with columns id, dialogue, summary, topic (cached after the first call)."""
    try:
        ds = load_dataset(DATASET_ID, cache_dir=str(CACHE_DIR))
    except Exception as err:  # Hub unreachable -> the authors' JSONL files
        print(f"[data] {DATASET_ID} failed ({type(err).__name__}: {err}); using the GitHub mirror")
        ds = _load_mirror()
    print("[data] DialogSum:", {k: len(v) for k, v in ds.items()})
    return ds


def _load_mirror() -> DatasetDict:
    out = {}
    for split, name in (("train", "train"), ("validation", "dev"), ("test", "test")):
        path = utils.DATA_DIR / "mirror" / f"dialogsum.{name}.jsonl"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MIRROR.format(split=name), path)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "summary" in r:
                rows.append({"id": r["fname"], "dialogue": r["dialogue"], "summary": r["summary"], "topic": r["topic"]})
            else:  # test: summary1..3 -> one row per reference, like the Hub version
                rows += [{"id": f"{r['fname']}_{i}", "dialogue": r["dialogue"], "summary": r[f"summary{i}"],
                          "topic": r[f"topic{i}"]} for i in (1, 2, 3)]
        out[split] = Dataset.from_list(rows)
    return DatasetDict(out)


# --------------------------------------------------------------------------- splits
def group_references(test: Dataset) -> Dataset:
    """One row per unique test dialogue, with all of its human summaries in `references`."""
    refs: dict[str, list[str]] = {}
    for dialogue, summary in zip(test["dialogue"], test["summary"]):
        refs.setdefault(dialogue, []).append(summary)
    return Dataset.from_dict({"dialogue": list(refs), "references": list(refs.values())})


def make_splits(ds: DatasetDict, cfg: Config) -> Splits:
    """Seeded subsets: train (fine-tuning), val (loss per epoch), test (ROUGE, used once at the end)."""
    def subset(d: Dataset, n: int) -> Dataset:
        return d.shuffle(seed=cfg.seed).select(range(min(n, len(d))))

    keep = ["dialogue", "summary"]
    train = subset(ds["train"], cfg.train_samples).select_columns(keep)
    val = subset(ds["validation"], cfg.val_samples).select_columns(keep)
    test = subset(group_references(ds["test"]), cfg.test_samples)
    splits = Splits(train, val, test)
    print("[data] subsets:", splits.sizes())
    return splits


def pick_shots(train: Dataset, cfg: Config) -> list[tuple[str, str]]:
    """In-context examples for one/few-shot prompting: the first k short dialogues of the train subset."""
    shots = [(d, s) for d, s in zip(train["dialogue"], train["summary"]) if len(d) <= cfg.shot_max_chars]
    if len(shots) < cfg.few_shot_k:  # tiny smoke subsets: fall back to the shortest ones
        order = sorted(range(len(train)), key=lambda i: len(train[i]["dialogue"]))
        shots = [(train[i]["dialogue"], train[i]["summary"]) for i in order]
    return shots[:cfg.few_shot_k]


def few_shot_prompt(dialogue: str, shots: list[tuple[str, str]]) -> str:
    """k solved examples, each in the exact training template, then the query dialogue."""
    return "\n\n".join([M.build_prompt(d) + s for d, s in shots] + [M.build_prompt(dialogue)])


# --------------------------------------------------------------------------- tokenization
def tokenize(dataset: Dataset, tokenizer, cfg: Config) -> Dataset:
    """Prompt -> input_ids (truncated, unpadded); summary -> labels. Padding happens per batch."""
    def _encode(batch):
        enc = tokenizer([M.build_prompt(d) for d in batch["dialogue"]], max_length=cfg.max_input_length,
                        truncation=True)
        enc["labels"] = tokenizer(text_target=batch["summary"], max_length=cfg.max_target_length,
                                  truncation=True)["input_ids"]
        return enc

    return dataset.map(_encode, batched=True, remove_columns=dataset.column_names)


def make_loaders(splits: Splits, tokenizer, cfg: Config) -> tuple[DataLoader, DataLoader]:
    """Dynamic padding (DataCollatorForSeq2Seq, label padding = -100): no compute wasted on pad tokens."""
    collate = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, return_tensors="pt")
    train = DataLoader(tokenize(splits.train, tokenizer, cfg), batch_size=cfg.batch_size, shuffle=True,
                       collate_fn=collate, num_workers=utils.num_workers())
    val = DataLoader(tokenize(splits.val, tokenizer, cfg), batch_size=cfg.eval_batch_size, shuffle=False,
                     collate_fn=collate, num_workers=utils.num_workers())
    return train, val


# --------------------------------------------------------------------------- RAG knowledge base
def load_kb() -> list[dict]:
    """The 41-document knowledge base of the fictional "Nova Gadgets" store (model/assets/kb.json)."""
    return utils.load_json(KB_PATH)["docs"]
