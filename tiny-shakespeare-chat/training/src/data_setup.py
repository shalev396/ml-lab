"""Data: download -> cache in training/data -> char tokenizer -> pretrain and chat-tune splits.

Corpus: Tiny Shakespeare (karpathy/char-rnn), 1,115,394 characters of Shakespeare plays.
- Stage A (pretrain): the raw text, encoded char by char, split 90/10 (the last 10 % is val).
- Stage B (chat-tune): the play parsed into consecutive (speaker, utterance) turns. Every pair of
  neighbouring turns becomes `<|user|> {line_i} <|end|> <|bot|> {line_i+1} <|end|>`, plus a variant
  whose reply starts with the speaker name. The samples are shuffled (seeded), joined with newlines,
  and the last 5 % of that stream is val.
"""
import random
import shutil
import urllib.request

import torch

import model as M  # ../model/model.py, put on sys.path by src/__init__.py

from .config import Config


def download_corpus(cfg: Config) -> str:
    """Fetch the corpus once (primary URL, then the fallback); later calls read the cache."""
    path = cfg.corpus_path
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        errors = []
        for url in (cfg.data_url, cfg.data_fallback_url):
            try:
                print(f"downloading {url}")
                with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as f:
                    shutil.copyfileobj(response, f)
                partial.replace(path)
                break
            except OSError as err:
                errors.append(f"{url}: {err}")
        else:
            raise RuntimeError("could not download Tiny Shakespeare:\n  " + "\n  ".join(errors))
    return path.read_text(encoding="utf-8")


def build_tokenizer(text: str) -> M.CharTokenizer:
    """Sorted set of corpus characters (65) + the 3 chat markers = 68 token ids."""
    return M.CharTokenizer.from_text(text)


def make_pretrain_split(cfg: Config, tok: M.CharTokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(train_ids, val_ids) of the raw play."""
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n_val = int(len(ids) * cfg.val_frac)
    return ids[:-n_val], ids[-n_val:]


def parse_dialogue(text: str) -> list[tuple[str, str]]:
    """Consecutive (speaker, utterance) turns. Each turn is a blank-line separated block such as
    "First Citizen:" followed by the spoken lines."""
    turns = []
    for block in text.split("\n\n"):
        lines = [line for line in block.strip().split("\n") if line.strip()]
        if len(lines) >= 2 and lines[0].endswith(":") and len(lines[0]) <= 40:
            speaker, utterance = lines[0][:-1].strip(), "\n".join(lines[1:]).strip()
            if speaker and utterance:
                turns.append((speaker, utterance))
    return turns


def build_chat_samples(turns: list[tuple[str, str]], seed: int) -> list[str]:
    """Neighbouring turns -> chat samples in two variants (plain reply, "SPEAKER: reply"), shuffled."""
    samples = []
    for (_, user_line), (bot_speaker, bot_line) in zip(turns, turns[1:]):
        samples.append(M.format_pair(user_line, bot_line))
        samples.append(M.format_pair(user_line, f"{bot_speaker}: {bot_line}"))
    random.Random(seed).shuffle(samples)
    return samples


def make_chat_split(cfg: Config, tok: M.CharTokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(train_ids, val_ids, n_pairs) of the chat-formatted stream."""
    turns = parse_dialogue(text)
    samples = build_chat_samples(turns, cfg.seed)
    ids = torch.tensor(tok.encode("\n".join(samples)), dtype=torch.long)
    n_val = max(cfg.block_size + 2, int(len(ids) * cfg.chat_val_frac))
    return ids[:-n_val], ids[-n_val:], len(samples) // 2


def chat_val_overlap(cfg: Config, tok: M.CharTokenizer, text: str) -> float:
    """Share of chat-val samples whose dialogue pair is ALSO in chat-train (as the other variant).

    The split is taken after the two variants of every pair are shuffled together, so most val
    samples have a twin in train: chat val loss is therefore not a clean held-out estimate.
    """
    turns = parse_dialogue(text)
    tagged = [(i, v) for i in range(len(turns) - 1) for v in (0, 1)]
    random.Random(cfg.seed).shuffle(tagged)       # same permutation as build_chat_samples
    samples = build_chat_samples(turns, cfg.seed)
    lengths = [len(tok.encode(s)) + 1 for s in samples]   # +1 for the joining newline
    n_ids = sum(lengths) - 1
    n_val = max(cfg.block_size + 2, int(n_ids * cfg.chat_val_frac))
    pos, train_pairs, val_pairs = 0, set(), []
    for (pair, _), length in zip(tagged, lengths):
        if pos + length <= n_ids - n_val:
            train_pairs.add(pair)
        else:
            val_pairs.append(pair)
        pos += length
    return sum(p in train_pairs for p in val_pairs) / max(1, len(val_pairs))


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random contiguous windows -> (x, y), y shifted by one character."""
    if len(data) <= block_size + 1:
        raise ValueError(f"split too small ({len(data)} tokens) for block_size={block_size}")
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
