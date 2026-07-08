import re

CHARS = " abcdefghijklmnopqrstuvwxyz'.,?!-"
stoi = {c: i + 1 for i, c in enumerate(CHARS)}
itos = {i: c for c, i in stoi.items()}
PAD = 0


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z'.,?! -]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode(text: str, max_len: int) -> list[int]:
    ids = [stoi.get(c, PAD) for c in normalize(text)[:max_len]]
    return ids + [PAD] * (max_len - len(ids))


if __name__ == "__main__":
    assert normalize("Hello,  WORLD!! 123") == "hello, world!!"
    assert encode("abc", 5) == [2, 3, 4, 0, 0]

