"""MinHash signatures over word 3-shingles, and a banded LSH index to find candidates.

Pr[min h(A) == min h(B)] = J(A, B) for a min-wise independent h, so the fraction of
equal components across NUM_PERM signatures estimates the Jaccard similarity. LSH splits
a signature into `bands` bands of `rows` rows; two items become candidates when any band
matches, with probability 1 - (1 - s^rows)^bands -- an S-curve around (1/bands)^(1/rows).
"""

import hashlib
import random
import re

NUM_PERM = 128
SHINGLE = 3
PRIME = (1 << 61) - 1          # Mersenne prime for h(x) = (a*x + b) mod p
MAX_HASH = PRIME               # signature value of an empty set

_TOKEN = re.compile(r"[a-z0-9]+")
_HEXISH = re.compile(r"^[0-9a-f]+$")
_DIGITS = re.compile(r"\d+")


def tokens(text, mask_numbers=False):
    """Lowercase alphanumeric words; with mask_numbers, every number (or hex id) becomes "0"."""
    toks = _TOKEN.findall(str(text).lower())
    if mask_numbers:
        toks = ["0" if _HEXISH.match(t) and any(c.isdigit() for c in t) else _DIGITS.sub("0", t)
                for t in toks]
    return toks


def shingles(text, k=SHINGLE, mask_numbers=False):
    """The set of k-word shingles; a text shorter than k words is one shingle."""
    toks = tokens(text, mask_numbers)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def jaccard(a, b):
    """Exact Jaccard of two sets, to check the estimate against."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _base_hash(shingle):
    """A stable 64-bit hash (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")


class MinHash:
    """NUM_PERM universal hash functions h_i(x) = (a_i*x + b_i) mod p, fixed by `seed`."""

    def __init__(self, num_perm=NUM_PERM, seed=580):
        rng = random.Random(seed)
        self.num_perm = num_perm
        self.params = [(rng.randrange(1, PRIME), rng.randrange(0, PRIME)) for _ in range(num_perm)]

    def signature(self, shingle_set):
        """Tuple of per-function minima; all MAX_HASH for an empty set."""
        if not shingle_set:
            return (MAX_HASH,) * self.num_perm
        xs = [_base_hash(s) % PRIME for s in shingle_set]
        return tuple(min((a * x + b) % PRIME for x in xs) for a, b in self.params)

    def of_text(self, text, mask_numbers=False):
        return self.signature(shingles(text, mask_numbers=mask_numbers))


def estimate(sig_a, sig_b):
    """Estimated Jaccard: the fraction of equal signature components."""
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


class LSHIndex:
    """Banded LSH over MinHash signatures: query returns candidates ranked by estimated Jaccard."""

    def __init__(self, bands=32, rows=4):
        self.bands, self.rows = bands, rows
        self.buckets = [{} for _ in range(bands)]
        self.signatures = {}

    def _bands(self, sig):
        if len(sig) != self.bands * self.rows:
            raise ValueError("signature length %d != bands*rows %d" % (len(sig), self.bands * self.rows))
        return [sig[i * self.rows:(i + 1) * self.rows] for i in range(self.bands)]

    def add(self, key, sig):
        self.signatures[key] = sig
        for table, band in zip(self.buckets, self._bands(sig)):
            table.setdefault(band, []).append(key)

    def candidates(self, sig):
        """Keys sharing at least one band with `sig`."""
        found = set()
        for table, band in zip(self.buckets, self._bands(sig)):
            found.update(table.get(band, ()))
        return found

    def query(self, sig, min_sim=0.0):
        """[(key, estimated Jaccard)] for candidates at or above min_sim, best first."""
        scored = [(key, estimate(sig, self.signatures[key])) for key in self.candidates(sig)]
        return sorted((pair for pair in scored if pair[1] >= min_sim), key=lambda p: (-p[1], str(p[0])))

    def __len__(self):
        return len(self.signatures)
