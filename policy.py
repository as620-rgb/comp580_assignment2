"""Task-level reuse: answer an incoming task from a verified record when it is the same item.

A verified record's answer is reused only when all three hold:
  1. Key: the visible fields each dataset's INSTRUCTIONS.md names under "Validation"
     are equal after normalisation (necessary, never sufficient on its own).
  2. Similarity: the MinHash estimate of 3-shingle Jaccard reaches the dataset's
     threshold. The threshold is calibrated on history.jsonl alone -- `margin` above the
     most similar pair of records that share a key but not an answer, never below
     `floor` -- and rises again if a later verified task reveals such a pair.
  3. Agreement: every record passing 1 and 2 carries the same answer; otherwise abstain.
Candidates come from the LSH index; anything else runs the agent.
"""

import json
import os
import re

import minhash

TEXT_FIELDS = {"tickets": ("subject", "body"), "bugs": ("title", "body", "stacktrace"),
               "docs": ("title", "body"), "citations": ("raw",)}
DEFAULT_TEXT = ("subject", "body")  # corpus requests

DOC_ID = re.compile(r"\b(CHG|INC|POL)-\d{3,4}\b|\bREL-\d+\.\d+\b", re.I)
DOC_CLASSES = {  # status words per class, from docs/INSTRUCTIONS.md, as inflection stems
    "CHG": {"request": r"request|pending", "approved": r"approv|authori[sz]",
            "rejected": r"reject|declin"},
    "INC": {"open": r"ongoing|under investigation", "resolved": r"resolv|closed"},
    "POL": {"draft": r"draft|for comment", "final": r"effective|in force"},
}


def _alnum(value):
    return re.sub(r"[^a-z0-9]", "", str(value if value is not None else "").lower())


def _words(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value if value is not None else "").lower()))


def _frame(frame):
    """lattice/net/http.py::fetch, lattice.net.http.fetch and ...fetch:12 are one frame."""
    frame = re.sub(r"\.py::", ".", str(frame).strip().lower())
    return re.sub(r":\d+$", "", frame.replace("/", ".").replace("::", "."))


def _series(version):
    match = re.search(r"(\d+)\.(\d+)", str(version))
    return "%s.%s" % match.groups() if match else _words(version)


def _doi(doi):
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", str(doi or "").strip().lower())
    return re.sub(r"^doi:\s*", "", doi).rstrip(". ")


def _author(name):
    """'Priya Nair', 'Nair, Priya' and 'P. Nair' -> ('nair', 'p'); 'et al.' stays itself."""
    name = str(name).strip()
    if _alnum(name) == "etal":
        return ("etal", "")
    if "," in name:
        surname, given = name.split(",", 1)
    else:
        parts = name.split()
        surname, given = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (name, "")
    return (_words(surname), _words(given)[:1])


def ticket_key(record):
    return (_words(record.get("product")), _alnum(record.get("error_code")),
            _words(record.get("platform")))


def bug_key(record):
    return (_words(record.get("exception")), _frame(record.get("top_frame")),
            _series(record.get("version")))


def doc_key(record):
    """(identifier, status class); None when the class cannot be read unambiguously."""
    match = DOC_ID.search(str(record.get("title", "")))
    if not match:
        return None
    ident = match.group(0).upper()
    family = ident.split("-")[0]
    if family == "REL":
        return (ident, "release")
    body = str(record.get("body", "")).lower()
    found = [name for name, stems in DOC_CLASSES[family].items()
             if re.search(r"\b(%s)" % stems, body)]
    return (ident, found[0]) if len(found) == 1 else None


def citation_key(record, venues):
    """All five graded values: DOI, title, author pairs, venue, year."""
    cited = record.get("citation") or {}
    venue = _alnum(cited.get("venue"))
    return (_doi(cited.get("doi")), _words(cited.get("title")),
            tuple(_author(a) for a in cited.get("authors") or ()),
            venues.get(venue, venue), str(cited.get("year", "")).strip())


def request_key(record):
    return (_alnum(record.get("scenario")), _words(record.get("kind")),
            _words(record.get("target")))


def history_answer(dataset, record):
    """The verified answer a history record carries, in answer.json form."""
    if dataset == "tickets":
        return {"resolution_id": record["resolution_id"]}
    if dataset == "bugs":
        return {"component": record["component"], "root_cause": record["root_cause"]}
    if dataset == "docs":
        return {"category": record["category"], "duplicate_of": record["hid"]}
    if dataset == "citations":
        return record["verdict"]
    return record["answer"]


def _venues(env_dir):
    """Every venue name and abbreviation -> one canonical spelling, from db.json."""
    path = os.path.join(env_dir, "db.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        venues = json.load(handle).get("venues", [])
    return {_alnum(name): _alnum(v["abbrev"][0] if v.get("abbrev") else v["name"])
            for v in venues for name in [v["name"]] + list(v.get("abbrev") or [])}


def _canon(answer):
    return json.dumps(answer, sort_keys=True)


class TaskIndex:
    def __init__(self, dataset, env_dir, threshold=None, floor=0.3, margin=0.1,
                 bands=64, rows=2, seed=580):
        """Index every verified record in env_dir/history.jsonl.

        threshold: a fixed similarity threshold (disables calibration), for sweeps.
        floor, margin: bounds of the calibrated threshold.
        bands, rows: LSH shape; 64x2 retrieves pairs down to J~0.2 with high probability.
        """
        self.dataset = dataset
        self.fields = TEXT_FIELDS.get(dataset, DEFAULT_TEXT)
        if dataset == "citations":
            venues = _venues(env_dir)
            self.key = lambda record: citation_key(record, venues)
        else:
            self.key = {"tickets": ticket_key, "bugs": bug_key,
                        "docs": doc_key}.get(dataset, request_key)
        self.minhash = minhash.MinHash(seed=seed)
        self.lsh = minhash.LSHIndex(bands=bands, rows=rows)
        self.items = {}    # id -> (key, signature, answer)
        self.by_key = {}   # key -> [id, ...], for calibration
        self.fixed = threshold
        self.floor, self.margin = floor, margin
        self.conflict = 0.0  # highest similarity seen between same-key, different-answer records
        self.last = None     # (matched id, similarity) of the last reuse, for the run log

        with open(os.path.join(env_dir, "history.jsonl"), encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    self._insert(record["hid"], record, history_answer(dataset, record))

    @property
    def threshold(self):
        if self.fixed is not None:
            return self.fixed
        return min(1.0, max(self.floor, self.conflict + self.margin))

    def _text(self, record):
        return "\n".join(str(record.get(field, "")) for field in self.fields)

    def _insert(self, ident, record, answer):
        key = self.key(record)
        if key is None:
            return
        sig = self.minhash.of_text(self._text(record))
        for other in self.by_key.get(key, ()):
            _, other_sig, other_answer = self.items[other]
            if _canon(other_answer) != _canon(answer):
                self.conflict = max(self.conflict, minhash.estimate(sig, other_sig))
        self.items[ident] = (key, sig, answer)
        self.by_key.setdefault(key, []).append(ident)
        self.lsh.add(ident, sig)

    def lookup(self, task):
        """Return a verified answer dict to reuse, or None to run the agent."""
        self.last = None
        key = self.key(task)
        if key is None:
            return None
        sig = self.minhash.of_text(self._text(task))
        hits = [(ident, sim) for ident, sim in self.lsh.query(sig, self.threshold)
                if self.items[ident][0] == key]
        if not hits or len({_canon(self.items[ident][2]) for ident, _ in hits}) != 1:
            return None
        self.last = hits[0]
        return json.loads(_canon(self.items[hits[0][0]][2]))  # a copy the caller may keep

    def add(self, task, answer):
        """Index a task whose answer just passed verification."""
        if isinstance(answer, dict):
            self._insert(task["tid"], task, answer)
