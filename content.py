"""Content-level reuse: fold a tool output that repeats an earlier one into an exact diff.

Within one trajectory, every tool output sent to the model in full is indexed by its
MinHash signature over 3-shingles with numbers masked, so copies that differ only in
values (ports, versions, timestamps) still find each other. A new output is folded when:
  1. LSH finds an earlier full output with estimated Jaccard >= threshold;
  2. the line diff against it is exact -- every line not shown is identical to the
     reference, so each changed number, identifier and setting appears verbatim;
  3. the fold saves at least `min_savings` of the output, which is >= `min_chars` long.
Otherwise the original output is returned. Only full outputs become references, so a
fold never points at another fold.
"""

import difflib
import re

import minhash

NUMBERED = re.compile(r"^(\d+)\t(.*)$")  # read_file lines are "N<TAB>text"


def lines_of(output):
    """The text lines of an output, without read_file's line-number prefix."""
    lines = output.split("\n")
    parsed = [NUMBERED.match(line) for line in lines]
    numbered = [m for m in parsed if m]
    # A truncated read_file ends in one unnumbered trailer line; keep it as text.
    if numbered and len(numbered) >= len(lines) - 1 and \
            all(int(m.group(1)) == i for i, m in enumerate(numbered, 1)):
        return [m.group(2) if m else line for m, line in zip(parsed, lines)]
    return lines


def diff(ref, new):
    """Hunks (i1, i2, j1, j2): ref[i1:i2] is replaced by new[j1:j2]; all else is equal."""
    matcher = difflib.SequenceMatcher(None, ref, new, autojunk=False)
    return [(i1, i2, j1, j2) for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal"]


def apply(ref, new_parts):
    """Rebuild the new lines from ref and [(i1, i2, replacement lines)]."""
    out, cursor = [], 0
    for i1, i2, lines in new_parts:
        out += ref[cursor:i1] + lines
        cursor = i2
    return out + ref[cursor:]


def _label(tool, args):
    if tool == "read_file" and "path" in args:
        return args["path"]
    return "%s(%s)" % (tool, ", ".join("%s=%r" % kv for kv in sorted(args.items())))


class ContentCache:
    def __init__(self, threshold=0.5, min_savings=0.3, min_chars=400, candidates=3,
                 show_old=False, bands=32, rows=4, seed=580):
        """One cache per task trajectory; threshold applies to the number-masked Jaccard.

        show_old: also print the replaced lines ('-'), which the model has already seen.
        """
        self.threshold, self.min_savings, self.show_old = threshold, min_savings, show_old
        self.min_chars, self.candidates = min_chars, candidates
        self.minhash = minhash.MinHash(seed=seed)
        self.lsh = minhash.LSHIndex(bands=bands, rows=rows)
        self.seen = []  # (label, lines, output) of outputs sent in full
        self.folds = 0

    def process(self, tool, args, output):
        """Return the text the model sees: `output` itself, or a [reuse] fold of it."""
        if len(output) < self.min_chars or output.startswith("ERROR"):
            return output
        label, lines = _label(tool, args), lines_of(output)
        sig = self.minhash.signature(minhash.shingles("\n".join(lines), mask_numbers=True))
        folds = [self.fold(label, lines, output, *self.seen[n])
                 for n, _ in self.lsh.query(sig, self.threshold)[:self.candidates]]
        folds = [f for f in folds if f is not None]
        best = min(folds, key=len, default=None)
        if best is not None and len(best) <= (1 - self.min_savings) * len(output):
            self.folds += 1
            return best
        self.lsh.add(len(self.seen), sig)
        self.seen.append((label, lines, output))
        return output

    def fold(self, label, lines, output, ref_label, ref_lines, ref_output):
        """The [reuse] text for `output` against an earlier full output, or None."""
        if output == ref_output:
            return "[reuse] %s: identical to %s, shown earlier." % (label, ref_label)
        hunks = diff(ref_lines, lines)
        if apply(ref_lines, [(i1, i2, lines[j1:j2]) for i1, i2, j1, j2 in hunks]) != lines:
            return None  # never send a fold that does not rebuild the output exactly
        text = ["[reuse] %s = %s (shown earlier) with the changes below; every other line is "
                "identical. '@@ -a,n +b,m' replaces the n lines of %s from line a with the m "
                "lines that follow, which are lines b.. of %s." % (label, ref_label, ref_label, label)]
        for i1, i2, j1, j2 in hunks:
            text.append("@@ -%d,%d +%d,%d" % (i1 + 1, i2 - i1, j1 + 1, j2 - j1))
            if self.show_old:
                text += ["- " + line for line in ref_lines[i1:i2]]
            text += ["+ " + line for line in lines[j1:j2]]
        return "\n".join(text)
