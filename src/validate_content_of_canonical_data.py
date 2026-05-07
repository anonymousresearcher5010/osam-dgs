import json
from pathlib import Path
from collections import Counter

FILE_A = Path("data/canonical_dataset/OSAM-DGS_canonical_dataset.json")
FILE_B = Path("data/canonical_dataset/OSAM-DGS_canonical_dataset_bt.json")
MAX_MISMATCH_EXAMPLES = 20


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _add_mismatch(mismatches, kind, path, a_value=None, b_value=None):
    mismatches.append(
        {
            "type": kind,
            "path": path,
            "a": a_value,
            "b": b_value,
        }
    )


def compare(a, b, path="root", stats=None, mismatches=None):
    if stats is None:
        stats = Counter()
    if mismatches is None:
        mismatches = []

    # Dicts
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys()) - {"back_translations"}
        keys_b = set(b.keys()) - {"back_translations"}

        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        for k in only_a:
            stats["missing_in_b"] += 1
            _add_mismatch(mismatches, "missing_in_b", f"{path}.{k}", a.get(k), None)
        for k in only_b:
            stats["missing_in_a"] += 1
            _add_mismatch(mismatches, "missing_in_a", f"{path}.{k}", None, b.get(k))

        for k in keys_a & keys_b:
            compare(a[k], b[k], f"{path}.{k}", stats, mismatches)
        return stats, mismatches

    # Lists
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            stats["list_length_mismatch"] += 1
            _add_mismatch(mismatches, "list_length_mismatch", path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", stats, mismatches)
        return stats, mismatches

    # Primitive values (and type mismatches that are not both dict/list)
    if a != b:
        stats["value_mismatch"] += 1
        _add_mismatch(mismatches, "value_mismatch", path, a, b)

    return stats, mismatches


if __name__ == "__main__":
    data_a = load_json(FILE_A)
    data_b = load_json(FILE_B)

    result, mismatches = compare(data_a, data_b)
    total = sum(result.values())

    print("Differences excluding 'backtranslations':")
    print(f"  total: {total}")
    for k, v in sorted(result.items()):
        print(f"  {k}: {v}")

    print(f"\nFirst {MAX_MISMATCH_EXAMPLES} mismatches:")
    for i, m in enumerate(mismatches[:MAX_MISMATCH_EXAMPLES], start=1):
        print(f"{i:>2}. [{m['type']}] {m['path']}")
        print(f"    A: {m['a']!r}")
        print(f"    B: {m['b']!r}")