from pathlib import Path
import re
from collections import defaultdict, Counter


def main():
    ROOT = Path("data/raw/physionet")

    EXPECTED_SUBJECTS = range(1, 110)
    EXPECTED_RUNS = range(1, 15)
    MI_RUNS = {4, 6, 8, 10, 12, 14}
    EXCLUDED_COMMON = {88, 89, 92, 100}

    pat = re.compile(r"S(\d{3})R(\d{2})\.edf$", re.I)
    files = list(ROOT.rglob("*.edf"))

    found = defaultdict(set)
    bad_names = []

    for f in files:
        m = pat.search(f.name)
        if not m:
            bad_names.append(str(f))
            continue
        s = int(m.group(1))
        r = int(m.group(2))
        found[s].add(r)

    missing = []
    for s in EXPECTED_SUBJECTS:
        for r in EXPECTED_RUNS:
            if r not in found.get(s, set()):
                missing.append(f"S{s:03d}R{r:02d}.edf")

    mi_missing = []
    for s in EXPECTED_SUBJECTS:
        for r in MI_RUNS:
            if r not in found.get(s, set()):
                mi_missing.append(f"S{s:03d}R{r:02d}.edf")

    zero_or_tiny = [str(f) for f in files if f.stat().st_size < 100_000]

    print("=== PhysioNet EEGMMIDB Integrity Check ===")
    print(f"Root: {ROOT.resolve()}")
    print(f"EDF files found: {len(files)}")
    print("Expected full EDF files: 109 subjects * 14 runs = 1526")
    print(f"Subjects found: {len(found)}")
    print(f"Missing full files: {len(missing)}")
    print(f"Missing MI files only: {len(mi_missing)}")
    print(f"Bad filename format: {len(bad_names)}")
    print(f"Zero/tiny files: {len(zero_or_tiny)}")

    usable_subjects = [
        s for s in EXPECTED_SUBJECTS
        if s not in EXCLUDED_COMMON and all(r in found.get(s, set()) for r in MI_RUNS)
    ]
    print()
    print(f"Usable subjects after excluding {sorted(EXCLUDED_COMMON)} and requiring MI runs: {len(usable_subjects)}")
    print("Expected if all normal: 105")

    if missing[:20]:
        print("\nFirst missing files:")
        for x in missing[:20]:
            print(" ", x)

    if mi_missing[:20]:
        print("\nFirst missing MI files:")
        for x in mi_missing[:20]:
            print(" ", x)

    if zero_or_tiny[:20]:
        print("\nFirst zero/tiny files:")
        for x in zero_or_tiny[:20]:
            print(" ", x)

    print("\nDone.")


if __name__ == "__main__":
    main()
