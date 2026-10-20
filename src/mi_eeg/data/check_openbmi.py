from pathlib import Path
import re
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/openbmi")
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2], choices=[1, 2])
    args = parser.parse_args()

    ROOT = Path(args.root)
    SESSIONS = args.sessions
    SUBJECTS = range(1, 55)

    pat = re.compile(r"sess(\d{2})_subj(\d{2})_EEG_MI\.mat$", re.I)
    files = list(ROOT.rglob("*EEG_MI*.mat"))

    found = defaultdict(set)
    bad_names = []
    tiny = []

    for f in files:
        m = pat.search(f.name)
        if not m:
            bad_names.append(str(f))
            continue
        sess = int(m.group(1))
        subj = int(m.group(2))
        found[sess].add(subj)

        # OpenBMI MI files are large, around 581 MB each.
        # Anything far smaller is almost certainly incomplete.
        if f.stat().st_size < 500_000_000:
            tiny.append((str(f), f.stat().st_size))

    missing = []
    for sess in SESSIONS:
        for subj in SUBJECTS:
            if subj not in found.get(sess, set()):
                missing.append(f"sess{sess:02d}_subj{subj:02d}_EEG_MI.mat")

    print("=== OpenBMI Integrity Check ===")
    print(f"Root: {ROOT.resolve()}")
    print(f"Sessions expected: {SESSIONS}")
    print(f"MI files found: {len(files)}")
    print(f"Expected MI files: {54 * len(SESSIONS)}")
    print(f"Missing files: {len(missing)}")
    print(f"Bad filename format: {len(bad_names)}")
    print(f"Tiny / likely incomplete files: {len(tiny)}")

    for sess in SESSIONS:
        print(f"Session {sess}: {len(found.get(sess, set()))}/54 subjects found")

    if missing[:30]:
        print("\nFirst missing files:")
        for x in missing[:30]:
            print(" ", x)

    if tiny[:30]:
        print("\nFirst tiny files:")
        for x, size in tiny[:30]:
            print(f"  {x} | {size / 1024 / 1024:.2f} MB")

    if bad_names[:20]:
        print("\nFirst bad-name files:")
        for x in bad_names[:20]:
            print(" ", x)

    print("\nDone.")


if __name__ == "__main__":
    main()
