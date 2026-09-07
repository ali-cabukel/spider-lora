"""Fetch Spider into data/spider/.

Execution accuracy needs the actual SQLite files, not just the query pairs, so
this pulls two things:

  1. train_spider.json / dev.json  -- question + gold SQL + db_id
  2. database/<db_id>/<db_id>.sqlite -- the databases queries run against

The official Yale page (yale-lily.github.io/spider) is gone, and the HF
dataset `xlangai/spider` ships only the question/SQL pairs. The SQLite files
are re-hosted as `spider_data.zip` on `HAL-9001/spider-databases`.

    python scripts/get_spider.py
    python scripts/get_spider.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spider"
DB_ROOT = DATA / "database"

# Re-host of the original Yale spider_data.zip (CC-BY-SA-4.0).
HF_DB_REPO = "HAL-9001/spider-databases"
HF_DB_FILE = "spider_data.zip"
HF_DB_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
HF_DB_URL = f"https://huggingface.co/datasets/{HF_DB_REPO}/resolve/main/{HF_DB_FILE}"

MANUAL = f"""
Could not fetch the Spider databases automatically.

Manual steps:
  1. Download spider_data.zip from:
       {HF_DB_URL}
     (HF dataset page: https://huggingface.co/datasets/{HF_DB_REPO})
  2. Unzip it. The archive contains spider_data/database/<db_id>/<db_id>.sqlite
     plus the split JSON files. Arrange them so the layout is:

     {DATA}/
       train_spider.json
       dev.json
       database/
         concert_singer/concert_singer.sqlite
         ...

  3. Re-run:  python scripts/get_spider.py --verify-only
"""


def _is_junk(path: Path | str) -> bool:
    parts = Path(path).parts
    return any(p == "__MACOSX" or p.startswith("._") or p == ".DS_Store" for p in parts)


def _real_sqlite_files(root: Path) -> list[Path]:
    """Return only <db_id>/<db_id>.sqlite — ignore macOS AppleDouble sidecars."""
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.glob("*/*.sqlite")
        if p.name == f"{p.parent.name}.sqlite" and not _is_junk(p)
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_json_splits() -> bool:
    if (DATA / "train_spider.json").exists() and (DATA / "dev.json").exists():
        print("JSON splits already present, skipping hub download")
        return True

    try:
        from datasets import load_dataset
    except ImportError:
        print("`datasets` not installed. pip install -r requirements.txt")
        return False

    DATA.mkdir(parents=True, exist_ok=True)
    try:
        for split, fname in [("train", "train_spider.json"), ("validation", "dev.json")]:
            ds = load_dataset("xlangai/spider", split=split)
            rows = [
                {"db_id": r["db_id"], "question": r["question"], "query": r["query"]} for r in ds
            ]
            (DATA / fname).write_text(json.dumps(rows, indent=1))
            print(f"wrote {fname}  ({len(rows)} examples)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"could not fetch splits from the hub: {e}")
        return False


def _download_zip(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _sha256(dest) == HF_DB_SHA256:
        print(f"using cached {dest.name}")
        return dest

    try:
        from huggingface_hub import hf_hub_download

        print(f"downloading {HF_DB_FILE} from {HF_DB_REPO} ...")
        cached = hf_hub_download(
            repo_id=HF_DB_REPO,
            filename=HF_DB_FILE,
            repo_type="dataset",
        )
        shutil.copy2(cached, dest)
    except Exception as e:  # noqa: BLE001
        print(f"huggingface_hub download failed ({e}); trying direct URL")
        import urllib.request

        print(f"downloading {HF_DB_URL} ...")
        urllib.request.urlretrieve(HF_DB_URL, dest)

    digest = _sha256(dest)
    if digest != HF_DB_SHA256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {dest.name}: got {digest}, expected {HF_DB_SHA256}"
        )
    print(f"verified SHA256 {digest}")
    return dest


def _copy_from_extracted(src: Path) -> None:
    """Copy database/ and split JSON out of an extracted spider_data tree.

    The Yale zip was packed on macOS, so it also contains `__MACOSX/` and
    `._*` AppleDouble sidecars. Copying those first looks like a success
    (166 `*.sqlite` paths) but none of the real databases are present.
    """
    matches = [
        p
        for p in src.rglob("database")
        if p.is_dir() and not _is_junk(p) and p.name == "database"
    ]
    if not matches:
        raise FileNotFoundError(f"no database/ folder inside {src}")
    db_src = matches[0]

    if DB_ROOT.exists():
        shutil.rmtree(DB_ROOT)
    shutil.copytree(
        db_src,
        DB_ROOT,
        ignore=lambda _dir, names: [n for n in names if n.startswith("._") or n in {".DS_Store", "__MACOSX"}],
    )
    copied = _real_sqlite_files(DB_ROOT)
    print(f"copied {len(copied)} sqlite files -> {DB_ROOT}")
    if len(copied) < 100:
        raise RuntimeError("extract produced too few real sqlite files; refused to keep AppleDouble-only copy")

    for fname in ("train_spider.json", "dev.json", "tables.json"):
        hits = [p for p in src.rglob(fname) if not _is_junk(p)]
        if hits and not (DATA / fname).exists():
            shutil.copy2(hits[0], DATA / fname)
            print(f"copied {fname}")


def fetch_databases(*, force: bool = False) -> bool:
    dbs = _real_sqlite_files(DB_ROOT)
    if len(dbs) >= 100 and not force:
        print(f"database/ already has {len(dbs)} sqlite files, skipping download")
        return True
    if DB_ROOT.exists() and not dbs:
        print("database/ has no real sqlite files (likely macOS AppleDouble leftovers); re-extracting")

    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / HF_DB_FILE
    extract_dir = DATA / "_extract"
    try:
        _download_zip(zip_path)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if _is_junk(info.filename):
                    continue
                zf.extract(info, extract_dir)
        _copy_from_extracted(extract_dir)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"could not fetch sqlite databases: {e}")
        return False
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)


def verify() -> bool:
    ok = True
    for fname in ("train_spider.json", "dev.json"):
        p = DATA / fname
        if p.exists():
            n = len(json.loads(p.read_text()))
            print(f"  OK   {fname}  ({n} examples)")
        else:
            print(f"  MISS {fname}")
            ok = False

    if not DB_ROOT.exists():
        print("  MISS database/  <- execution accuracy cannot run without this")
        return False

    dbs = _real_sqlite_files(DB_ROOT)
    print(f"  OK   database/  ({len(dbs)} sqlite files)")
    if len(dbs) < 100:
        print("  WARN expected ~166 databases for the full Spider release")
        ok = False

    # Confirm the referenced databases actually exist for the dev split.
    dev = DATA / "dev.json"
    if dev.exists() and dbs:
        needed = {r["db_id"] for r in json.loads(dev.read_text())}
        have = {p.parent.name for p in dbs if p.name == f"{p.parent.name}.sqlite"}
        missing = needed - have
        if missing:
            print(f"  WARN {len(missing)} db_ids in dev.json have no sqlite file, e.g. {sorted(missing)[:3]}")
            ok = False
        else:
            print(f"  OK   all {len(needed)} dev db_ids resolve to a sqlite file")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download and re-extract even if data exists")
    args = ap.parse_args()

    if not args.verify_only:
        fetch_json_splits()
        fetch_databases(force=args.force)

    print("\nverifying data/spider ...")
    if verify():
        print("\nready. next: python -m src.evaluate --config configs/qwen3_1p7b_mps.yaml --tag zeroshot --limit 100")
        return 0
    print(MANUAL)
    return 1


if __name__ == "__main__":
    sys.exit(main())
