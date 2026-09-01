"""Normalize `created_at` timestamps in the SQLite DB to timezone-aware UTC.

This script:
- Backs up the DB file to `data/blood_analyzer.db.bak` (if present)
- Scans `analysis_records.created_at`
- For naive datetimes (no timezone info), assumes UTC and rewrites them as ISO UTC strings

Run:
    python scripts/normalize_created_at.py
"""
from datetime import datetime, timezone
from pathlib import Path
import shutil

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from database.connection import get_engine
from sqlalchemy import text

DB_PATH = Path("data") / "blood_analyzer.db"
if not DB_PATH.exists():
    print("No DB file found at", DB_PATH)
    raise SystemExit(0)

bak = DB_PATH.with_suffix(".db.bak")
shutil.copy2(DB_PATH, bak)
print("Backup saved to", bak)

engine = get_engine()
with engine.begin() as conn:
    rows = conn.execute(text("SELECT id, created_at FROM analysis_records WHERE created_at IS NOT NULL")).fetchall()
    updated = 0
    for r in rows:
        rid, created = r
        if isinstance(created, str):
            # Already ISO formatted string — skip
            continue
        if created is None:
            continue
        # created may be a naive datetime object
        try:
            if created.tzinfo is None or created.tzinfo.utcoffset(created) is None:
                new_ts = created.replace(tzinfo=timezone.utc)
                conn.execute(text("UPDATE analysis_records SET created_at = :ts WHERE id = :id"), {"ts": new_ts.isoformat(), "id": rid})
                updated += 1
        except Exception as e:
            print("Skipping id", rid, "error:", e)

print(f"Normalization complete — updated {updated} rows")
