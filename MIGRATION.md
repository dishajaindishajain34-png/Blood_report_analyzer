DB migration notes — Blood Report Analyzer

This repository introduces small nullable schema additions and a timezone-aware `created_at` column. Follow these steps when upgrading an existing deployment.

What changed
- Added nullable anemia fields to `analysis_records`:
  - `anemia_detected` (Boolean, default False)
  - `anemia_grade` (String)
  - `estimated_hb` (Float)
  - `estimated_hct` (Float)
  - `anemia_type_hint` (String)
- `created_at` column now uses timezone-aware `DateTime(timezone=True)` with a UTC default.

Compatibility
- These new fields are nullable (or have defaults), so existing SQLite databases will continue to work without mandatory migration.
- `created_at` may contain naive timestamps from older rows; application code will return `created_at.isoformat()` when present. If you rely on timezone awareness for queries, consider normalizing existing rows.

Recommended upgrade steps
1. Backup your existing database file (e.g., `data/blood_analyzer.db`) before making changes.

2. If you are OK with a clean slate (discarding past records):
   - Stop the app, delete the DB file, then restart; the app will create a fresh DB with the new schema.

3. To keep existing records but apply schema changes in-place:
   - SQLite allows adding nullable columns without migrations. The new columns are already present if you recreate the DB using the updated models.
   - If you need full migration tooling, use `alembic` to generate a migration script. Example (manual):
     - Install alembic: `pip install alembic`
     - Initialize alembic and create an `env.py` connecting to your `data/blood_analyzer.db`.
     - Generate and run a migration adding the new columns and altering the `created_at` column if needed.

4. Normalizing `created_at` (optional):
   If old rows are naive and you want them explicitly stored as UTC, run a simple script to read all rows, parse/assume UTC for naive timestamps, and write back normalized ISO timestamps into a new column or update the rows.

Notes
- The application writes `anemia_*` fields only when PDF generation completes successfully; these values may be `NULL` for older records.
- If you need help generating an Alembic migration for your environment, I can prepare a sample migration script.

Contact
- If you encounter issues during migration, please provide the DB file or schema dump and I will assist.