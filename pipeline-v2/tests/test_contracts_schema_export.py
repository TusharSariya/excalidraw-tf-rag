"""Regenerate-and-diff check: schemas/*.schema.json must match what the
pydantic models in contracts.py would export right now.

This is the guard against contracts.py drifting away from the checked-in
JSON Schema files (e.g. someone adds a field to StageMessage and forgets to
regenerate). Run ``python -c "from pipeline_v2.contracts import
write_json_schemas; write_json_schemas()"`` to refresh the files after an
intentional contract change.
"""

from __future__ import annotations

import json

from pipeline_v2.contracts import TOPIC_MODELS, export_all_json_schemas, schema_filename


def test_all_topics_have_a_checked_in_schema_file():
    for topic in TOPIC_MODELS:
        path = schema_filename(topic)
        assert path.exists(), (
            f"missing schemas/{path.name} for topic {topic!r} — run write_json_schemas()"
        )


def test_checked_in_schemas_match_current_models():
    current = export_all_json_schemas()
    for topic, schema in current.items():
        path = schema_filename(topic)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk == schema, (
            f"schemas/{path.name} is stale for topic {topic!r} — "
            "contracts.py changed without regenerating the JSON Schema export "
            "(run write_json_schemas())"
        )


def test_no_stray_schema_files():
    schemas_dir = schema_filename(next(iter(TOPIC_MODELS))).parent
    on_disk_names = {p.name for p in schemas_dir.glob("*.schema.json")}
    expected_names = {schema_filename(topic).name for topic in TOPIC_MODELS}
    assert on_disk_names == expected_names
