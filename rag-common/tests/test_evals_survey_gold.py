from __future__ import annotations

import sqlite3

import pytest
from rag_common.evals import survey_gold as sg


def _fixture_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE papers(
          oa_id TEXT PRIMARY KEY, doi TEXT, doc_id TEXT, title TEXT, year INTEGER,
          cited_by_count INTEGER DEFAULT 0, in_corpus INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE cites(src_oa TEXT NOT NULL, dst_oa TEXT NOT NULL);
        """
    )
    return db


@pytest.fixture
def db_low_refs():
    """A 'survey'-titled doc with only a couple of in-corpus refs (keyword match only)."""
    db = _fixture_db()
    db.executescript(
        """
        INSERT INTO papers VALUES ('oa1', NULL, 'doc_survey', 'A Survey of Widgets', 2020, 0, 1);
        INSERT INTO papers VALUES ('oa2', NULL, 'doc_ref1', 'Widget Basics', 2010, 0, 1);
        INSERT INTO papers VALUES ('oa3', NULL, 'doc_ref2', 'Advanced Widgets', 2012, 0, 1);
        INSERT INTO cites VALUES ('oa1', 'oa2');
        INSERT INTO cites VALUES ('oa1', 'oa3');
        """
    )
    db.commit()
    return db


def test_survey_detected_by_keyword(db_low_refs):
    meta = {
        "doc_survey": sg.DocMeta("doc_survey", "A Survey of Widgets", ()),
        "doc_ref1": sg.DocMeta("doc_ref1", "Widget Basics", ()),
        "doc_ref2": sg.DocMeta("doc_ref2", "Advanced Widgets", ()),
    }
    surveys = sg.find_surveys(db_low_refs, meta, min_refs=100)
    assert len(surveys) == 1
    assert surveys[0].survey_doc_id == "doc_survey"
    assert surveys[0].matched_by == "keyword"
    assert sorted(surveys[0].sections["_all"]) == ["doc_ref1", "doc_ref2"]


def test_no_keyword_no_high_ref_count_not_a_survey(db_low_refs):
    # min_refs above the fixture's 2 out-refs, and rename via override: simulate a non-
    # survey title by using a fresh DB with a neutral title.
    db = _fixture_db()
    db.executescript(
        """
        INSERT INTO papers VALUES ('oa1', NULL, 'doc_x', 'Widgets in Practice', 2020, 0, 1);
        INSERT INTO papers VALUES ('oa2', NULL, 'doc_ref1', 'Widget Basics', 2010, 0, 1);
        INSERT INTO cites VALUES ('oa1', 'oa2');
        """
    )
    db.commit()
    surveys = sg.find_surveys(db, {}, min_refs=100)
    assert surveys == []


def test_survey_detected_by_ref_count():
    db = _fixture_db()
    rows = [
        "INSERT INTO papers VALUES ('oa_main', NULL, 'doc_main', 'Widgets in Practice', 2020, 0, 1);"
    ]
    for i in range(120):
        rows.append(
            f"INSERT INTO papers VALUES ('oa_r{i}', NULL, 'doc_r{i}', 'Ref {i}', 2010, 0, 1);"
        )
        rows.append(f"INSERT INTO cites VALUES ('oa_main', 'oa_r{i}');")
    db.executescript("\n".join(rows))
    db.commit()

    surveys = sg.find_surveys(db, {}, min_refs=100)
    assert len(surveys) == 1
    assert surveys[0].survey_doc_id == "doc_main"
    assert surveys[0].matched_by == "ref_count"
    assert surveys[0].ref_count == 120


def test_fallback_keywords_opt_in():
    db = _fixture_db()
    db.executescript(
        """
        INSERT INTO papers VALUES ('oa1', NULL, 'doc_h', 'The Widget Handbook', 2020, 0, 1);
        INSERT INTO papers VALUES ('oa2', NULL, 'doc_ref1', 'Widget Basics', 2010, 0, 1);
        INSERT INTO cites VALUES ('oa1', 'oa2');
        """
    )
    db.commit()
    meta = {"doc_h": sg.DocMeta("doc_h", "The Widget Handbook", ())}

    assert sg.find_surveys(db, meta, min_refs=100, use_fallback_keywords=False) == []
    with_fallback = sg.find_surveys(db, meta, min_refs=100, use_fallback_keywords=True)
    assert len(with_fallback) == 1
    assert with_fallback[0].survey_doc_id == "doc_h"


def test_to_gold_dict_shape(db_low_refs):
    meta = {"doc_survey": sg.DocMeta("doc_survey", "A Survey of Widgets", ())}
    surveys = sg.find_surveys(db_low_refs, meta, min_refs=100)
    gold = sg.to_gold_dict(surveys)
    assert set(gold) == {"doc_survey"}
    assert "_all" in gold["doc_survey"]


def test_self_reference_excluded():
    db = _fixture_db()
    db.executescript(
        """
        INSERT INTO papers VALUES ('oa1', NULL, 'doc_a', 'A Survey', 2020, 0, 1);
        INSERT INTO cites VALUES ('oa1', 'oa1');
        """
    )
    db.commit()
    surveys = sg.find_surveys(db, {}, min_refs=1)
    assert surveys == [] or surveys[0].ref_count == 0
