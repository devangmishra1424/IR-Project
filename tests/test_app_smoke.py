"""Smoke test of the Streamlit app against the real artefacts.

Runs the whole app script headlessly with Streamlit's own test harness. It needs the indexes built
(see the README) and is skipped otherwise, so a fresh clone still has a green test suite.
"""

import pytest

from maillens import config

BUILT = (config.ARTIFACTS_DIR / "dense" / "meta.json").exists() and (
    config.PROCESSED_DIR / "emails.parquet"
).exists()
pytestmark = pytest.mark.skipif(not BUILT, reason="build the indexes first (see the README)")

APP = str(config.ROOT / "maillens" / "app" / "streamlit_app.py")
QUESTION = "Who should Katja contact about copyright for the credit article?"


def cards(app) -> list[str]:
    return [m.value for m in app.markdown if 'class="ml-card"' in m.value]


@pytest.fixture(scope="module")
def empty_app():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=600)
    app.run()
    return app


def test_the_app_starts_without_an_exception(empty_app):
    assert not empty_app.exception
    assert any("e-mails" in m.value for m in empty_app.sidebar.markdown)


def test_without_a_query_it_shows_guidance_not_results(empty_app):
    assert cards(empty_app) == []
    assert any("Pick an example" in info.value for info in empty_app.info)


def test_a_query_fills_four_columns_with_cards():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=600)
    app.session_state["active_query"] = QUESTION
    app.run()
    assert not app.exception
    shown = cards(app)
    assert len(shown) == 4 * 5  # four rankers, five results each by default
    assert all("ml-chip" in card for card in shown)  # every card says where the others ranked it
    assert any("<mark>" in card for card in shown)  # query terms are highlighted


def test_untrusted_html_in_a_query_cannot_inject_markup():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP, default_timeout=600)
    app.session_state["active_query"] = '<script>alert("x")</script> gas'
    app.run()
    assert not app.exception
    assert not any("<script>" in m.value for m in app.markdown)
