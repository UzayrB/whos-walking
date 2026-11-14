"""Shared fixtures. No bytecode anywhere (the thesis code in ../Code is only ever read); OpenDay importable from any
cwd."""

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import openday  # noqa: E402


@pytest.fixture
def cfg_shipped():
    """The configuration as shipped."""
    return openday.load_config()


@pytest.fixture
def cfg(request, cfg_shipped):
    """The configuration a test gets as `cfg`. By default the shipped one with the nearest-centre rule (m0) and three
    learning walks each (Uzayr too): no trained model needed. A test marked `shipped` gets the configuration as
    shipped (the round of the day: Uzayr walks pools.n_enrol_uzayr, the visitor pools.n_enrol); one marked
    `alternate_round` gets the older alternating round: three learning walks each, Uzayr first, his walks wiped with
    each player (no Redo my walks). The second look (classifier.second_look) is as shipped in all three."""
    if request.node.get_closest_marker("shipped"):
        return cfg_shipped
    if request.node.get_closest_marker("alternate_round"):
        cfg_shipped["pools"].update(enrol_order="alternate", n_enrol=3, n_enrol_uzayr=3, uzayr_reuse_min=0)
        return cfg_shipped
    c = openday.load_config()
    c["classifier"]["kind"] = "m0"
    c["pools"].update(n_enrol=3, n_enrol_uzayr=3)
    return c


@pytest.fixture(autouse=True)
def _close_figures():
    """Every figure a test opened (each app draws its screen headless) is closed after it."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(scope="session")
def _test_logs(tmp_path_factory):
    return str(tmp_path_factory.mktemp("logs"))


@pytest.fixture(autouse=True)
def _logs_in_tmp(_test_logs, monkeypatch):
    """Run logs and replay summaries from the tests go to the temporary folder, not OpenDay/logs/."""
    monkeypatch.setattr(openday, "LOGS", _test_logs)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Temporary files stay inside OpenDay/ whatever directory pytest is run from (nothing is written outside it).
    Two markers for tests that need what only the owner's laptop has, and two that choose the configuration a
    test's `cfg` is."""
    config.option.basetemp = os.path.join(openday.OPENDAY, ".pytest_tmp")
    config.addinivalue_line("markers", "needs_code: needs the thesis code and recordings in ../Code and the "
                                       "participant code list configs/people.local.yaml")
    config.addinivalue_line("markers", "needs_model: needs the trained model in state/model")
    config.addinivalue_line("markers", "shipped: cfg is the configuration as shipped")
    config.addinivalue_line("markers", "alternate_round: cfg is the shipped configuration with the older "
                                       "alternating round (pools.enrol_order alternate)")


def pytest_runtest_setup(item):
    """Skip a marked test when this machine lacks what it needs (a public clone has neither)."""
    if item.get_closest_marker("needs_code") and not (os.path.isdir(os.path.join(openday.CODE, "pipeline"))
                                                      and openday.have_people()):
        pytest.skip("needs the thesis code in ../Code and configs/people.local.yaml (laptop only)")
    if item.get_closest_marker("needs_model") and not os.path.exists(
            os.path.join(openday.OPENDAY, openday.load_config()["classifier"]["model_path"])):
        pytest.skip("needs the trained model in state/model (laptop only)")


def pytest_sessionstart(session):
    """A stamp for this test session (kept outside the temporary folder, which pytest empties), so the last test
    can check nothing under Code/ changed during it."""
    stamp = os.path.join(openday.STATE, ".code_stamp_session")
    os.makedirs(openday.STATE, exist_ok=True)
    with open(stamp, "w") as f:
        f.write("")
    os.environ["OPENDAY_SESSION_STAMP"] = stamp
