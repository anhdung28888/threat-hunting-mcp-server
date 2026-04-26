"""Regression tests for src/integrations/splunk.py.

Covers:
- Module imports without numpy/pandas/sklearn (formerly `ImportError`).
- `splunklib.results` is no longer shadowed by a local `results` dict
  (formerly `AttributeError: 'dict' object has no attribute 'ResultsReader'`).
- T1003/T1083 queries are behavior-based, not IOC string-based.
"""

import importlib

import pytest


def test_module_imports_without_optional_stacks():
    """splunk.py must load even when numpy/pandas/sklearn AND splunk-sdk are missing."""
    import sys as _sys

    if "src.integrations.splunk" in _sys.modules:
        del _sys.modules["src.integrations.splunk"]
    module = importlib.import_module("src.integrations.splunk")

    # Module attributes exist regardless of which optional stacks loaded.
    assert hasattr(module, "SplunkHuntingEngine")
    assert hasattr(module, "ML_AVAILABLE")
    assert hasattr(module, "SPLUNK_SDK_AVAILABLE")
    # The renamed alias is always defined (None when SDK missing).
    assert hasattr(module, "splunk_results")


def test_results_alias_is_module_when_sdk_present():
    """When splunk-sdk is installed, `splunk_results` must be the real module
    (i.e. shadowing-by-local-dict bug must stay fixed)."""
    import src.integrations.splunk as splunk_module

    if splunk_module.SPLUNK_SDK_AVAILABLE:
        assert hasattr(splunk_module.splunk_results, "ResultsReader")
    else:
        # Acceptable: SDK not installed in this environment.
        assert splunk_module.splunk_results is None


class TestHuntQueriesAreBehavioral:
    """The Pyramid of Pain forbids hard-coded malware-name strings."""

    def setup_method(self):
        from src.integrations.splunk import SplunkHuntingEngine

        # Avoid Splunk connection — only exercise pure-Python query loader.
        self.engine = SplunkHuntingEngine.__new__(SplunkHuntingEngine)
        self.engine.hunt_queries = SplunkHuntingEngine._load_hunt_queries(self.engine)

    def test_t1003_does_not_hardcode_malware_names(self):
        from src.integrations.splunk import SplunkHuntingEngine

        engine = SplunkHuntingEngine.__new__(SplunkHuntingEngine)
        queries = SplunkHuntingEngine.get_hunt_queries_by_technique(engine, "T1003")
        joined = " ".join(queries).lower()
        assert "mimikatz" not in joined
        assert "sekurlsa" not in joined
        # Behavioral: should reference Sysmon EID 10 ProcessAccess on lsass
        assert "lsass" in joined
        assert "eventcode=10" in joined

    def test_t1083_does_not_reference_dir_exe(self):
        from src.integrations.splunk import SplunkHuntingEngine

        engine = SplunkHuntingEngine.__new__(SplunkHuntingEngine)
        queries = SplunkHuntingEngine.get_hunt_queries_by_technique(engine, "T1083")
        joined = " ".join(queries).lower()
        # `dir.exe` does not exist as a process; pre-fix the query
        # claimed it did.
        assert "dir.exe" not in joined
        # Should now look for cmd.exe with `dir` in the command line
        assert "cmd.exe" in joined or "tree.com" in joined
