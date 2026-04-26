"""Regression tests for src/cognitive/hunter_brain.py.

Specifically pins down the corrected pyramid-of-pain weighting. The
old implementation took a count-weighted *average* of pyramid weights,
which meant adding 10 hashes to a single TTP would *lower* the score
from 1.0 to ~0.18 — the opposite of the intended semantics.
"""

import pytest

from cognitive.hunter_brain import ThreatHunterCognition


@pytest.fixture
def cog() -> ThreatHunterCognition:
    return ThreatHunterCognition()


class TestPyramidOfPainWeighting:
    def test_single_ttp_anchors_score_at_max(self, cog):
        score = cog._weight_by_pyramid_level({"ttps": ["T1003"]})
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_only_hashes_low_score(self, cog):
        # 0.1 is the hash weight; expect score == 0.1 with no other levels.
        score = cog._weight_by_pyramid_level({"hash_values": ["a" * 64]})
        assert score == pytest.approx(0.1, abs=1e-6)

    def test_low_pyramid_evidence_does_not_dilute_high(self, cog):
        # Pre-fix: TTP + 10 hashes scored ~0.18 (worse than just the TTP).
        # Post-fix: must remain >= the TTP-only score.
        ttp_only = cog._weight_by_pyramid_level({"ttps": ["T1003"]})
        ttp_plus_hashes = cog._weight_by_pyramid_level(
            {"ttps": ["T1003"], "hash_values": ["x" * 64] * 10}
        )
        assert ttp_plus_hashes >= ttp_only

    def test_score_clamped_to_one(self, cog):
        score = cog._weight_by_pyramid_level(
            {
                "ttps": ["T1003", "T1021"],
                "tools": ["mimikatz", "psexec"],
                "host_artifacts": ["a", "b", "c"],
                "network_artifacts": ["x", "y"],
                "domain_names": ["bad.example"],
                "ip_addresses": ["1.2.3.4"],
                "hash_values": ["h" * 64],
            }
        )
        assert score <= 1.0

    def test_empty_evidence_returns_zero(self, cog):
        assert cog._weight_by_pyramid_level({}) == 0.0


class TestPublicAPI:
    def test_generate_competing_hypotheses_is_sync(self, cog):
        # The reviewer flagged this as `async` despite no `await`.
        # It is now synchronous.
        result = cog.generate_competing_hypotheses(
            "lateral movement detected", context={}
        )
        assert isinstance(result, list)
        assert len(result) >= 4
        for h in result:
            assert h.text  # nonempty

    def test_assess_hunt_confidence_is_sync(self, cog):
        result = cog.assess_hunt_confidence(
            evidence={
                "ttps": ["T1003"],
                "positive_indicators": ["a"],
                "negative_indicators": ["b"],
            },
            hypothesis="credential dumping",
        )
        assert "final_confidence" in result
        assert 0.0 <= result["final_confidence"] <= 1.0
