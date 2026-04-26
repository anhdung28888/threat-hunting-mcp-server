"""Regression tests for src/correlation/graph_engine.py.

Covers the bugs identified in the April 2026 code review:
- `_event_to_entity` only emitted process/user/host, so
  `find_critical_paths` (which targeted domain_controller / database /
  admin_account) silently returned nothing.
- `_event_to_relationships` only emitted `created` / `connected_to`,
  so `_extract_ttps_from_path` (which looked for `lateral_movement`
  / `credential_access`) silently produced no TTPs.
- `_calculate_betweenness_centrality` was a path-frequency hack
  using DFS over ALL paths, not Brandes-on-shortest-paths.
"""

from datetime import datetime

import pytest

from correlation.graph_engine import (
    DEFAULT_HIGH_VALUE_TYPES,
    AttackGraph,
    GraphCorrelationEngine,
    NETWORKX_AVAILABLE,
)


@pytest.fixture
def engine() -> GraphCorrelationEngine:
    return GraphCorrelationEngine()


class TestEntityClassification:
    def test_user_with_admin_hint_promoted_to_admin_account(self, engine):
        node = engine._event_to_entity(
            {"user_name": "domain.admin", "timestamp": datetime.utcnow()}
        )
        assert node is not None
        assert node.entity_type == "admin_account"

    def test_explicit_account_type_admin(self, engine):
        node = engine._event_to_entity(
            {"user_name": "joe", "account_type": "admin"}
        )
        assert node.entity_type == "admin_account"

    def test_regular_user_not_promoted(self, engine):
        node = engine._event_to_entity({"user_name": "alice"})
        assert node.entity_type == "user"

    def test_host_with_dc_hint_promoted(self, engine):
        node = engine._event_to_entity({"host_name": "corp-dc-01"})
        assert node.entity_type == "domain_controller"

    def test_host_with_role_hint(self, engine):
        node = engine._event_to_entity(
            {"host_name": "anything", "host_role": "database"}
        )
        assert node.entity_type == "database"

    def test_explicit_typed_entity_passthrough(self, engine):
        node = engine._event_to_entity(
            {"entity_id": "kv-prod", "entity_type": "secrets_vault"}
        )
        assert node.entity_id == "kv-prod"
        assert node.entity_type == "secrets_vault"


class TestRelationshipExtraction:
    def test_smb_connection_yields_lateral_movement_edge(self, engine):
        edges = engine._event_to_relationships(
            {
                "source_ip": "10.0.0.5",
                "dest_ip": "10.0.0.10",
                "dest_port": 445,
            }
        )
        assert any(e.relationship_type == "lateral_movement" for e in edges)

    def test_rdp_connection_yields_lateral_movement_edge(self, engine):
        edges = engine._event_to_relationships(
            {"source_ip": "10.0.0.5", "dest_ip": "10.0.0.10", "dest_port": 3389}
        )
        assert any(e.relationship_type == "lateral_movement" for e in edges)

    def test_random_connection_remains_connected_to(self, engine):
        edges = engine._event_to_relationships(
            {"source_ip": "10.0.0.5", "dest_ip": "8.8.8.8", "dest_port": 443}
        )
        assert any(e.relationship_type == "connected_to" for e in edges)
        assert not any(e.relationship_type == "lateral_movement" for e in edges)

    def test_lsass_access_yields_credential_access_edge(self, engine):
        edges = engine._event_to_relationships(
            {
                "source_image": "C:\\\\Windows\\\\System32\\\\malicious.exe",
                "target_image": "C:\\\\Windows\\\\System32\\\\lsass.exe",
            }
        )
        assert any(e.relationship_type == "credential_access" for e in edges)


@pytest.mark.asyncio
class TestCriticalPathFinding:
    async def test_find_critical_paths_to_domain_controller(self, engine):
        events = [
            # Workstation pivot (compromised)
            {
                "host_name": "ws-7",
                "timestamp": datetime.utcnow(),
            },
            # Domain controller
            {
                "host_name": "corp-dc-01",
                "timestamp": datetime.utcnow(),
            },
            # SMB lateral movement from pivot to DC
            {
                "source_ip": "ws-7",
                "dest_ip": "corp-dc-01",
                "dest_port": 445,
                "timestamp": datetime.utcnow(),
            },
        ]
        graph = await engine.build_attack_graph(events)
        # Smoke test that nodes exist with expected types
        assert graph.nodes["ws-7"].entity_type == "host"
        assert graph.nodes["corp-dc-01"].entity_type == "domain_controller"

        paths = await engine.find_critical_paths(
            graph, pivot_points=["ws-7"], confidence_threshold=0.0
        )
        # Pre-fix this returned [] because DC nodes were never created.
        assert len(paths) >= 1
        target_types = {p.nodes[-1].entity_type for p in paths}
        assert "domain_controller" in target_types

    async def test_default_high_value_types_includes_admin_account(self):
        # Guards against accidental regression in the constant.
        assert "domain_controller" in DEFAULT_HIGH_VALUE_TYPES
        assert "admin_account" in DEFAULT_HIGH_VALUE_TYPES
        assert "database" in DEFAULT_HIGH_VALUE_TYPES


class TestBetweennessCentrality:
    def test_returns_finite_scores_for_simple_chain(self, engine):
        # Chain: a -> b -> c -> d
        graph = AttackGraph()
        from correlation.graph_engine import EntityNode, RelationshipEdge

        now = datetime.utcnow()
        for nid in ("a", "b", "c", "d"):
            graph.add_entity_node(
                EntityNode(
                    entity_id=nid,
                    entity_type="host",
                    properties={},
                    first_seen=now,
                    last_seen=now,
                )
            )
        for src, dst in [("a", "b"), ("b", "c"), ("c", "d")]:
            graph.add_relationship_edge(
                RelationshipEdge(
                    source_id=src,
                    target_id=dst,
                    relationship_type="connected_to",
                    timestamp=now,
                    properties={},
                )
            )

        scores = engine._calculate_betweenness_centrality(graph)
        # Middle nodes should have higher centrality than endpoints.
        # End nodes either receive 0 or are absent from the dict.
        b = scores.get("b", 0.0)
        c = scores.get("c", 0.0)
        a = scores.get("a", 0.0)
        d = scores.get("d", 0.0)
        assert b > a
        assert c > d
        # All scores finite, non-negative.
        for v in scores.values():
            assert v >= 0.0
            assert v < float("inf")

    @pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="networkx not installed")
    def test_uses_networkx_when_available(self, engine):
        # Just confirm no exception path on a non-trivial graph.
        graph = AttackGraph()
        from correlation.graph_engine import EntityNode, RelationshipEdge

        now = datetime.utcnow()
        for nid in "abcde":
            graph.add_entity_node(
                EntityNode(
                    entity_id=nid,
                    entity_type="host",
                    properties={},
                    first_seen=now,
                    last_seen=now,
                )
            )
        for src, dst in [("a", "b"), ("b", "c"), ("c", "d"), ("a", "c"), ("c", "e")]:
            graph.add_relationship_edge(
                RelationshipEdge(
                    source_id=src,
                    target_id=dst,
                    relationship_type="connected_to",
                    timestamp=now,
                    properties={},
                )
            )
        scores = engine._calculate_betweenness_centrality(graph)
        assert isinstance(scores, dict)
        assert "c" in scores  # high-traffic intermediate
