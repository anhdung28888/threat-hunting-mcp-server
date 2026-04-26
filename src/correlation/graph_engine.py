"""Graph-based threat detection and correlation engine"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Entity types treated as high-value targets when looking for critical
# attack paths. Callers can override per-graph.
DEFAULT_HIGH_VALUE_TYPES = (
    "domain_controller",
    "database",
    "admin_account",
    "host",  # any host is a fallback target if nothing more specific known
)

# Heuristic markers used by `_event_to_entity` to upgrade a generic
# host/user node to a high-value node when the event tells us so.
DOMAIN_CONTROLLER_HINTS = ("dc", "domain_controller", "domaincontroller")
ADMIN_ACCOUNT_HINTS = ("admin", "administrator", "domain admins", "enterprise admins")


@dataclass
class EntityNode:
    """Represents an entity in the attack graph"""

    entity_id: str
    entity_type: str  # process, user, host, file, network_connection
    properties: Dict
    first_seen: datetime
    last_seen: datetime
    suspicious_score: float = 0.0


@dataclass
class RelationshipEdge:
    """Represents a relationship between entities"""

    source_id: str
    target_id: str
    relationship_type: str  # created, accessed, connected_to, executed
    timestamp: datetime
    properties: Dict


@dataclass
class AttackPath:
    """Represents a potential attack path through the graph"""

    path_id: str
    nodes: List[EntityNode]
    edges: List[RelationshipEdge]
    confidence: float
    kill_chain_stages: List[str]
    ttps: List[str]


class AttackGraph:
    """Graph structure for tracking attack progression"""

    def __init__(self):
        self.nodes: Dict[str, EntityNode] = {}
        self.edges: List[RelationshipEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)

    def add_entity_node(self, entity: EntityNode):
        """Adds an entity node to the graph"""
        self.nodes[entity.entity_id] = entity

    def add_relationship_edge(self, edge: RelationshipEdge):
        """Adds a relationship edge to the graph"""
        self.edges.append(edge)
        self.adjacency[edge.source_id].append(edge.target_id)

    def get_neighbors(self, entity_id: str) -> List[EntityNode]:
        """Gets neighboring entities"""
        neighbor_ids = self.adjacency.get(entity_id, [])
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def find_paths(self, start_id: str, end_id: str, max_depth: int = 10) -> List[List[str]]:
        """Finds all paths between two entities"""
        paths = []
        visited = set()

        def dfs(current: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return

            if current == target:
                paths.append(path.copy())
                return

            visited.add(current)
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, target, path, depth + 1)
                    path.pop()
            visited.remove(current)

        dfs(start_id, end_id, [start_id], 0)
        return paths


class ProvenanceGraph:
    """Tracks data provenance and lineage"""

    def __init__(self):
        self.lineage: Dict[str, List[str]] = defaultdict(list)

    def add_provenance(self, entity_id: str, source_id: str):
        """Adds provenance tracking"""
        self.lineage[entity_id].append(source_id)

    def get_ancestry(self, entity_id: str) -> List[str]:
        """Gets complete ancestry of an entity"""
        ancestry = []
        to_visit = [entity_id]
        visited = set()

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue

            visited.add(current)
            ancestry.append(current)
            to_visit.extend(self.lineage.get(current, []))

        return ancestry


class GraphCorrelationEngine:
    """Graph-based threat detection and correlation"""

    def __init__(self):
        self.provenance_tracker = ProvenanceGraph()
        self.lolbin_signatures = self._load_lolbin_signatures()
        self.suspicious_parent_child = self._load_suspicious_relationships()

    async def build_attack_graph(self, events: List[Dict]) -> AttackGraph:
        """Builds directed graph of attack progression"""
        graph = AttackGraph()

        # Create nodes for entities
        for event in events:
            entity = self._event_to_entity(event)
            if entity:
                graph.add_entity_node(entity)

        # Create relationship edges
        for event in events:
            edges = self._event_to_relationships(event)
            for edge in edges:
                graph.add_relationship_edge(edge)

        return graph

    async def detect_living_off_the_land(self, process_tree: Dict) -> List[Dict]:
        """Detects LOLBin abuse through graph analysis"""
        suspicious_patterns = []

        # Check for unusual parent-child relationships
        if self._is_suspicious_parent(process_tree):
            suspicious_patterns.append(
                {
                    "type": "unexpected_child_process",
                    "confidence": self._calculate_lol_confidence(process_tree),
                    "parent": process_tree.get("parent_process"),
                    "child": process_tree.get("process_name"),
                    "explanation": "Unusual parent-child process relationship detected",
                }
            )

        # Detect rapid sequential LOLBin execution
        if self._detect_lolbin_chain(process_tree):
            suspicious_patterns.append(
                {
                    "type": "lolbin_chain",
                    "confidence": 0.8,
                    "tools": self._extract_lolbin_sequence(process_tree),
                    "explanation": "Multiple living-off-the-land binaries executed in sequence",
                }
            )

        # Detect LOLBin with suspicious command line
        if self._has_suspicious_cmdline(process_tree):
            suspicious_patterns.append(
                {
                    "type": "suspicious_lolbin_cmdline",
                    "confidence": 0.7,
                    "process": process_tree.get("process_name"),
                    "cmdline": process_tree.get("command_line"),
                    "explanation": "LOLBin executed with suspicious command line arguments",
                }
            )

        return suspicious_patterns

    async def find_critical_paths(
        self,
        graph: AttackGraph,
        pivot_points: List[str],
        high_value_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.6,
    ) -> List[AttackPath]:
        """Identifies critical attack paths through the graph.

        `high_value_types` defaults to DEFAULT_HIGH_VALUE_TYPES which
        includes the entity types this engine actually emits. Callers
        can override per-environment.
        """
        critical_paths: List[AttackPath] = []
        target_types = set(high_value_types or DEFAULT_HIGH_VALUE_TYPES)

        for pivot in pivot_points:
            if pivot not in graph.nodes:
                continue

            for node_id, node in graph.nodes.items():
                if node_id == pivot:
                    continue
                if node.entity_type not in target_types:
                    continue

                paths = graph.find_paths(pivot, node_id)
                for path_ids in paths:
                    attack_path = self._analyze_attack_path(graph, path_ids)
                    if attack_path.confidence >= confidence_threshold:
                        critical_paths.append(attack_path)

        return critical_paths

    async def identify_pivot_nodes(self, graph: AttackGraph) -> List[Dict]:
        """
        Identifies key pivot points using betweenness centrality
        Nodes with high betweenness are critical to attack progression
        """
        pivot_nodes = []

        # Calculate betweenness centrality
        centrality_scores = self._calculate_betweenness_centrality(graph)

        # Sort by centrality
        sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)

        # Top 10% are potential pivots
        top_count = max(1, len(sorted_nodes) // 10)
        for node_id, score in sorted_nodes[:top_count]:
            node = graph.nodes.get(node_id)
            if node:
                pivot_nodes.append(
                    {
                        "entity_id": node_id,
                        "entity_type": node.entity_type,
                        "centrality_score": score,
                        "properties": node.properties,
                        "explanation": "High betweenness centrality indicates critical pivot point",
                    }
                )

        return pivot_nodes

    def _event_to_entity(self, event: Dict) -> Optional[EntityNode]:
        """Converts an event to an entity node.

        Promotes generic host/user nodes to high-value types
        (domain_controller, admin_account, database) when the event
        carries explicit role/type hints, so downstream methods like
        `find_critical_paths` can actually find them.
        """
        timestamp = event.get("timestamp", datetime.utcnow())

        if "process_name" in event:
            return EntityNode(
                entity_id=event.get("process_guid", event.get("process_name")),
                entity_type="process",
                properties=event,
                first_seen=timestamp,
                last_seen=timestamp,
            )

        if "user_name" in event:
            return EntityNode(
                entity_id=event.get("user_name"),
                entity_type=self._classify_user(event),
                properties=event,
                first_seen=timestamp,
                last_seen=timestamp,
            )

        if "host_name" in event:
            return EntityNode(
                entity_id=event.get("host_name"),
                entity_type=self._classify_host(event),
                properties=event,
                first_seen=timestamp,
                last_seen=timestamp,
            )

        # Allow callers to express a typed entity directly.
        if "entity_id" in event and "entity_type" in event:
            return EntityNode(
                entity_id=event["entity_id"],
                entity_type=event["entity_type"],
                properties=event,
                first_seen=timestamp,
                last_seen=timestamp,
            )

        return None

    @staticmethod
    def _classify_user(event: Dict) -> str:
        explicit = (event.get("account_type") or "").lower()
        if explicit in {"admin", "admin_account"}:
            return "admin_account"

        groups = event.get("groups") or []
        if isinstance(groups, str):
            groups = [groups]
        groups_lower = [str(g).lower() for g in groups]
        username = str(event.get("user_name", "")).lower()
        if any(hint in username for hint in ADMIN_ACCOUNT_HINTS):
            return "admin_account"
        if any(hint in g for g in groups_lower for hint in ADMIN_ACCOUNT_HINTS):
            return "admin_account"
        return "user"

    @staticmethod
    def _classify_host(event: Dict) -> str:
        explicit = (event.get("host_role") or event.get("role") or "").lower()
        if explicit in {"domain_controller", "database"}:
            return explicit
        host_name = str(event.get("host_name", "")).lower()
        if any(hint in host_name for hint in DOMAIN_CONTROLLER_HINTS):
            return "domain_controller"
        if "sql" in host_name or "db" in host_name:
            return "database"
        return "host"

    def _event_to_relationships(self, event: Dict) -> List[RelationshipEdge]:
        """Extracts relationships from an event.

        Emits semantic relationship types (lateral_movement,
        credential_access) when the event signature implies them, so
        `_extract_ttps_from_path` can map them to MITRE techniques.
        """
        edges: List[RelationshipEdge] = []
        timestamp = event.get("timestamp", datetime.utcnow())

        # Process creation relationship.
        if "parent_process" in event and "process_name" in event:
            edges.append(
                RelationshipEdge(
                    source_id=event.get("parent_process"),
                    target_id=event.get("process_guid", event.get("process_name")),
                    relationship_type="created",
                    timestamp=timestamp,
                    properties=event,
                )
            )

        # Network connection relationship.
        if "source_ip" in event and "dest_ip" in event:
            relationship_type = "connected_to"
            # Treat connections to typical lateral-movement ports/services
            # as `lateral_movement` so TTP extraction can match T1021.
            dest_port = event.get("dest_port")
            service = str(event.get("service") or event.get("protocol") or "").lower()
            try:
                dest_port_int = int(dest_port) if dest_port is not None else None
            except (TypeError, ValueError):
                dest_port_int = None
            lateral_ports = {22, 135, 139, 445, 3389, 5985, 5986}
            lateral_services = {"smb", "rdp", "winrm", "ssh", "psexec", "wmi"}
            if dest_port_int in lateral_ports or service in lateral_services:
                relationship_type = "lateral_movement"
            edges.append(
                RelationshipEdge(
                    source_id=event.get("source_ip"),
                    target_id=event.get("dest_ip"),
                    relationship_type=relationship_type,
                    timestamp=timestamp,
                    properties=event,
                )
            )

        # Credential access — Sysmon EID 10 / 4656 against lsass, etc.
        target_image = str(event.get("target_image") or event.get("TargetImage") or "").lower()
        object_name = str(event.get("object_name") or event.get("Object_Name") or "").lower()
        if "lsass" in target_image or "lsass" in object_name:
            source = (
                event.get("source_image")
                or event.get("SourceImage")
                or event.get("process_name")
            )
            target = event.get("target_image") or event.get("TargetImage") or "lsass.exe"
            if source and target:
                edges.append(
                    RelationshipEdge(
                        source_id=source,
                        target_id=target,
                        relationship_type="credential_access",
                        timestamp=timestamp,
                        properties=event,
                    )
                )

        return edges

    def _is_suspicious_parent(self, process_tree: Dict) -> bool:
        """Checks if parent-child relationship is suspicious"""
        parent = process_tree.get("parent_process", "").lower()
        child = process_tree.get("process_name", "").lower()

        # Check against known suspicious relationships
        for suspicious_pair in self.suspicious_parent_child:
            if suspicious_pair["parent"] in parent and suspicious_pair["child"] in child:
                return True

        return False

    def _detect_lolbin_chain(self, process_tree: Dict) -> bool:
        """Detects chained LOLBin execution"""
        children = process_tree.get("children", [])

        lolbin_count = 0
        for child in children:
            child_name = child.get("process_name", "").lower()
            if any(lolbin in child_name for lolbin in self.lolbin_signatures):
                lolbin_count += 1

        # 3+ LOLBins in sequence is suspicious
        return lolbin_count >= 3

    def _extract_lolbin_sequence(self, process_tree: Dict) -> List[str]:
        """Extracts sequence of LOLBins"""
        sequence = []
        children = process_tree.get("children", [])

        for child in children:
            child_name = child.get("process_name", "")
            if any(lolbin in child_name.lower() for lolbin in self.lolbin_signatures):
                sequence.append(child_name)

        return sequence

    def _has_suspicious_cmdline(self, process_tree: Dict) -> bool:
        """Checks for suspicious command line arguments"""
        cmdline = process_tree.get("command_line", "").lower()
        process_name = process_tree.get("process_name", "").lower()

        # PowerShell obfuscation indicators
        if "powershell" in process_name:
            suspicious_patterns = [
                "-enc",
                "-w hidden",
                "-nop",
                "downloadstring",
                "invoke-expression",
                "iex",
                "bypass"]
            return any(pattern in cmdline for pattern in suspicious_patterns)

        # WMIC suspicious usage
        if "wmic" in process_name:
            suspicious_patterns = ["process call create", "/node:", "shadowcopy"]
            return any(pattern in cmdline for pattern in suspicious_patterns)

        # Certutil abuse
        if "certutil" in process_name:
            suspicious_patterns = ["-decode", "-urlcache", "-split"]
            return any(pattern in cmdline for pattern in suspicious_patterns)

        return False

    def _calculate_lol_confidence(self, process_tree: Dict) -> float:
        """Calculates confidence score for LOLBin detection"""
        confidence = 0.5

        # Increase if multiple indicators
        if self._has_suspicious_cmdline(process_tree):
            confidence += 0.2

        if self._is_suspicious_parent(process_tree):
            confidence += 0.2

        # Decrease if common legitimate scenarios
        if process_tree.get("signed", False):
            confidence -= 0.1

        return min(1.0, max(0.0, confidence))

    def _analyze_attack_path(self, graph: AttackGraph, path_ids: List[str]) -> AttackPath:
        """Analyzes an attack path and maps to kill chain"""
        nodes = [graph.nodes[nid] for nid in path_ids if nid in graph.nodes]
        edges = []

        # Find edges connecting the path
        for i in range(len(path_ids) - 1):
            source = path_ids[i]
            target = path_ids[i + 1]
            matching_edges = [e for e in graph.edges if e.source_id ==
                              source and e.target_id == target]
            edges.extend(matching_edges)

        # Map to kill chain stages
        kill_chain_stages = self._map_to_kill_chain(nodes, edges)

        # Extract TTPs
        ttps = self._extract_ttps_from_path(nodes, edges)

        # Calculate confidence
        confidence = self._calculate_path_confidence(nodes, edges)

        return AttackPath(
            path_id=f"path_{hash(tuple(path_ids))}",
            nodes=nodes,
            edges=edges,
            confidence=confidence,
            kill_chain_stages=kill_chain_stages,
            ttps=ttps,
        )

    def _calculate_betweenness_centrality(self, graph: AttackGraph) -> Dict[str, float]:
        """Calculates betweenness centrality for all nodes.

        Uses NetworkX's Brandes-algorithm betweenness when available
        (correct, efficient). Falls back to a normalized shortest-path-
        frequency approximation when NetworkX is missing — still based
        on shortest paths via BFS, NOT all-paths DFS like the previous
        implementation.
        """
        if NETWORKX_AVAILABLE:
            digraph = nx.DiGraph()
            digraph.add_nodes_from(graph.nodes.keys())
            for edge in graph.edges:
                digraph.add_edge(edge.source_id, edge.target_id)
            return dict(nx.betweenness_centrality(digraph, normalized=True))

        # Fallback: BFS shortest paths only.
        from collections import deque

        centrality: Dict[str, float] = defaultdict(float)
        node_ids = list(graph.nodes.keys())

        def bfs_shortest_paths(source: str) -> Dict[str, List[List[str]]]:
            paths: Dict[str, List[List[str]]] = {source: [[source]]}
            queue = deque([source])
            distance = {source: 0}
            while queue:
                current = queue.popleft()
                for neighbor in graph.adjacency.get(current, []):
                    if neighbor not in distance:
                        distance[neighbor] = distance[current] + 1
                        paths[neighbor] = [p + [neighbor] for p in paths[current]]
                        queue.append(neighbor)
                    elif distance[neighbor] == distance[current] + 1:
                        paths[neighbor].extend(p + [neighbor] for p in paths[current])
            return paths

        for source in node_ids:
            shortest = bfs_shortest_paths(source)
            for target, target_paths in shortest.items():
                if source == target or not target_paths:
                    continue
                total = len(target_paths)
                for path in target_paths:
                    for node_id in path[1:-1]:
                        centrality[node_id] += 1.0 / total

        # Normalize so results are comparable to NetworkX's normalized output.
        n = len(node_ids)
        if n > 2:
            scale = 1.0 / ((n - 1) * (n - 2))
            return {k: v * scale for k, v in centrality.items()}
        return dict(centrality)

    def _map_to_kill_chain(self, nodes: List[EntityNode],
                           edges: List[RelationshipEdge]) -> List[str]:
        """Maps attack path to cyber kill chain stages.

        Inference is driven by both node types and edge relationship
        types so it works on the entities/relationships this engine
        actually emits.
        """
        stages: List[str] = []

        for node in nodes:
            props_str = str(node.properties).lower()
            if node.entity_type in {"external_connection", "external_ip"}:
                stages.append("delivery")
            elif node.entity_type == "process" and "exploit" in props_str:
                stages.append("exploitation")
            elif node.entity_type in {"persistence_mechanism", "scheduled_task", "registry_run_key"}:
                stages.append("installation")
            elif node.entity_type in {"c2_connection", "beacon"}:
                stages.append("command_and_control")
            elif node.entity_type in {"domain_controller", "database", "admin_account"}:
                stages.append("actions_on_objectives")

        for edge in edges:
            if edge.relationship_type == "credential_access":
                stages.append("exploitation")
            elif edge.relationship_type == "lateral_movement":
                stages.append("lateral_movement")

        return list(dict.fromkeys(stages))  # de-dup, preserve order

    def _extract_ttps_from_path(
            self, nodes: List[EntityNode], edges: List[RelationshipEdge]) -> List[str]:
        """Extracts MITRE ATT&CK TTPs from attack path"""
        ttps = []

        # Analyze patterns to infer TTPs
        for edge in edges:
            if edge.relationship_type == "lateral_movement":
                ttps.append("T1021")  # Remote Services
            elif edge.relationship_type == "credential_access":
                ttps.append("T1003")  # Credential Dumping

        return list(set(ttps))

    def _calculate_path_confidence(
            self, nodes: List[EntityNode], edges: List[RelationshipEdge]) -> float:
        """Calculates confidence in attack path"""
        if not nodes:
            return 0.0

        # Average suspicious scores of nodes
        avg_score = sum(n.suspicious_score for n in nodes) / len(nodes)

        # Bonus for longer paths (more evidence)
        length_bonus = min(0.2, len(nodes) * 0.02)

        return min(1.0, avg_score + length_bonus)

    def _load_lolbin_signatures(self) -> List[str]:
        """Loads Living-off-the-Land binary signatures"""
        return [
            "powershell",
            "cmd",
            "wmic",
            "certutil",
            "bitsadmin",
            "mshta",
            "regsvr32",
            "rundll32",
            "msiexec",
            "wscript",
            "cscript",
            "installutil",
            "regasm",
            "regsvcs",
            "msxsl",
        ]

    def _load_suspicious_relationships(self) -> List[Dict]:
        """Loads suspicious parent-child process relationships"""
        return [
            {"parent": "winword", "child": "powershell"},
            {"parent": "excel", "child": "powershell"},
            {"parent": "outlook", "child": "powershell"},
            {"parent": "winword", "child": "cmd"},
            {"parent": "excel", "child": "wmic"},
            {"parent": "adobe", "child": "powershell"},
            {"parent": "explorer", "child": "wscript"},
        ]
