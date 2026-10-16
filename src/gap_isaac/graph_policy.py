"""Small, auditable directed-graph executor for simulator-local skill actions."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NodeResult:
    outcome: str = "succeeded"
    detail: str = ""
    outputs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    skill: str
    action: str
    routes: Mapping[str, str]


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    node_id: str
    skill: str
    action: str
    outcome: str
    next_node: str
    detail: str = ""
    outputs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class GraphExecutionResult:
    graph_id: str
    terminal: str
    status: str
    trace: tuple[TraceEvent, ...]
    state: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "terminal": self.terminal,
            "status": self.status,
            "trace": [asdict(event) for event in self.trace],
            "state": dict(self.state),
        }


@dataclass(frozen=True)
class GraphPolicy:
    graph_id: str
    start: str
    nodes: Mapping[str, GraphNode]
    terminals: Mapping[str, str]
    metadata: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GraphPolicy:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported graph schema version")
        graph_id = str(payload.get("graph_id", "")).strip()
        start = str(payload.get("start", "")).strip()
        if not graph_id or not start:
            raise ValueError("graph_id and start are required")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("graph metadata must be an object")

        raw_terminals = payload.get("terminals")
        if not isinstance(raw_terminals, Mapping) or not raw_terminals:
            raise ValueError("graph must declare terminal statuses")
        terminals = {str(name): str(status) for name, status in raw_terminals.items()}

        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError("graph must declare nodes")
        nodes: dict[str, GraphNode] = {}
        for raw_node in raw_nodes:
            if not isinstance(raw_node, Mapping):
                raise ValueError("graph node must be an object")
            node_id = str(raw_node.get("id", "")).strip()
            skill = str(raw_node.get("skill", "")).strip()
            action = str(raw_node.get("action", "")).strip()
            routes = raw_node.get("routes")
            if not node_id or not skill or not action or not isinstance(routes, Mapping):
                raise ValueError("every node requires id, skill, action, and routes")
            if node_id in nodes:
                raise ValueError(f"duplicate graph node: {node_id}")
            nodes[node_id] = GraphNode(
                node_id=node_id,
                skill=skill,
                action=action,
                routes={str(outcome): str(target) for outcome, target in routes.items()},
            )

        if start not in nodes:
            raise ValueError(f"start node does not exist: {start}")
        known_targets = set(nodes) | set(terminals)
        for node in nodes.values():
            kind = metadata.get("node_kinds", {}).get(node.node_id)
            if kind == "router":
                if not {"true", "false", "failed"} <= set(node.routes):
                    raise ValueError(
                        f"router requires true false and failed routes: {node.node_id}")
            elif "succeeded" not in node.routes:
                raise ValueError(f"node has no succeeded route: {node.node_id}")
            unknown = set(node.routes.values()) - known_targets
            if unknown:
                raise ValueError(
                    f"node {node.node_id} routes to unknown targets: {sorted(unknown)}"
                )

        return cls(graph_id, start, nodes, terminals, dict(metadata))

    @classmethod
    def from_json(cls, path: Path) -> GraphPolicy:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


SkillAction = Callable[[MutableMapping[str, Any]], NodeResult]


class GraphPolicyExecutor:
    """Run declared skill actions and route strictly from their outcomes."""

    def __init__(
        self,
        policy: GraphPolicy,
        actions: Mapping[str, SkillAction],
        *,
        max_steps: int = 64,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        missing = {node.action for node in policy.nodes.values()} - set(actions)
        if missing:
            raise ValueError(f"graph actions are unavailable: {sorted(missing)}")
        self._policy = policy
        self._actions = dict(actions)
        self._max_steps = max_steps

    def execute(self, initial_state: Mapping[str, Any] | None = None) -> GraphExecutionResult:
        state: MutableMapping[str, Any] = dict(initial_state or {})
        trace: list[TraceEvent] = []
        current = self._policy.start
        # the graph chooses the next action from the reported outcome
        for sequence in range(self._max_steps):
            node = self._policy.nodes[current]
            try:
                result = self._actions[node.action](state)
            except Exception as error:  # noqa: BLE001 - graph errors must be routed and traced
                result = NodeResult(
                    outcome="failed",
                    detail=f"{type(error).__name__}: {error}",
                )
            # pass action outputs forward without inventing a new route
            if result.outputs:
                state.update(result.outputs)
            next_node = node.routes.get(result.outcome)
            if next_node is None:
                raise RuntimeError(
                    f"node {node.node_id} has no route for outcome {result.outcome!r}"
                )
            # retain failures as well as successes for later diagnosis
            trace.append(
                TraceEvent(
                    sequence=sequence,
                    node_id=node.node_id,
                    skill=node.skill,
                    action=node.action,
                    outcome=result.outcome,
                    next_node=next_node,
                    detail=result.detail,
                    outputs=dict(result.outputs) if result.outputs is not None else None,
                )
            )
            if next_node in self._policy.terminals:
                return GraphExecutionResult(
                    graph_id=self._policy.graph_id,
                    terminal=next_node,
                    status=self._policy.terminals[next_node],
                    trace=tuple(trace),
                    state=dict(state),
                )
            current = next_node
        raise RuntimeError(f"graph exceeded its {self._max_steps}-step execution bound")
