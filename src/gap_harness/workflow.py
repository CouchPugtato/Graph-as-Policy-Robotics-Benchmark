"""Validate hierarchical contracts and compile existing action bindings."""

from __future__ import annotations

from collections.abc import Mapping

from gap_harness.models import CheckpointSet, Subgraph, Task, Topology
from gap_isaac.graph_policy import GraphPolicy


def unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def routes_map(routes):
    unique([route.outcome for route in routes], "route outcomes")
    return {route.outcome: route.target for route in routes}


def check_reachability(start, adjacency, terminals):
    if start not in adjacency:
        raise ValueError("entry must name a node")
    seen, pending = set(), [start]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(adjacency.get(node, []))
    if set(adjacency) - seen:
        raise ValueError("unreachable nodes are not allowed")
    # every reachable node must have some bounded path to an exit
    exiting = set(terminals)
    while True:
        expanded = exiting | {node for node, targets in adjacency.items()
                              if any(target in exiting for target in targets)}
        if expanded == exiting:
            break
        exiting = expanded
    if set(adjacency) - exiting:
        raise ValueError("graph contains a region with no terminal path")
    if not seen & set(terminals):
        raise ValueError("graph has no reachable exit")


def validate_topology(topology: Topology, task: Task, max_subgraphs=8):
    if len(topology.subgraphs) > max_subgraphs:
        raise ValueError("orchestrator exceeded the skill agent budget")
    specs = {spec.name: spec for spec in topology.subgraphs}
    terminals = {terminal.name: terminal.status for terminal in topology.terminals}
    unique([spec.name for spec in topology.subgraphs], "subgraph names")
    unique([terminal.name for terminal in topology.terminals], "terminal names")
    if set(specs) & set(terminals):
        raise ValueError("subgraphs and terminals must have distinct names")
    if set(terminals.values()) != {"succeeded", "failed"}:
        raise ValueError("workflow must declare success and failure terminals")
    skills = {skill.name for skill in task.skills}
    adjacency = {}
    for spec in topology.subgraphs:
        if spec.skill not in skills:
            raise ValueError(f"undeployed skill {spec.skill}")
        if next(skill for skill in task.skills if skill.name == spec.skill).shared_tools:
            raise ValueError("shared perception tools belong inside physical action subgraphs")
        unique(spec.exits, "subgraph exits")
        if spec.on_error in spec.exits or not spec.on_error:
            raise ValueError("failure exit must be distinct from success exits")
        routes = routes_map(spec.routes)
        if set(routes) != set(spec.exits) | {spec.on_error}:
            raise ValueError("every subgraph exit requires exactly one outer route")
        if set(routes.values()) - (set(specs) | set(terminals)):
            raise ValueError("outer route targets an unknown subgraph or terminal")
        if routes[spec.on_error] in terminals and terminals[routes[spec.on_error]] != "failed":
            raise ValueError("failure cannot route directly to success")
        for ports in (spec.inputs, spec.outputs):
            unique([port.name for port in ports], "ports")
        adjacency[spec.name] = list(routes.values())
    check_reachability(topology.start, adjacency, terminals)
    # intersect predecessor outputs so a port exists on every incoming path
    types = {port.name: port.type for port in task.initial_inputs}
    for spec in topology.subgraphs:
        for port in [*spec.inputs, *spec.outputs]:
            if port.name in types and types[port.name] != port.type:
                raise ValueError(f"conflicting type for port {port.name}")
            types[port.name] = port.type
    initial = {port.name for port in task.initial_inputs}
    available = {name: set(types) for name in specs}
    available[topology.start] = set(initial)
    while True:
        updated = {}
        for name in specs:
            paths = [set(initial)] if name == topology.start else []
            for parent in topology.subgraphs:
                for route in parent.routes:
                    if route.target == name:
                        outputs = ({port.name for port in parent.outputs}
                                   if route.outcome in parent.exits else set())
                        paths.append(available[parent.name] | outputs)
            updated[name] = set.intersection(*paths) if paths else set()
        if updated == available:
            break
        available = updated
    for spec in topology.subgraphs:
        if {port.name for port in spec.inputs} - available[spec.name]:
            raise ValueError(f"subgraph {spec.name} consumes unavailable inputs")


def compile_workflow(topology: Topology, subgraphs: list[Subgraph], task: Task,
                     checkpoints: CheckpointSet | None = None, max_subgraphs=8):
    validate_topology(topology, task, max_subgraphs)
    unique([subgraph.name for subgraph in subgraphs], "generated subgraphs")
    generated = {subgraph.name: subgraph for subgraph in subgraphs}
    specs = {spec.name: spec for spec in topology.subgraphs}
    if set(generated) != set(specs):
        raise ValueError("skill agents must fill exactly the declared subgraphs")
    skills = {skill.name: skill for skill in task.skills}
    nodes, membership, kinds = [], {}, {}

    def entry(target):
        if target in generated:
            return f"{target}__{generated[target].start}"
        return target

    for name, spec in specs.items():
        graph = generated[name]
        unique([node.name for node in graph.nodes], "node names")
        local = {node.name for node in graph.nodes}
        exits = set(spec.exits) | {spec.on_error}
        if local & exits:
            raise ValueError("local nodes cannot shadow exits")
        actions = {action.name: action for skill in task.skills
                   if skill.shared_tools for action in skill.actions}
        actions.update({action.name: action for action in skills[spec.skill].actions})
        assigned = {action.name for action in skills[spec.skill].actions
                    if action.kind != "router"}
        shared = {action.name for skill in task.skills if skill.shared_tools
                  for action in skill.actions}
        used = {node.action for node in graph.nodes}
        if used & shared and not used & assigned:
            raise ValueError("perception tools must accompany an assigned physical action "
                             "inside the same subgraph")
        outer = routes_map(spec.routes)
        adjacency = {}
        for node in graph.nodes:
            if node.action not in actions or actions[node.action].kind != node.kind:
                raise ValueError(f"undeployed action or mismatched kind {node.action}")
            routes = routes_map(node.routes)
            if set(routes) != set(actions[node.action].outcomes):
                raise ValueError("node must route every declared action outcome")
            positive = "true" if node.kind == "router" else "succeeded"
            if positive not in routes or "failed" not in routes:
                raise ValueError("actions must expose success and failure outcomes")
            if node.kind == "router" and "false" not in routes:
                raise ValueError("routers must expose true false and failed outcomes")
            if set(routes.values()) - (local | exits):
                raise ValueError("inner route leaves its subgraph without an exit")
            if routes["failed"] in spec.exits:
                raise ValueError("failed actions cannot declare a successful subgraph exit")
            adjacency[node.name] = list(routes.values())
            node_id = f"{name}__{node.name}"
            membership[node_id] = name
            kinds[node_id] = node.kind
            nodes.append({"id": node_id, "skill": spec.skill, "action": node.action,
                          "routes": {outcome: (entry(outer[target]) if target in exits
                                               else f"{name}__{target}")
                                     for outcome, target in routes.items()}})
        check_reachability(graph.start, adjacency, exits)
    if len(nodes) > 64:
        raise ValueError("workflow exceeds the existing executor node budget")
    deployed = {node["action"] for node in nodes}
    if not set(task.required_actions + task.required_executed_actions) <= deployed:
        raise ValueError("workflow omitted required task actions")
    if checkpoints is not None:
        counts = {name: 0 for name in specs}
        unique([(check.subgraph, check.name) for check in checkpoints.checkpoints], "checkpoints")
        for check in checkpoints.checkpoints:
            if check.subgraph not in specs or check.evidence not in task.checkpoint_evidence:
                raise ValueError("checkpoint references an unknown subgraph or evidence path")
            counts[check.subgraph] += 1
        if any(not 1 <= count <= 6 for count in counts.values()):
            raise ValueError("each subgraph requires one to six checkpoints")
    result = {"schema_version": 1, "graph_id": topology.graph_id,
              "start": entry(topology.start), "nodes": nodes,
              "terminals": {terminal.name: terminal.status for terminal in topology.terminals},
              "metadata": {"task_id": task.task_id, "subgraph_membership": membership,
                           "node_kinds": kinds,
                           "hierarchical": True}}
    unique([node["id"] for node in nodes], "compiled node names")
    if {node["id"] for node in nodes} & set(result["terminals"]):
        raise ValueError("compiled nodes cannot shadow terminals")
    definitions = {action.name: action for skill in task.skills for action in skill.actions}
    completed = {node["id"]: set(definitions) for node in nodes}
    while True:
        updated = {}
        for node in nodes:
            paths = [set()] if node["id"] == result["start"] else []
            for parent in nodes:
                for outcome, target in parent["routes"].items():
                    if target == node["id"]:
                        paths.append(completed[parent["id"]] | (
                            {parent["action"]} if outcome == (
                                "true" if kinds[parent["id"]] == "router" else "succeeded")
                            else set()))
            updated[node["id"]] = set.intersection(*paths) if paths else set()
        if updated == completed:
            break
        completed = updated
    for node in nodes:
        if set(definitions[node["action"]].requires) - completed[node["id"]]:
            raise ValueError(f"action prerequisites not guaranteed before {node['action']}")
        for outcome, target in node["routes"].items():
            if result["terminals"].get(target) == "succeeded":
                reached = completed[node["id"]] | (
                    {node["action"]} if outcome == (
                        "true" if kinds[node["id"]] == "router" else "succeeded") else set())
                if set(task.required_actions) - reached:
                    raise ValueError("success path can skip required task actions")
    GraphPolicy.from_dict(result)
    return result


def lookup(document, path):
    value = document
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value
