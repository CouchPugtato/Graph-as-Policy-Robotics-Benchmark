"""Task contracts and structured agent outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Port(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    type: str = Field(min_length=1)


class Route(StrictModel):
    outcome: str = Field(min_length=1)
    target: str = Field(min_length=1)


class Terminal(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    status: Literal["succeeded", "failed"]


class SubgraphSpec(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    skill: str
    description: str = Field(min_length=1)
    inputs: list[Port]
    outputs: list[Port]
    exits: list[str] = Field(min_length=1)
    on_error: str
    routes: list[Route]


class Topology(StrictModel):
    graph_id: str = Field(min_length=1)
    rationale: str
    start: str
    terminals: list[Terminal] = Field(min_length=2)
    subgraphs: list[SubgraphSpec] = Field(min_length=1)


class Node(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    kind: Literal["tool", "script", "router"]
    action: str
    routes: list[Route]


class Subgraph(StrictModel):
    name: str
    rationale: str
    start: str
    nodes: list[Node] = Field(min_length=1)


class Checkpoint(StrictModel):
    subgraph: str
    name: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence: str
    expected: Literal["true", "false", "succeeded", "failed"]


class CheckpointSet(StrictModel):
    checkpoints: list[Checkpoint] = Field(min_length=1)


class ActionDefinition(StrictModel):
    name: str
    kind: Literal["tool", "script", "router"] = "tool"
    description: str
    outcomes: list[str] = Field(default_factory=lambda: ["succeeded", "failed"])
    requires: list[str] = Field(default_factory=list)


class SkillDefinition(StrictModel):
    name: str
    shared_tools: bool = False
    description: str
    instructions: str
    actions: list[ActionDefinition] = Field(min_length=1)
    reference_files: list[str] = Field(default_factory=list)


class RunnerDefinition(StrictModel):
    command: list[str] = Field(min_length=1)
    cwd: str = "."
    timeout_s: int = Field(default=120, ge=1, le=600)
    receipt: str = "receipt.json"


class Task(StrictModel):
    schema_version: Literal[1] = 1
    task_id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$")
    instruction: str = Field(min_length=1)
    skills: list[SkillDefinition] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    initial_inputs: list[Port] = Field(default_factory=list)
    checkpoint_evidence: list[str] = Field(min_length=1)
    required_true: list[str] = Field(min_length=1)
    required_actions: list[str] = Field(default_factory=list)
    required_executed_actions: list[str] = Field(default_factory=list)
    max_trace_steps: int = Field(default=64, ge=1, le=160)
    runner: RunnerDefinition | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_catalog(self) -> Task:
        names = [skill.name for skill in self.skills]
        if len(names) != len(set(names)) or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("skill names and seeds must be unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        actions = [action.name for skill in self.skills for action in skill.actions]
        if not set(self.required_actions + self.required_executed_actions) <= set(actions):
            raise ValueError("required actions must be deployed")
        if not set(self.required_true) <= set(self.checkpoint_evidence):
            raise ValueError("required criteria must be declared evidence")
        return self
