# Specification

## Mission

The goal of this project is to build a trainer-agnostic framework for
reinforcement learning from verifiable rewards (RLVR) over executable tasks.
A task may be a single-step prompt/completion/verifier workload or a multi-step
stateful environment. The framework should make it possible to train and
evaluate LLM agents on tasks where correctness is grounded in executable
verification, canonical state, and reproducible trajectories.

- Support both dominant RLVR contracts: prompt-only verification and multi-step
  `reset()`/`step()` episodes.
- Keep each verifier or environment session scalar, not batch-native.
- Keep canonical task state and executable verification authoritative.
- Make trajectories first-class across both single-step and multi-step tasks.
- Treat text, images, and tool outputs as observations over canonical state,
  not as the source of truth themselves.
- Keep the framework flexible enough to host procedural reasoning, coding,
  tool-using workflows, and games behind shared runtime concepts.
- Treat bundled games as reference environments and abstraction stress-tests,
  not as the sole product identity.

## Task Instances And Sessions

The framework should distinguish immutable task identity from mutable session
execution.

A task instance is the verifier-owned prompt, seed, state seed, or task payload
that defines what should be solved. A task session is one scalar execution
against that task instance. For single-step RLVR, trainers may create many
sessions that share one task instance so they can sample multiple completions
for the same prompt. For multi-step environments, a session usually corresponds
to one environment episode.

Task sessions are mutable and should not be reused across concurrent rollouts.
Task instance identity should be stable enough for grouping completions,
trajectory analysis, and reward aggregation.

## Interaction Contracts

### Single-Step Verifier Tasks

The high-throughput RLVR path is a single prompt or task instance, one or more
model completions, and executable verification of the resulting output.

```python
task = task_source.sample(seed=seed)
completion = agent.act(task.prompt)
result = verifier.verify(task=task, completion=completion)
```

This contract should still produce a trajectory-like record containing the
prompt, completion, reward, verifier outputs, and public-safe debug metadata.

### Multi-Step Environments

The stateful path is a session with canonical state, repeated actions,
observations, and terminal conditions.

```python
observation, reset_info = env.reset(seed=seed)

while not env.episode_finished:
    raw_action = agent.act(observation)
    step_result = env.step(raw_action)
    observation = step_result.observation
```

This contract is the right fit for games, tool use, browsing, software
engineering, and other tasks where intermediate interaction matters.

## Required Task Components

Each task implementation should define the pieces needed for executable,
reproducible verification:

- canonical state or verifier-owned task inputs
- a model-facing prompt or observation surface
- a submission format that can be parsed and validated
- an executable verifier or transition function
- reward assignment grounded in executable checks
- terminal or truncation conditions
- trajectory and debug metadata

## Runtime Boundaries

The framework should keep responsibilities clean:

- The environment or verifier layer owns per-session state, submission parsing
  and validation, action legality checks where applicable, transitions,
  tool/resource execution, and verification.
- The workflow layer owns trainer-facing turn packaging, message adaptation,
  and session helpers built on top of the verifier or environment.
- The rollout controller owns concurrency, queueing, retries, cancellation,
  backpressure, and async overlap between generation and verification.
- The trainer owns model inference, rollout fan-out, minibatch construction,
  freshness or staleness policy, and policy updates.

The framework should not push trainer-side batching semantics down into each
environment implementation. One verifier or environment instance should
describe one logical task session; higher-level rollout and trainer code should
decide how many sessions run concurrently.

## Domain Priorities

The near-term priority order should track where RLVR demand is strongest:

- procedural reasoning and verifier-rich prompt tasks
- coding and software-engineering tasks with executable checks
- multi-step tool-using tasks such as browser or workplace workflows
- games as bundled reference environments for stateful interaction

## Games

Bundled reference environments today:

- Chess
- 2048
- Connect 4
- Mastermind
- Minesweeper
- Yahtzee

New games are useful when they expose missing reusable abstractions such as
partial observability, stochastic reset events, auto-advance policies, richer
trajectory metadata, or distinctive verifier-backed reward structures.
Adding games is not the primary roadmap by itself.

## Non-Goals

- Treat rendered text or images as the authoritative state.
- Tie the framework to one training library, one serving system, or one rollout
  engine.
- Require every task to be multi-step, multimodal, or game-shaped.
