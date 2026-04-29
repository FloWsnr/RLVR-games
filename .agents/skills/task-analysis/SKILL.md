---
name: task-analysis
description: Use this skill when asked to play-test, benchmark, evaluate difficulty, or analyze the interaction quality of an RLVR task. It guides blind multi-seed task play through the public play CLI, records success ratio and difficulty signals, and keeps solver evaluation separate from code inspection.
---

# Task Analysis Play-Test

Use this workflow to estimate task difficulty and interaction quality from the solver's point of view. Treat the play-test as a blind benchmark unless the user explicitly asks for code review instead.

## Blind Setup

Do not inspect task implementation code, fixtures, tests, private backbones, answer keys, or debug-only fields before or during scored play. Use only the public interaction surface shown by the play CLI. If you accidentally see privileged information, mark the run as contaminated and restart scored evaluation with fresh seeds or a fresh agent.

Discover available task aliases only through:

```bash
uv run play --list
```

Run an interactive task with:

```bash
uv run play <task-alias> --instance-seed <instance-seed> --session-seed <session-seed>
```

Add task-specific CLI parameters only if they are public and part of the task's intended play interface.

## Run Plan

Use 1-2 exploratory runs to understand the public action schema, interaction rhythm and check for errors

For the scored benchmark, we want to use subagents with multiple tries:
- Decide on 3 models (e.g. gpt-5.5 (xhigh)) for the subagents. Choose the most capable model you have access to, the least capable model, and one in between to get a range of performance.
- Run 5 scored runs per model, for a total of 15 scored runs.
- Each run should be a single play session by a subagent with the respective model. You are explicitly allowed to use subagents!
- Instruct each subagent to play **a single run** and report back the results. Start **a new agent** for each run to avoid cross-run contamination.
- Vary both `--instance-seed` and `--session-seed`; use deterministic seed tables so results are reproducible.
- Choose all scored seeds before starting and do not replace seeds based on outcomes.

For visual or renderer-dependent tasks, use the public renderer output exactly as a player would. Do not inspect generated files, serialized internals, or alternate renderers unless the benchmark plan says they are public.

## During Play (for the subagents)

Record each run while playing, not from memory afterward:

- task alias and exact command
- instance seed and session seed
- public observations or measurements used
- actions submitted and any invalid-action feedback
- terminal or truncated status
- public reward, score, correctness, or error metrics
- number of turns/actions used
- qualitative notes on ambiguity, misleading feedback, missing affordances, or suspected leaks

Do not edit task code during scored play. If the public protocol is broken, stop the scored batch, fix or report the protocol issue, then rerun a new scored batch.

## Report

Report the scored results as a compact table and summary:

```text
Task: <task-alias>
Scored runs: <n>
Success ratio: <successes>/<n> = <percent>
Mean reward/score: <value if public>
Truncation rate: <count>/<n>
Invalid action rate: <count>/<n>

Seeds:
| run | instance_seed | session_seed | success | reward/score | turns | notes |
| --- | ------------- | ------------ | ------- | ------------ | ----- | ----- |
```

Include a difficulty estimate based on success ratio, turn count, error size, strategy complexity, and failure modes. Also summarize public protocol quality: whether the instructions were sufficient, whether feedback supported recovery, whether action validation was clear, and whether any public output appeared to leak private state.

## After Blind Evaluation

Only after the blind scored report is complete, inspect implementation code if the user asks for debugging, leak auditing, or task improvement. Keep that code-informed analysis separate from the blind benchmark result.
