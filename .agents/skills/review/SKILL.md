---
name: review
description: Use this skill when asked to review a task you did or independently after a bigger change. The reviewer must run on GPT-5.4 with xhigh reasoning, receive a concise neutral summary of the task and implemented solution, and check for bugs, regressions, and missing tests without being primed toward a positive conclusion.
---

# Independent Review Workflow

Run this skill after larger tasks or changes. A meaningful change is any non-trivial behavior change, refactor, or update that materially affects the implementation. Do not run the reviewer for small fixes or immediately after addressing the last review comment.

## Goal

Use a separate review agent to challenge the latest change before handoff. The review should be independent and skeptical, with the primary goal of finding bugs, regressions, weak assumptions, and missing coverage. Reviewing takes time. Make sure to let the agent finish its work, don't abort it early.

## Required Workflow

1. After completing a meaningful change, start an independent review agent with `model: gpt-5.4` and `reasoning_effort: xhigh`.
2. Keep the review context intentionally narrow. Do not fork the full conversation by default. Pass only the task-local information the reviewer needs.
3. Give the reviewer a short neural summary of:
   - the user objective
   - the constraints or invariants that matter
   - the changed files or code areas
   - the behavior that was added, removed, or modified
4. Describe the implementation neutrally. Do not frame it as already correct, clean, fixed, or improved. Avoid language that nudges the reviewer toward approval.
5. Ask explicitly for critical review focused on bugs, regressions, edge cases, invalid assumptions, and missing tests. Prefer concrete findings with file references. Also ask whether the refactor fits in the overall architectural direction, including for future work and scaling.
6. For long tasks, review each completed chunk after it becomes meaningful instead of waiting for one final review at the end.
7. Incorporate the review findings into your work or explain why a finding was not applied.

## Prompt Guidance

The reviewer should receive a compact, neutral summary. Good inputs state what changed and what constraints matter. Bad inputs sell the solution.

Use a prompt in this shape:

```text
Review this change independently. Look for bugs, behavioral regressions, invalid assumptions, edge cases, and missing tests. Also ask whether the refactor fits in the overall architectural direction, including for future work and scaling.

Task summary:
- ...

Constraints:
- ...

Implemented change summary:
- ...

Changed files:
- ...
```

Avoid prompts in this shape:

```text
I fixed the issue cleanly. Please sanity-check the solution.
```

The point of the review agent is to pressure-test the work, not to confirm a positively framed story.
