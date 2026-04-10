---
name: commit
description: This skill should be used when the user asks to "commit", "commit changes", "make a commit", "git commit", or wants to save their work to git. Invoked via /commit.
---

## Commit/Merge Workflow

Commit only the changes made during the current session unless the user explicitly asks to commit other changes. Follow these steps:

### Step 1: Identify Session Changes

Run `git status` and `git diff` (staged + unstaged) to see all modified, added, and deleted files.

Compare against the session context — only stage files that were created or modified as part of the current conversation. If uncertain whether a file was changed in this session, ask the user.

### Step 2: Check Code Quality (MANDATORY)

Before committing, check ALL changed Python files for:

1. **Unused imports** — imports that are not referenced anywhere in the file
2. **Imports inside functions** — move these to the top of the file unless there is a clear circular-import or conditional-import reason
3. **Linting errors** — run ruff to check for any linting issues and fix them
4. **Type errors** — run pyright to check for type errors and fix them

**For large changesets (4+ files changed):** Use fast subagents in parallel to check each file simultaneously. Each subagent should read one file and report any unused imports or function-level imports found.

**For small changesets (1-3 files):** Check the files directly without subagents.

If issues are found, fix them before proceeding to the commit. Do NOT ask the user — just fix them silently and include the fixes in the commit.

### Step 3: Tests

Make sure we ran all relevant tests already. If not, run them now and fix any failures before committing.

### Step 4: Update AGENTs.md, and README.md if relevant

If the commit includes important changes in API or functionality, update the `AGENTS.md` documentation file to reflect these changes. This ensures that the documentation stays accurate and helpful for future reference. Also check and update the `README.md` files if relevant.


### Step 5: Stage and Commit

1. Stage only the session-relevant files by name (never `git add -A` or `git add .`)
2. Run `git log --oneline -5` to match the repository's commit message style
3. Write a concise commit message (1-2 sentences) focusing on the "why" not the "what"
4. Run `git status` after to verify success


### Important Rules

- Never use `git add -A` or `git add .`
- Never push unless explicitly asked
- Never amend previous commits unless explicitly asked
- Never skip pre-commit hooks
- If a pre-commit hook fails, fix the issue and create a NEW commit (do not amend)
- Do not commit files that likely contain secrets (.env, credentials, tokens)
