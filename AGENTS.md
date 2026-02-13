# Agent Guidelines

- Always create small, focused commits.
- Never delete large files without asking first.
- Always leave the exact commands to run after changes.

## Canonical Command Framework

- Canonical framework: `docs/AGENT_COMMAND_FRAMEWORK_v1.md`.
- Canonical command specs: `docs/commands/**`.
- `TECHNIQUE_*.md` files are research/reference sources and do not override canonical command specs.

## Command Execution Workflow

- Execute one command spec per session.
- Respect command `depends_on` and do not skip integration checkpoints.
- Before completion, run command verification checks.
- After completion:
  - Update `PROGRESS.md` using the command completion format.
  - Commit and stop.

## Framework Validation

- Run `npm run validate:agent-framework` before opening a PR.
- A command spec is valid only when schema, dependency, and technique-link checks all pass.
