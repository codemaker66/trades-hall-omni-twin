---
id: TRACK-1
track: track-name
title: Command title
depends_on: []
source_technique: TECHNIQUE_X.md
session_boundary_required: true
status: ready
---

## OBJECTIVE
Deliver one atomic implementation outcome for this command.

## CONTEXT
Explain how this command fits into the track and why it exists.

## TECHNICAL SPECIFICATION
Specify exact implementation behavior, data flow, algorithms, and constraints.

## FILES TO CREATE
List exact files and required public interfaces.

## TESTS REQUIRED
List concrete tests with expected behavior.

## EDGE CASES
List edge cases that must be handled and tested.

## ANTI-SKELETON RULES
- Implement complete logic.
- Validate all inputs.
- Handle errors explicitly with meaningful messages.
- Avoid placeholder comments.

## VERIFICATION
- [ ] `npm run typecheck`
- [ ] `npm run test`
- [ ] `rg -n "<placeholder-pattern>" src/` returns no matches
- [ ] `rg -n "<any-or-unsafe-type-pattern>" src/` returns no matches

## INTEGRATION CONTRACT
- PREREQUISITES: List command IDs and interfaces required before execution.
- EXPOSES: List outputs available to future commands.
- DO NOT REFERENCE: List modules not yet available.

## COMPLETION CRITERIA
1. All files and interfaces are implemented.
2. Tests and verification checks are complete.
3. Update PROGRESS.md with implemented scope and test results.
4. Commit and stop.
