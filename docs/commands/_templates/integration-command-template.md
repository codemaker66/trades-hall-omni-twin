---
id: TRACK-INT-1
track: track-name
title: Integration command title
depends_on: []
source_technique: TECHNIQUE_X.md
session_boundary_required: true
status: ready
---

## OBJECTIVE
Integrate a bounded set of previously completed commands into one working flow.

## CONTEXT
Explain why this integration checkpoint exists and what risk it reduces.

## TECHNICAL SPECIFICATION
Define the end-to-end wiring, interfaces, and expected runtime behavior.

## FILES TO CREATE
List integration files, adapters, and integration tests.

## TESTS REQUIRED
List end-to-end and regression tests for integrated behavior.

## EDGE CASES
List integration failures and mismatch scenarios that must be handled.

## ANTI-SKELETON RULES
- No partial wiring.
- No silent fallback behavior.
- No placeholder sections.

## VERIFICATION
- [ ] `npm run typecheck`
- [ ] `npm run test`
- [ ] `rg -n "<placeholder-pattern>" src/` returns no matches
- [ ] `rg -n "<any-or-unsafe-type-pattern>" src/` returns no matches

## INTEGRATION CONTRACT
- PREREQUISITES: Commands included in this checkpoint.
- EXPOSES: End-to-end flow and test coverage for downstream commands.
- DO NOT REFERENCE: Future track commands not yet completed.

## COMPLETION CRITERIA
1. Integrated flow is executable end-to-end.
2. Integration tests pass.
3. Update PROGRESS.md with integrated scope and gotchas.
4. Commit and stop.
