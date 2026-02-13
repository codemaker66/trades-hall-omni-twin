# Agent Command Framework v1
## Engineering Standard for Command-Driven Delivery

This framework defines how command specifications are written for coding agents.
The goal is to prevent shallow implementations, missing integration, and weak verification.

## Core Problem Patterns

Agents often fail in predictable ways:
- Skeleton implementations (signatures without real logic)
- Uneven depth (first part deep, later parts shallow)
- Happy-path-only logic with weak edge case handling
- Thin wrappers around libraries with little added value
- Placeholder comments replacing implementation
- Missing integration between modules
- Cosmetic tests that validate nothing meaningful
- Hardcoded configuration and silent error handling
- Type declarations that do not constrain behavior

## Structural Rules

### 1. Atomic Scope
Each command must deliver one atomic sub-domain that can be completed at full depth in one focused session.

### 2. Concrete Deliverables
Each command must specify exact files and public interfaces.

### 3. Verification Gates
Each command must include objective checks that run before completion claims.

### 4. Anti-Skeleton Rules
Commands must require complete logic, explicit validation, explicit error handling, and meaningful tests.

### 5. Integration Contract
Each command must define prerequisites, exports, and forbidden forward references.

### 6. Edge Case Coverage
Commands must list edge cases that are required to be implemented and tested.

## Required Command Template

All command specs must contain:
- YAML front matter metadata
- Sections:
  - OBJECTIVE
  - CONTEXT
  - TECHNICAL SPECIFICATION
  - FILES TO CREATE
  - TESTS REQUIRED
  - EDGE CASES
  - ANTI-SKELETON RULES
  - VERIFICATION
  - INTEGRATION CONTRACT
  - COMPLETION CRITERIA

## Sequencing Rules

- Commands follow dependency order with no forward references.
- Insert integration commands every 3-4 implementation commands.
- Each session completes one command, updates progress tracking, commits, and stops.

## Session Boundary Rules

A command is complete only when:
1. Specified files and interfaces are implemented.
2. Required tests and verification checks pass.
3. Progress log is updated with implemented scope and test outcomes.
4. Commit is created and the session stops.

## Quality Bar

A command spec is acceptable only when another engineer can execute it with zero unresolved decisions.
