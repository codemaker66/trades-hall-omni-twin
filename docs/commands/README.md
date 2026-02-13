# Command Specs

This directory is the canonical source of agent command specifications.

## Authority
- Execution authority is defined by files in `docs/commands/**`.
- `TECHNIQUE_*.md` files are research references and source context.

## Required Metadata
Every command file must include YAML front matter:
- `id`
- `track`
- `title`
- `depends_on`
- `source_technique`
- `session_boundary_required`
- `status`

## Required Sections
Every command file must include all required `##` headings from the framework.

## Track Layout
- `stochastic-pricing/`
- `physics-solvers/`
- `hpc/`
- `stat-learning/`
- `signal-processing/`
- `optimal-control/`
- `gnn/`
- `computer-vision/`

## Templates
Use templates in `_templates/` for both implementation and integration commands.
