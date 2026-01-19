# AGENTS.md

## Build System

- We use CMake as the build system.
- Use make.sh script.

## Commit

- Format files using script/format.sh before commiting

Example COMMIT:
```
Refactor user service to reduce duplication

Extract shared validation logic into a helper module,
simplifying maintenance and improving test coverage.
```

## Issue Tracking                                                                                   │
                                                                                                    │
This project uses **bd (beads)** for issue tracking.                                                │
Run `bd prime` for workflow context.
                                                                                                    │
**Quick reference:**                                                                                │
- `bd ready` - Find unblocked work                                                                  │
- `bd create "Title" --type task --priority 2` - Create issue                                       │
- `bd close <id>` - Complete work                                                                   │
- `bd sync` - Sync with git (run at session end)                                                    │
                                                                                                    │
For full workflow details: `bd prime`
