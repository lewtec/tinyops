# AGENTS.md

Project law and agent procedure live in **[SPEC.md](./SPEC.md)**. Read and follow that document.

Compat scope backlog: **[CHECKLIST.md](./CHECKLIST.md)** (maintainer adds items; agents implement and mark done when tests pass).

## Commands

```bash
mise run test                              # full test suite (CI)
mise run test -- -k resize                 # subset by test name
mise run test -- --cov=tinyops.compat --cov-fail-under=100   # compat coverage gate
mise run lint                              # ruff
mise run ci                                # lint + test
```
