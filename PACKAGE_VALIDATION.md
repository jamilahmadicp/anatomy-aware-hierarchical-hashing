# Package validation

The repository was checked before packaging with the following local validation steps:

- Python syntax compilation of `anatomy_hash/` and `scripts/`.
- Unit tests: **8 passed**.
- Bash syntax checks for the supplied `.sh` runners.
- Editable package installation with dependencies disabled, confirming that `import anatomy_hash` resolves correctly.
- Repository-wide text scan for temporary development notes and cache artifacts.

The PowerShell runners were structurally reviewed but were not executed in the Linux validation environment. Full GPU training is intentionally not part of package validation; final scientific results must come from the documented experiment runs on the target hardware and datasets.
