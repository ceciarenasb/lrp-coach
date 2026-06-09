---
description: Run tests and linting for lrp-coach. Use when the user wants to check, test, or verify the code.
allowed-tools: Bash
---

Run tests and linter:

```bash
make test
make lint
```

Report:
- How many tests passed / failed
- Any lint errors (file + line)
- Overall status: PASS or FAIL

If tests fail, show the failing test name and error. If lint fails, show each violation. Do not attempt fixes unless asked.
