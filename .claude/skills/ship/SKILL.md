---
description: Commit and push changes for lrp-coach. Use when the user says "push", "ship", or "commit and push".
disable-model-invocation: true
allowed-tools: Bash
---

IMPORTANT: Never push without explicit user confirmation. The user's standing rule is "ask me before any push".

1. Run `git status` and `git diff` to see what changed.
2. Stage relevant files (avoid secrets, binaries, .venv/).
3. Write a commit message explaining *why* the change was made.
4. Show the user the staged files and commit message.
5. **Ask: "Ready to push to origin/main?"** — wait for confirmation.
6. Only after confirmation: `git push origin main`.

If $ARGUMENTS is provided, use it as context for the commit message.
