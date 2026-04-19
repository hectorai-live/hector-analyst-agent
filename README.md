# Hector Analyst Agent — Live Diagnostic

This branch (`gh-pages`) serves the static founder-ready dashboard via GitHub
Pages. The real source lives on `claude/continue-agent-build-axGjd` (or `main`
after merge).

To regenerate:

```bash
git checkout main              # or the dev branch
python scripts/build_dashboard.py
cp docs/dashboard.html <path-to-gh-pages-worktree>/index.html
```

Once Pages is enabled (Settings → Pages → Source: `gh-pages` branch → `/` root)
the dashboard is served at:

    https://<org>.github.io/hector-analyst-agent/
