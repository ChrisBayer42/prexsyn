# Snyk Security Setup — Design

**Date:** 2026-02-22
**Status:** Approved

## Goal

Add automated security scanning to the PrexSyn repository covering both
third-party dependency vulnerabilities and source-code static analysis (SAST).

## Scan scope

| Scan | Snyk command | Triggers |
|---|---|---|
| Dependency vulnerabilities | `snyk test` | Push to `main`, every PR |
| Source code (SAST) | `snyk code test` | Push to `main`, every PR |
| Dashboard snapshot | `snyk monitor` | Daily cron (01:00 UTC) |

## Dependency export approach

The project uses pixi (conda-based), which Snyk does not natively support.
In CI, dependencies are resolved by running `pip install .` in a clean Python
environment, then `pip freeze > requirements.txt`. This captures exact pinned
versions of all installed packages and is fed to `snyk test --file=requirements.txt`.

## GitHub Actions workflow

Single file: `.github/workflows/snyk.yml`

**Job 1 — `snyk-security`** (push to `main` + PRs):
1. Checkout code
2. Set up Python 3.11
3. `pip install .`
4. `pip freeze > requirements.txt`
5. `snyk test --file=requirements.txt --package-manager=pip --severity-threshold=high`
6. `snyk code test --severity-threshold=high`

**Job 2 — `snyk-monitor`** (daily cron, 01:00 UTC):
1. Checkout + pip install + pip freeze (same as above)
2. `snyk monitor --file=requirements.txt` — pushes snapshot to Snyk dashboard

## Failure behaviour

- `--severity-threshold=high` on both scan commands: low/medium findings are
  reported in the log but do not fail the build. High/critical findings fail it.
- `snyk monitor` never fails the build (informational only).

## Required secret

`SNYK_TOKEN` — permanent API token from https://app.snyk.io/account
Added under: GitHub repo → Settings → Secrets and variables → Actions
