# Contributing

Thanks for your interest. This is an experimental research pipeline; contributions that
improve rigour, reproducibility or honesty of results are especially welcome.

## Ground rules

Read `SPEC.md` section 0 first. In short:

1. No public disclosure of newly detected candidate locations (see `SECURITY.md`).
2. Never fabricate identifiers, dates, dataset names or accuracy numbers. Record failures.
3. Credentials only from environment variables (`.env`, git-ignored). Never commit them.
4. Desk analysis only. No fieldwork planning, routes or access instructions.
5. Honest output: confidence and validation scores on every result; "candidate area for
   verification", never "waste found". Negative results are valid results.
6. Every raster carries a `*.prov.json` provenance sidecar.

## Development setup

```
uv sync                 # core + dev tools (enough for the whole test suite)
uv run pytest           # offline: tests never touch the network
uv run ruff check src tests
uv run ruff format src tests
```

Heavy extras (`uv sync --all-extras`: torch, sam2, earthengine-api, anthropic) are only needed
for real runs. Tests that need them are marked `@pytest.mark.heavy` and skipped otherwise.

## Code conventions

- Shared contracts (tile grid, score tables, provenance, restricted outputs) are in
  `ARCHITECTURE.md`. Change a contract there and in code together.
- External services sit behind small client interfaces with fakes for tests; add a fixture,
  not a network call.
- Anything with candidate coordinates is written through `sbf.core.restricted` and never logged
  (log counts and IDs only).
- Tests use clearly synthetic data (`TEST_FIXTURE_*` IDs, generated rasters). Do not commit
  real imagery, real scene metadata dumps or anything from `data/` or `outputs/`.

## Pull requests

Keep PRs small and focused, include tests, and fill in the PR template checklist. CI runs the
test suite, ruff, and a guard that fails if restricted or data files are tracked.
