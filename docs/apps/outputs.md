# Reading `ecosys` outputs

This is the common guide for every app. Each app page explains its own files
in more detail.

## Start with the manifest

Every finished run has a `manifest.json`. Open it first. It says what command
ran, which inputs it used, which Git revision produced it, and which files
belong to the run.

Outputs normally live here:

```
outputs/<site-or-run-name>/<command>/<subcommand>/
```

Use `--outdir` to choose another root and `--name` where the app supports it.
A rerun using the same destination replaces files with the same names, so give
important runs a distinct name or output root.

## The integration contract

The manifest is the authoritative file list. Do not guess filenames from the
command name. Contract version 1.1 records `status: complete`, the absolute
`output_dir`, and every artifact relative to that directory.

In Python, `RunDir.finalize()` returns the same information:

```python
{
    "output_dir": "/absolute/run/directory",
    "manifest": "/absolute/run/directory/manifest.json",
    "files": {"summary.csv": "/absolute/run/directory/summary.csv"},
}
```

`ecosys analyze` and `ecosys report` print this record when they finish. A run
without a manifest was interrupted or failed; do not treat its partial files as
a completed result.

Use Parquet for analysis and CSV for a quick look. NumPy `.npz` files hold named
arrays for technical follow-up; they are not intended to be read like a sheet.

## Which output answers which question?

| Workflow | Begin with | It helps answer |
|---|---|---|
| `optimize` | `diagnostics.json`, then `summary.parquet` | Did the fit converge, and what turnover parameters were fitted? |
| `warming` | `summary.parquet` | What does this fitted model project under one stated warming experiment? |
| `information shapley` | `metrics.parquet`, `shapley_by_parameter.parquet` | Which observations locally resolve which fitted parameters? |
| `model run` | `diagnostics.json`, `forward_output.npz` | What does the prescribed forward simulation produce? |
| `mcmc` | generated README and regression tables | How uncertain are a cross-site relationship and its robustness checks? |
| `analyze` / `report` | `manifest.json` | Which tables and figures were produced for this synthesis? |

## A safe reporting habit

Keep the manifest, configuration snapshot when present, and source tables next
to every figure. Say whether a result is a fitted parameter, a diagnostic, or a
model projection. For cross-site comparisons, also state the site set, forcing,
observation choices, and uncertainty method.

For a concrete worked example, see [Harvard Forest: reading a complete run](harvard-forest-example.md).
