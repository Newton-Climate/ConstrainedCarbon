# `ecosys analyze`

Use these commands after you have fitted sites and want to answer a broader scientific question.

| Subcommand | Question it helps answer |
|---|---|
| `model` | How well does one fitted model reproduce its observations? |
| `network` | How do results compare across a set of sites? |
| `transit` | How long does carbon remain in the modeled soil system? |
| `transit-vulnerability` | Does transit time help explain warming vulnerability across sites? |
| `cross-ecosystem` | What is the overall pattern across ecosystems? |

```bash
ecosys analyze network \
  --site-set configs/site_sets/direct_warming_network_24.yaml \
  --include-er-constraint --workers 4

ecosys analyze transit --mode intrinsic
```

These analyses use different input tables and assumptions, so check `ecosys analyze <subcommand> --help` before running one.
