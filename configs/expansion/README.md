# Expansion Site Configs

These configs are intentionally kept outside `configs/multisite/` so they do not
enter the default `ecosys optimize --all` sweep before their FluxCom GPP
series have been added under `data/shared/fluxcom/`.

They target ecosystem classes that are underrepresented or absent in the current
tower-backed multisite set:

- `luquillo_experimental_forest.yaml` — tropical wet montane forest
- `hi_andisol.yaml` — tropical volcanic Andisol forest
- `ca_mollisol.yaml` — Mediterranean coastal grassland
- `az_mollisol.yaml` — semi-arid montane grassland

Run one explicitly once its FluxCom file exists:

```bash
ecosys optimize configs/expansion/luquillo_experimental_forest.yaml
ecosys analyze model configs/expansion/luquillo_experimental_forest.yaml
```
