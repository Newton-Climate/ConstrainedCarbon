# Multi-site ISRaD radiocarbon inversion — workflow & findings

This documents an end-to-end effort to run one soil-carbon optimal-estimation (OE)
inversion recipe across every ISRaD **field-flux + respiration** site, driven by
**real flux-tower GPP**, and constrained by radiocarbon plus SoilGrids soil-carbon
stocks.

- **Env:** model code (jax, `ecosystem_complexity.*`) runs under the conda env
  `ecosystem-complexity`:
  `/Users/newtonnguyen/miniforge3/envs/ecosystem-complexity/bin/python`.

---

## 1. Site selection — ISRaD eligibility & flux colocation

`apps/locate_site.py` now handles the generalized site-lookup workflow across
**all 5 ISRaD flat
datatypes** (layer, fraction, flux, incubation, interstitial) and given a
site-level eligibility rule:

> A site is eligible if it has **fraction ¹⁴C**, **or** **bulk/layer ¹⁴C together
> with respiration ¹⁴C** (respiration = field **flux** and/or lab **incubation**).

Outputs:
- `exports/israd_eligible_sites.csv` — 241 eligible ISRaD sites with per-datatype
  obs counts and an `eligibility_path` flag (`fraction` / `pool+respiration` / both).
- `exports/flux_tower_israd_colocations.csv` (+ priority/best subsets) — nearest
  eligible ISRaD site per AmeriFlux/NEON tower.

The **field-flux + bulk-layer** subset (20 sites) became the inversion target set.
The best-colocated sites (<1 km) are overwhelmingly **forest**, spanning
boreal → temperate → Mediterranean → tropical, with single tundra / cropland
representatives.

---

## 2. Forcing data — GPP downloaders

Each inversion is driven by the observed **daily GPP time series** from the
co-located flux tower (not a constant mean). Two self-contained downloaders were
written; both are idempotent and delete the archive after extraction.

### `apps/download_flux.py` — AmeriFlux FLUXNET
- Uses the AmeriFlux data-download REST API (same endpoint as the `amerifluxr`
  R client). Credentials come from a gitignored `.env`
  (`AMERIFLUX_USER_ID`, `AMERIFLUX_EMAIL`); download requires `--accept-data-policy`.
- Correct FLUXNET payload: `data_product=FLUXNET`, `data_variant=FULLSET`,
  controlled-vocabulary `intended_use`, response URLs at `data_urls[].url`.
  Falls back to `BASE-BADM` per-site where no FLUXNET release exists.
- Pulled: **BR-Sa3, BR-Ma2, CA-NS1, CA-NS4** (+ US-Ha2 BASE). US-Ho1 / US-Ha1
  were already in the repo.

### `apps/download_flux.py` — ICOS Carbon Portal (multi-network)
- The ICOS portal is a **multi-network hub**: it serves ONEFlux FLUXNET products
  (same FLUXNET2015 format as AmeriFlux) from several networks by filename prefix:
  `ICOS_` (Europe), `EUF_` (European Fluxes DB), `JPF_` (JapanFlux/Asia).
- Colocates the international field-flux ISRaD sites against the full 474-station
  ICOS catalogue via SPARQL, resolves each station's FLUXNET product, and
  downloads under CC-BY-4.0 (licence cookie; gated by `--accept-license`).
- Pulled across **3 continents + Arctic**: Auchencorth Moss/UK (UK-AMo),
  Adventdalen/Svalbard (SJ-Adv), Solling/DE (DE-Rns), Appi/Japan (JP-Api),
  Baram/Malaysia (MY-LHP).

> All AmeriFlux FULLSET and ICOS FLUXMET DD files share the same column
> convention, so a single loader (`load_howland_forest`) reads them all. GPP is
> re-extracted robustly (`_load_gpp_series`: best-populated of the NT/DT × VUT/CUT
> columns, gap-filled) because the loader's QC filter nukes ~75 % of GPP days and
> hardcodes `GPP_NT_VUT_REF`, which is empty for the tropical BR sites.

---

## 3. Soil-carbon stocks — `download_soilgrids_soc.py`

ISRaD `lyr_soc` is empty at these sites, so **¹²C SOC stocks** come from
**SoilGrids v2.0** (ISRIC). SOC content + bulk density + coarse-fragment fractions
are fetched by depth and converted to a stock, prorated onto the pool depth bins:

```
SOC_stock(gC m⁻²) = 10 · SOC(g/kg) · BD(g/cm³) · thickness(cm) · (1 − coarse)
```

→ `exports/soilgrids_soc_pools.csv` (site × pool × stock × sigma).

**Peat/permafrost guard:** SoilGrids over-estimates SOC in organic soils (BD not
lowered enough). Sites with implausible totals (>60 000 gC m⁻² to 1.3 m) —
CZ_Old_Black_Spruce, CZ_1964burn, Baram, Adventdalen — have the C-stock constraint
dropped; those regions are peatlands outside the 3-pool *mineral* model anyway.

---

## 4. The inversion — `sites/multisite_canonical.py`

**One canonical recipe, one config YAML per site.** Every site has its own
transparent, version-controlled config in `configs/multisite/<site>.yaml`. Each
file carries the *same* canonical `model` / `external_inputs` / `parameters` /
`inversion` recipe (so cross-site results stay comparable) plus a site-specific
`site` block (id, name, lat/lon, tower_id, biome) and `datasource` block
(`israd_name`, `forcing_glob`, `forcing_kind`) read by the driver. Adding a site
is a copy-and-edit of one YAML — no code change. The shared recipe is applied
identically at every site:

- 3 SOM pools (active / slow / passive), canonical depth structure & τ priors
- external soil input = observed daily `GPP_obs`
- `optimize_oe` (Levenberg–Marquardt) over fields `(log_tau, log_f_transfer)`
- observation vector:
  - **bulk-layer Δ¹⁴C** → pools by depth (ISRaD layer table), one block per obs year
  - **field-flux Δ¹⁴C** → respired CO₂ (ISRaD flux table)
  - **SoilGrids SOC** → per-pool C-stock (where plausible)

Result → `exports/multisite_canonical_inversions.csv`.

### Results (with C-stock constraint)

| ISRaD site | tower | region | GPP (gC/m²/yr) | SOC (gC/m²) | τ active | τ slow | τ passive | converged |
|---|---|---|--:|--:|--:|--:|--:|:--:|
| ZF2 | BR-Ma2 | Amazon | 3247 | 13 206 | 2.0 | 19.8 | 204 | ✓ |
| Baram | MY-LHP | Borneo | 3059 | *peat* | 2.0 | 21.2 | 199 | ✓ |
| FLONA | BR-Sa3 | Amazon | 3131 | 11 768 | 2.0 | 18.0 | 214 | ✓ |
| Solling | DE-Rns | Germany | 1377 | 26 518 | 1.9 | 20.6 | 230 | ✓ |
| Appi | JP-Api | Japan | 1255 | 44 327 | 1.7 | 23.2 | 162 | ✓ |
| Auchencorth | UK-AMo | UK peat | 756 | 47 873 | 1.6 | 15.9 | 172 | ✓ |
| Adventdalen | SJ-Adv | Arctic | 127 | *peat* | 1.6 | 17.9 | 245 | ✓ |
| Howland | US-Ho1 | Maine | 915 | 35 498 | — | — | — | ✗ (NaN) |
| CZ_Old_Black_Spruce | CA-NS1 | boreal | 655 | *peat* | — | — | — | ✗ (NaN) |
| CZ_1964burn_NSA | CA-NS4 | boreal | 289 | *peat* | — | — | — | ✗ (NaN) |

**7 of 10 converged**, with real GPP forcing spanning 127 (high-Arctic tundra) to
3250 gC/m²/yr (tropical Borneo) under one recipe.

---

## 5. Findings & caveats

**Global turnover gradient recovered.** Fast active pools (~2 yr) everywhere;
slow ~16–23 yr; passive ~160–245 yr, all from a single uniform recipe driven by
site-specific GPP.

**3 sites fail with a NaN (Howland, CA-NS1, CA-NS4).** The failure is numerical,
not data: the **bulk-layer Δ¹⁴C block gradient goes NaN** in `optimize_oe`'s
Jacobian (resp-only converges; pool-only NaNs, at both single- and year-averaged
time indices). It is a reverse-mode autodiff edge case in the single-pool Δ¹⁴C
trajectory and would need a gradient-safety fix in `tracer_14C.py` / `soil.py`.

**The SoilGrids C-stock is active but soft.** At the default 50 % sigma it barely
moves the fit (e.g. FLONA τ_passive 208→214 yr) — the constraint is real but
throttled. Tightening sigma makes it bite (FLONA τ_passive sweep below), at the
cost of ¹⁴C fit, because SoilGrids SOC and the ¹⁴C-consistent turnover genuinely
disagree (SoilGrids ~5764 gC/m² passive vs ¹⁴C-consistent ~3300):

| SOC sigma | τ_passive | passive C12 (target 5764) |
|---|--:|--:|
| none | 208 yr | 3295 |
| 50 % (default) | 214 yr | 3414 |
| 20 % | 251 yr | 4167 |
| 10 % | 308 yr | 5275 |

**The active-pool SOC target is a category error.** The active *kinetic* pool
(τ≈2 yr) structurally cannot hold the depth-integrated 0–10 cm SOC (~2558 gC/m²);
its C12 stays ~1475 at every sigma. Depth ≠ kinetics — SOC should constrain the
**slow + passive** pools only.

**Constraints remain weak at data-poor sites.** The deep pool stays near its prior
where ¹⁴C is sparse (ZF2/Baram have 1–2 ¹⁴C obs) and tower records are short — a
limit SOC alone can't remove.

**Other limitations.**
- ISRaD sampling years often predate the tower GPP record, so out-of-window
  Δ¹⁴C obs are clamped to the nearest date.
- SoilGrids point values are 250 m pixels; peat/permafrost stocks are unreliable
  (guarded, see §3).

---

## 6. How to run

```bash
PY=/Users/newtonnguyen/miniforge3/envs/ecosystem-complexity/bin/python

# 1. site selection
ecosys config locate --flux-tower US-Ha1 --out notebooks/exports/flux_tower_israd_colocations.csv

# 2. forcing (needs .env for AmeriFlux; ICOS is CC-BY)
ecosys fetch flux harvard_forest --accept-policy
ecosys fetch flux solling --accept-license

# 3. soil-carbon stocks
$PY notebooks/download_soilgrids_soc.py

# 4. config + inversion + analysis
ecosys config build --selector harvard_forest
ecosys optimize solling
ecosys analyze model solling
```

Every `ecosys <verb>` run writes its artifacts under `./outputs/{site_or_set_name}/{verb}/`
with a `manifest.json` declaring the file list (see
`src/ecosystem_complexity/outputs.py` for the shared contract).

## 7. Files

| File | Purpose |
|---|---|
| `apps/fetch.py` | `ecosys fetch {flux\|fluxcom\|clm\|israd\|atm14c}` — download / stage forcing and observation data |
| `apps/config.py` | `ecosys config {build\|incubation\|locate}` — config-authoring utilities |
| `apps/optimize.py` | `ecosys optimize` — canonical OE inversion for site / site-set / sweep |
| `apps/analyze.py` | `ecosys analyze {model\|network\|transit\|transit-vulnerability\|cross-ecosystem}` |
| `notebooks/download_soilgrids_soc.py` | SoilGrids ¹²C SOC stocks by pool |
| `configs/israd_multisite_3pool_config.yaml` | the shared recipe template (source for the per-site configs) |
| `configs/multisite/<site>.yaml` | one transparent config per site (recipe + `site`/`datasource`) |
| `src/ecosystem_complexity/sites/driver.py` | shared per-site OE driver used by the apps |
| `notebooks/exports/israd_eligible_sites.csv` | eligible ISRaD sites |
| `notebooks/exports/soilgrids_soc_pools.csv` | SOC stocks per site/pool |
| `notebooks/exports/multisite_canonical_inversions.csv` | inversion results |
