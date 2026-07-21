# HANDOFF — LIG SST Data Mining (Anna Chen, HISEAS)

Read this file first. It is a complete brief; you should not need any other context.
Everything referenced lives in `SST-RSOI-Git/AC_essential/`.

---

## 1. The overall project

**HISEAS asks: how much and why did ice sheets melt during the Last Interglacial (LIG)?**

Anna's part is to reconstruct **global full-field sea-surface temperature (SST) maps for
130–115 ka at 0.1 ka steps**, using Reduced Space Optimal Interpolation (RSOI).

- Modern HadISST (1870–2020) is decomposed by EOF; the leading 20 modes form a reduced space.
- Sparse LIG proxy SSTs from Hoffman et al. (2017) are projected into that space by optimal
  interpolation, then expanded back to a full field.
- The scheme is iterated (iterations 0–5) to re-estimate the assumed mean **M** and covariance
  **C**, following Alexey Kaplan's write-up.
- Code: `AC_essential/SST_RSOI.ipynb`. It is finished and runs.
- Method docs: `0223_RSOI_Methodology.pdf` (Anna's) and `0306_Methodology_From_Alexey.pdf`
  (Alexey's correction to section 2.5). **Where they conflict, Alexey's maths wins.**

Mentors: **Alexey Kaplan** (method/statistics), **Jerry McManus** (paleoceanography, data).

## 2. What we are doing right now — data mining

The reconstruction needs **independent validation data**: SST time-series from ocean cores that
are *not* already in Hoffman et al. (2017). We compare the reconstruction against them via
**bias, RMSE/uRMSE, and correlation r** — the same idea as Anna's other project (GOBM vs
independent data, with Galen).

We also want data **older than Hoffman's 130 ka limit**, back to **140 ka**, to cover the
penultimate deglaciation (Termination II).

**Target window: 115,000–140,000 cal yr BP.** Only data inside this window goes in the sheets.

### Acceptance rules
A core is accepted if all of these hold:
1. `"sea surface temperature"` / `"sea-surface temperature"` / `SST` appears in the header or
   metadata (searched in study name, notes, keywords, publication title, data-table notes,
   file link text, variable definitions — **not** abstracts).
2. Marine ocean core (`locationName` under `Ocean>`), elevation negative.
   If everything else passes but elevation is not negative → flag, don't reject.
3. Has a real **age/time** variable, not just depth. (Depth-only → Jerry's file.)
4. Each SST value has its **own** inferred age. Interval-average products (e.g. "Early LIG
   maximum annual temperature", Turney et al.) are **rejected** and flagged.
5. Not already held. Dedup against **both** `Hoffman_Metadata.csv` (103 cores) **and** Anna's
   existing 24 tabs, by **1°×1° cell** (`floor(x)+0.5`) **and** by normalised core ID.
   Grid alone is not enough — see Mental Note M1.
   Duplicates go to `same_core_different_chronology.xlsx`, not the main sheet.
6. Proxy outside Hoffman's six (`UK37, Foram, MgCa, Radiolaria, Diatom, Cocc`) → still extract,
   but flag it.

Sparse cores (**fewer than 5 points** in-window) are **kept and flagged**, not rejected —
unless specifically excluded below.

## 3. Most important files

| File | What it is |
|---|---|
| `SST_Data_Mining.xlsx` | **Anna's original. NEVER modify.** 24 cores she mined by hand. |
| `SST_Data_Mining_corrected_2026-07-20.xlsx` | **The working copy — all edits go here.** |
| `same_core_different_chronology.xlsx` | Cores duplicating Hoffman/Anna's, kept for comparison |
| `SST_Depth_Only_For_Jerry.xlsx` | Cores with SST vs depth but no age model |
| `need_to_confirm.docx` | Anna's review doc. **Contains 66 of her comments + section I specs.** |
| `rejected_with_reason.docx` | Rejections with reasons. **Contains 11 of her comments.** |
| `Hoffman_Metadata.csv` | 103 Hoffman cores with lon/lat — the dedup reference |
| `ncei_search_full.json` | 2.6 MB, 122 studies — the full NCEI search result |
| `mined_txt/` | 188 downloaded NCEI data files + Anna's manual additions |
| `parser2.py` | Current parser (multi-table). See §5. |
| `manual_specs.py` | Hand-mapped columns for files parser2 can't resolve |
| `screened.json` | Screening decisions: `acc`, `chrono`, `jerry`, `flag`, `rej` |
| `flags.json` / `rejects.json` | Feed the two .docx reports |
| `corrections_log.json` | The 20 corrections applied to Anna's own tabs |
| `write_outputs.py` / `write_docs.py` | Build the spreadsheets / the reports |

## 4. Spreadsheet column format (must match exactly)

Row 1 = headers, row 2 = first data row **and** core metadata. One tab per core; add
`_{proxy}` only when one study reports several proxies for one core (e.g. `ODP1148_UK37`).

| Col | Content |
|---|---|
| A | Core ID (no proxy suffix) |
| B, C | Longitude, Latitude — decimal degrees, numeric |
| D | Proxy type |
| E–K | **Leave blank** (Hoffman-specific) |
| L, M | Benthic d18O depth / value, if present |
| N | Depth — **always cm** (convert m → cm) |
| O | Proxy value (e.g. the Mg/Ca or UK37 number) |
| P | SST — copy of col T unless the study interpolated (then flag) |
| Q | Blank |
| R | Age — **always years BP** |
| S, T, U | SST −2sd, SST mean, SST +2sd |
| V | Publication citation (DOI if available) |
| W | Direct URL to the data file |
| X | Chronology / age-model info, quoted from source |
| Y | Method — calibration used **and** alternatives, proxy species, seasonality, unit conversions |
| Z | Elevation (negative; NaN if unknown) |

**Rules for the data:**
- Age in years BP; only rows with 115,000 ≤ age ≤ 140,000.
- SST sanity bounds **−5 °C to +40 °C**. (Do *not* use a 10 °C floor — that would delete real
  Southern Ocean data.)
- 1sd → **double** it for ±2sd. Free-text statements count too (e.g. "error of ±1.5 °C").
- The error column must belong to **SST** — never `age_uncert`, never a proxy or d18O error.
- **Seasonal SST:** if August *and* February columns both exist, annual = their mean; average
  the two errors then double. Record the original Aug and Feb values in col Y. If only one
  season exists, keep it and flag the core as a seasonal/summer-signal core.

## 5. Workflow and hard-won technique

**Token efficiency is the binding constraint.** Pulling data through the web tool is what
exhausts context; local parsing is free. Use this pattern:

1. `mcp__workspace__web_fetch` caps responses near 100k chars and **spills oversized results to
   a file on disk**. That file is *not* reachable from bash, but **is** reachable by `Read` and
   `Grep`.
2. So: fetch big → let it spill → `Grep` with `-o` and a tight regex to pull only the fields you
   need. This cut a 27k-token payload to ~5k.
3. Better still: have Anna save the JSON locally (she already did — `ncei_search_full.json`),
   then parse with bash/python at **zero** context cost.
4. NCEI API notes: `headersOnly=true` gives ~100 tokens/study instead of ~7,000. `page` is
   **not** supported. URLs longer than ~200 chars are rejected by the fetch tool.
5. Never fetch URLs with curl/requests — not permitted. Use the web tool or ask Anna.

**Do the heavy work in Python over `mined_txt/`, not over the network.**

### Parser design (`parser2.py`)
NOAA txt files are irregular. Bugs already fixed — do not reintroduce:
- Files often hold **several numeric tables**; the SST series is frequently in table 2 or 5,
  with chronology or isotopes in table 1. Parse **all** tables.
- Column names may live in prose (`Column 1: ...`) above a headerless table — `legends()`.
- Delimiter priority **tabs > 2+ spaces > commas**. Splitting on commas breaks `"Depth,cm"`.
- A leading label column can span several fields (`167 1018C 1H 1`), shifting everything right;
  try both left- and right-aligned mappings.
- Age units: if a column reaches our window in *years* its max must be ≥115,000. So:
  `Ma` in name → ×1e6; `ka`/`kyr` → ×1e3; else max ≥115,000 → years; ≥115 → ka; else Ma.
  (A 1543-ka record was silently mis-read as years before this rule.)
- Regex `(?<![a-z])sst` fails on `FebSST`. Match `sst` plainly.
- `-999`, `-9999`, `-99.9`, `999` all mean missing.

## 6. Progress queue

**Done**
- Search: 122 studies, 485 data files, 188 files downloaded.
- 20 corrections to Anna's tabs, applied to the **copy** only: 10 longitude sign flips
  (Pisias & Mix 1997 eastern-equatorial-Pacific radiolaria cores were recorded as +East)
  and 10 age columns converted ka → yr BP. Logged in `corrections_log.json`.
- First build of all five outputs (40 new cores). Anna reviewed and commented.
- `parser2.py` rewritten. Recovery: 19/66 previously-rejected files now parse;
  accepted 57/95, chronology 74/89.
- `manual_specs.py` written from Anna's section I specs.

**Remaining — this is the job**
1. **Re-run screening with `parser2.py` + `manual_specs.py`** and rebuild all five outputs.
   Old outputs were built with the buggy v1 parser; treat them as stale.
2. Apply every one of Anna's 66 comments in `need_to_confirm.docx` and 11 in
   `rejected_with_reason.docx`. Read them from the docx XML:
   `word/comments.xml` for text, `commentRangeStart/End` in `word/document.xml` for anchors.
3. Special-format files (all already in `mined_txt/`):
   - `ikehara2000-tsp-2mc.txt` — col2 age (ka), col8 UK37, col9 SST.
   - `b_ca_tripati_2009_(manually_edited).xlsx` — Anna made this readable by hand. Proxy MgCa,
     age col C (LR04 model), SST col I, proxy col L, rows 112 and 114, species
     *G. ruber* white. Note in col Y that *G. sacculifer* and the Medina-Elizalde & Lea (2005)
     age model are also available.
   - `herbert2016-med.txt` and `herbert2016-odp883_884.txt` — **stacks of several cores**. Split
     by the `sitename` column into one tab each; use `age_kaBP`, `Uk'37`, `SST_P`, `SST_M`;
     lon/lat from the respective publications.
   - `mined_txt/Ruddiman_Fossil_Plankton/` — 31 files. Filename = core ID; last two columns are
     `SSTwarm` and `SSTcold`; average to annual; lon/lat on line 1. **These go to Jerry's sheet**
     as depth-vs-SST (no age model).
4. Extend Jerry's sheet with `PS1778-5`, `E45-29` (depths 249–284 cm), `E49-17`, `E49-18` —
   cores where depth is sampled more densely than age.
5. Rebuild the two .docx reports (see §7 for the edits Anna asked for).

### Confirmed per-core decisions
**Exclude → `rejected_with_reason.docx`,** reason: *record spans an interval far longer than the
study window (e.g. 20 Myr), so the depth–age model is too poorly constrained over the LIG*:
`DSDP 588`, `ODP 1021`, `ODP 1088`, `ODP 1208`, `ODP 1010`.
Only these four herbert2016 cores are excluded — **other herbert2016 cores are kept** and flagged sparse.

**Also reject:** `E49-23` (1 point), `MD97-2141` (255 in-window rows, all SST blank),
`IODP U1488`, `DSDP 475`, `ODP 806` (no rows in window), `ODP 980` (`oppo2006.txt` header
unreadable), `odp1012.txt` (broken header — use `odp1012-tab.txt` instead, which parses fine),
`white2020-806-mgca-sst.txt` (1 sample), `tran2025-u1482-coretop.txt` (multiple lat/lon),
`rustic2020.txt` and `turney2021.txt` (interval bins, no 1-to-1 age–SST correspondence).
Remove `Lingtai` from the "not marine" section — it is the ODP 1146 deep-ocean dataset.

**Keep** (sparse but sound — Anna's reasoning: the *ages* are well sampled, only some samples
lack a proxy→SST conversion): `Q200`, `R657`, `U938`, `ODP 847`, `PS1778-5`, `RS147-GC14`,
`E45-29`, `E49-17`, `E49-18`, `E49-21`, `DSDP 593`.

**Specific fixes**
- `ODP 847` — use the **`sst-mg/ca.adj`** column (adjusted for Mg/Ca variation), not the raw one.
- `RS147-GC14` — source states "SSTs from Uk'37 have an error of ±1.5 °C"; use it for S and U.
- `LPAZ-21P` — same core as Hoffman's **`LAPAZ21`**; say so in col Y. Its stated 0.15 °C
  precision gives ±0.3 for 2sd.
- `PS1778-5` — proxy is **Radiolaria**; col Y should read: *"Mean sea surface temperature,
  summer, Dec–March, 10 m water depth, calculated from radiolaria, using Transfer function
  (Imbrie & Kipp, 1971, in Turekian, Yale Univ Press)."*
- `SO136-111` (crosta2004) — February SST → **summer-signal core** in the Southern Ocean; flag.
- `MD99-2331` — `SST.for-JAS` is July–August–September → **summer-signal core**; flag.
  `-i`/`-s` are probably inferior/superior bounds; unspecified, so unused.
- `ODP 820` — elevation NaN, but extract the data.
- `IODP U1485` (`bova2020-u1485-sst.txt`) — **not** an interval-average product; it has per-sample
  ages. Its reported error is on the *anomaly*, so derive the baseline (SST minus SST_anom) to
  get the SST error.
- `ODP 1014A` — Anna's spec describes **`odp1014a-tab.txt`**, not `yamamoto2007.txt`. Use the former.
- Delete the `RC13-110` and `NH22P` tabs from the chronology file — Anna confirmed the versions
  already in her sheet are correct.
- For chronology duplicates generally: keep the **richest** record. If the duplicate is a Hoffman
  core, keep it only if it extends past 130 ka toward 140 ka (meaning Hoffman truncated it). If it
  duplicates one of Anna's mined cores, keep whichever is richer; if the two cover
  non-overlapping ages, **merge** them and explain in col Y.

## 7. Edits Anna asked for in the reports

`need_to_confirm.docx`
- Section D: drop the "no uncertainty column in source" rows; keep only cores where an
  uncertainty *was* derived.
- Section E: remove cores that are now excluded; leave the rest as logged.
- Section G: mark each core `(Hoffman)` or `(newly mined)` in brackets after the ID.
- Keep flags for cores Anna decided to keep — she is reporting these to Jerry.
- Add a note that **AC manually created** `b_ca_tripati_2009_(manually_edited).xlsx`.
- Add the `GeoB10083` / `GeoB10163` / `GeoB10285` fix: the correct error column is `sst_err`
  (~0.3), **not** `d18Og.rub250-350_err` (~0.08). 0.08 °C is not physically achievable.

## 8. To-do list for later (not now)

1. **Notebook code fixes** in `SST_RSOI.ipynb`:
   - Summer-core flags are lost or misapplied at 5 of 81 sites, because the site's
     representative ID is `g["ID"].iloc[0]`. `MD95-2040` and `ODP 1089` lose the correction;
     `M23323-1`, `MD84-527`, `MD88-770` over-apply it to a blended annual+summer site.
     The August correction should be applied **per core, before** optimal estimation.
   - Cell 21's "global area-weighted mean" is a plain grid-cell mean — not area-weighted.
     Use `Σw·T / Σw` as cell 17 does.
   - `R_t` keeps only the diagonal of the rank-truncation covariance; document or test this.
   - σ = 0 gives a zero-error observation; σ = NaN silently drops a site. Add counters.
   - The convergence check (Λ₁ ≟ ⟨p pᵀ⟩ + P^OI) is never actually evaluated.
   - Cell 25 writes iteration EOF files shaped (151, …) when only 20 modes are filled.
2. **Validate against all six reconstructions** (iterations 0–5) to find which is best and where
   iterating stops improving.
3. **Coordinate discrepancies to resolve:** Anna's `V19-29` lat is −3.25, NCEI says −3.57
   (**NCEI is correct — change to −3.57**). Anna's `NH22P` lat is 22.52, NCEI says 23.52
   (Anna's is correct).
4. **`bova2020` regional files** (`bova2020-ProxyData-extratropics.txt` / `-tropics.txt`) —
   low priority. From row 184 the columns are `{coreID}_{variable}`. Split into separate cores,
   find lon/lat from NCEI/NOAA, and find the **SST-anomaly baseline** in the Bova et al. 2020
   publication so the anomalies can be converted to absolute SST. Then dedup and add.
5. **`ODP 980` / `oppo2006.txt`** — revisit the original publication for column definitions.
6. Cores with SST but no age model are still useful to **Jerry** — keep logging them.

## 9. Mental notes

- **M1 — the 1°×1° grid rule alone is not sufficient.** `RC13-110` and `NH22P` were missed
  because Anna's coordinate and NCEI's straddle a cell boundary: `floor(−96.08) = −97` but
  `floor(−96) = −96`. Always dedup by **normalised core ID as well**.
- **M2 — same cell ≠ same core.** Of 43 Hoffman matches, several are genuine neighbours:
  `NA87-25` ← `ODP 980`, `MD95-2036` ← `KNR191-CDH19`, `MD94-102` ← `RC11-120`,
  `V34-88` ← `ODP 722`. Check the ID before calling something a duplicate.
- **M3 — 64 of the downloaded cores already exist** (43 Hoffman + 21 of Anna's). Expected:
  Hoffman compiled in 2017, after most of these studies.
- **M4 — only ~18% of new datapoints have ±2sd.** Most NCEI files publish no uncertainty. This
  matters for the RSOI: with σ = NaN, `wsum` is 0 and the site is silently dropped at that
  timestep. A policy (e.g. proxy-typical σ) is needed before these enter the reconstruction.
- **M5 — openpyxl rewrites float noise** (`1601.0000000000002` → `1601`, ~1e-11). Harmless, but
  the working copy is a rewrite, not a surgical patch. Anna's original is untouched.
- **M6 — Anna's own spreadsheet had unit and sign errors.** Check new data the same way:
  compare the core's coordinates against the basin named in the study.
- **M7 — always ask** rather than guess on scientific judgement calls (calibration choice,
  seasonality, what counts as a duplicate). Anna prefers a flagged decision to a silent one.
  Use a multiple-choice question when it changes what you'd do.
