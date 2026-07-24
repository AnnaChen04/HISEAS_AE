#!/usr/bin/env python3
"""
Bespoke handlers for the files listed in handoff.md section 6.3.

These files defeat both parser2 and the manual_specs column-index scheme,
either because the table is a stack of several cores, because the header
carries no usable column names, or because the source is an .xlsx.

Every decision recorded here was confirmed by AC on 2026-07-21; the
confirming rationale is in the docstring of each handler.
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser2 as P

AGE_MIN, AGE_MAX = P.AGE_MIN, P.AGE_MAX
SST_LO, SST_HI = P.SST_LO, P.SST_HI


# --------------------------------------------------------------- rejections
# Confirmed by AC 2026-07-21. These files are named in section 6.3 as things to
# process, but inspection showed there is nothing in them to process.
SPECIAL_REJECT = {
    'herbert2016-odp883_884.txt':
        'Stack of ODP 883 and ODP 884: no samples fall within 115-140 ka. Site 883 spans '
        '2705-11345 ka and site 884 spans 6546-7863 ka, i.e. both records are entirely '
        'Miocene-Pliocene. Both also span far longer than the study window, so the '
        'depth-age model would be too poorly constrained over the LIG even if samples existed.',
    'b_ca_tripati_2009_(manually_edited).xlsx':
        'ODP 806: two conflicting age models are reported for the same samples (Medina-Elizalde '
        '& Lea 2005 and LR04), and only two samples fall within 115-140 ka under either model. '
        'The record is too sparsely sampled across the study window to constrain LIG SST. '
        'NOTE: this .xlsx was created manually by AC to make the wrapped source file '
        'b_ca_tripati_2009.txt machine-readable; the transcription is sound, the rejection '
        'is on scientific grounds, not parsing grounds.',
}


# --------------------------------------------------------------- ikehara2000
def _ikehara(path, meta):
    """ikehara2000-tsp-2mc.txt - Termination II biomarker table, Southern Ocean.

    Column layout (0-based, tab-split; note the file carries blank spacer
    fields, so these indices are NOT the same as the visual column numbers):
        0 depth cm | 1 age ka | 2 MIS regime | 3 CPI | 4 ACL | 5 C25-C35
        6 (spacer) | 7 C37-C39 | 8 UK'37 | 9 SST degC

    Core ID: AC confirmed TSP-2PC (2026-07-21). NCEI files the study under
    TSP-2MC, but the table inside is titled "Sediment Core TSP-2PC" and a
    165-199 cm record spanning 112-144 ka is piston-core, not multicore,
    sampling. The conflict is recorded in the Method column.
    """
    L = P.read_lines(path)
    rows = []
    for line in L:
        t = P.split_row(line)
        if len(t) > 9 and P.is_num(t[0]) and P.is_num(t[1]):
            rows.append(t)
    recs = []
    for r in rows:
        a, s = P.num(r[1]), P.num(r[9])
        if a is None or s is None:
            continue
        a *= 1e3
        if not (AGE_MIN <= a <= AGE_MAX) or not (SST_LO <= s <= SST_HI):
            continue
        recs.append(dict(depth=P.num(r[0]), pval=P.num(r[8]), sst=s, age=a,
                         lo=None, hi=None, bd18o=None))
    if not recs:
        return [], 'No SST values inside 115-140 ka'
    recs.sort(key=lambda x: x['age'])
    notes = ['Columns hand-mapped (section 6.3): depth cm, age ka, UK37 index, SST degC',
             'CORE ID CONFLICT: NCEI files this study under site "TSP-2MC", but the data '
             'table is titled "TSP-2PC"; AC confirmed TSP-2PC',
             'No SST uncertainty column in source; -2sd/+2sd blank']
    if len(recs) < 5:
        notes.append(f'SPARSE: only {len(recs)} entries within 115-140 ka')
    method = ('Alkenone SST from the UK\'37 unsaturation index (source column "UK37\'", '
              'reported alongside "SST"). Depth in cm, age in ka converted to years BP. '
              'Core ID recorded as TSP-2PC per the data-table title "Summary of Mass '
              'Accumulation Rates of Biomarkers for the Penultimate Deglaciation from a '
              'Sediment Core TSP-2PC"; NCEI files the same study under site TSP-2MC. '
              'AC confirmed TSP-2PC on 2026-07-21 on the grounds that a 165-199 cm record '
              'spanning 112-144 ka is piston-core sampling, not a multicore. '
              'Source reports no SST uncertainty.')
    return [dict(col='manual:ikehara2000-tsp-2mc', proxy='UK37', recs=recs, notes=notes,
                 method=method, table_line=0, seasonal=False,
                 core_override='TSP-2PC')], None


# --------------------------------------------------------- herbert2016 stack
# ODP 967, Eratosthenes Seamount, eastern Mediterranean. The NCEI record for
# this file gives a single site "Central/Southern Italy" at 43.62N 13.59E,
# which is a LAND coordinate for the Italian outcrop sections also in the
# stack - it must not be applied to ODP 967. Coordinates below are the site's
# own, via PANGAEA (Grant et al. 2022, doi:10.1594/PANGAEA.939929).
HERBERT_SITES = {
    '967': dict(core='ODP 967', lon=32.725433, lat=34.0696, elev=-2553.2),
}


def _herbert_stack(path, meta):
    """herbert2016-med.txt - 16 sitenames in one table.

    Only ODP 967 has data inside 115-140 ka (21 points). The other 15 names are
    Mediterranean land sections (Vrica, Monte del Casino/MDC, Singa/SN, Punta
    Piccola/PP, EM11) and ODP 964, all spanning 1.4-13 Ma with zero in-window
    samples - so the stack collapses to a single core.

    SST_P (Prahl et al. 1988) is used, with SST_M (Muller et al. 1998) recorded
    as the alternative. AC confirmed SST_P on 2026-07-21 for consistency: it is
    what parser2 already selects for every other herbert2016 core (ODP 982, 846,
    722), so the collection stays internally comparable.
    """
    L = P.read_lines(path)
    tables = P.find_all_tables(L)
    if not tables:
        return [], 'Could not locate a numeric data table'
    cols, rows, li = max(tables, key=lambda t: len(t[1]))
    try:
        i_site = cols.index('sitename')
        i_age = cols.index('age_kaBP')
        i_uk = cols.index("Uk'37")
        i_p = cols.index('SST_P')
        i_m = cols.index('SST_M')
    except ValueError as e:
        return [], f'Expected stack columns missing ({e})'
    i_mcd = cols.index('depth_mcd') if 'depth_mcd' in cols else None
    i_mbsf = cols.index('depth_mbsf') if 'depth_mbsf' in cols else None

    out = []
    for key, site in HERBERT_SITES.items():
        recs, alt_m = [], []
        for r in rows:
            if len(r) <= max(i_p, i_age, i_site) or r[i_site].strip() != key:
                continue
            a, s = P.num(r[i_age]), P.num(r[i_p])
            if a is None or s is None:
                continue
            a *= 1e3
            if not (AGE_MIN <= a <= AGE_MAX) or not (SST_LO <= s <= SST_HI):
                continue
            d = None
            for i in (i_mcd, i_mbsf):
                if i is not None and len(r) > i and P.num(r[i]) is not None:
                    d = P.num(r[i]) * 100.0      # mbsf/mcd are metres -> cm
                    break
            recs.append(dict(depth=d, pval=P.num(r[i_uk]), sst=s, age=a,
                             lo=None, hi=None, bd18o=None))
            alt_m.append(P.num(r[i_m]))
        if not recs:
            continue
        recs.sort(key=lambda x: x['age'])
        mvals = [v for v in alt_m if v is not None]
        notes = [f'Extracted from the multi-site stack {os.path.basename(path)} by '
                 f'sitename == "{key}"; the file\'s other 15 sitenames are Mediterranean '
                 f'land sections and ODP 964, none with samples in 115-140 ka',
                 'Coordinates taken from the ODP 967 site record, NOT from the file\'s NCEI '
                 'metadata, which gives a single land coordinate (43.62N 13.59E, '
                 '"Central/Southern Italy") for the whole stack',
                 'Alternative calibration SST_M (Muller et al. 1998) available; SST_P used',
                 'No SST uncertainty column in source; -2sd/+2sd blank']
        if len(recs) < 5:
            notes.append(f'SPARSE: only {len(recs)} entries within 115-140 ka')
        method = (
            f"Alkenone SST from the UK'37 unsaturation index. Value used is the source "
            f"column \"SST_P\" (calibration of Prahl et al. 1988). The file also reports "
            f"\"SST_M\" (calibration of Muller et al. 1998)"
            + (f", which over these {len(mvals)} in-window samples ranges "
               f"{min(mvals):.2f}-{max(mvals):.2f} degC" if mvals else "")
            + ". SST_P was chosen by AC (2026-07-21) for consistency with the other "
              "herbert2016 cores in this compilation (ODP 982, ODP 846, ODP 722), which "
              "also use SST_P. Depth converted from metres (composite depth mcd where "
              "available, else mbsf) to centimetres. Age reported in ka BP, converted to "
              "years BP. Chronology follows Tzanova et al. 2015, Herbert et al. 2014 and "
              "Emeis et al. 2000, 2003. Source reports no SST uncertainty; the "
              "age_uncert_ka column is an AGE uncertainty and is deliberately not used "
              "as an SST error.")
        out.append(dict(col='manual:herbert2016-med[967]', proxy='UK37', recs=recs,
                        notes=notes, method=method, table_line=li, seasonal=False,
                        core_override=site['core'], lon=site['lon'], lat=site['lat'],
                        elev=site['elev']))
    if not out:
        return [], 'No site in this stack has SST inside 115-140 ka'
    return out, None


# ------------------------------------------------------------------ dispatch
SPECIAL = {
    'ikehara2000-tsp-2mc.txt': _ikehara,
    'herbert2016-med.txt': _herbert_stack,
    'herbert2016-odp883_884.txt': None,                       # reject
    'b_ca_tripati_2009_(manually_edited).xlsx': None,         # reject
}


def extract_special(path, meta):
    fn = os.path.basename(path)
    if fn in SPECIAL_REJECT:
        return [], SPECIAL_REJECT[fn]
    fn_handler = SPECIAL.get(fn)
    if fn_handler is None:
        return [], f'No special handler registered for {fn}'
    return fn_handler(path, meta)
