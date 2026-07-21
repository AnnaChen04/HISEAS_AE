#!/usr/bin/env python3
"""
Hand-mapped column specs supplied by Anna (need_to_confirm.docx section I)
for files parser2 could not resolve automatically.

Column indices are 0-based. 'age_unit' is the multiplier to years BP.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser2 as P

# Anna's explicit rejections from section I
MANUAL_REJECT = {
    'odp1012.txt': 'Header row is irrecoverably broken (mis-tabbed); superseded by odp1012-tab.txt',
    'oppo2006.txt': 'Column header not interpretable even manually; revisit the original publication '
                    '(Oppo et al. 2006) for column definitions - deferred, low priority',
    'white2020-806-mgca-sst.txt': 'Only one sample falls within 115-140 ka',
    'tran2025-u1482-coretop.txt': 'Core-top file carries multiple lat/lon records, no single site',
    'rustic2020.txt': 'Discrete time-interval bins; no one-to-one correspondence of timestamp with SST',
    'turney2021.txt': 'Discrete time-interval bins; no one-to-one correspondence of timestamp with SST',
    'yamamoto2007.txt': 'Prose/abstract file for ODP 1014A with no parsable data table; '
                        'the data are in odp1014a-tab.txt, which is used instead',
}

# file -> spec
SPECS = {
    # AC 2026-07-21: the section-I spec describes odp1014a-tab.txt, NOT yamamoto2007.txt
    # (the latter is the prose/abstract file and carries no parsable table).
    # Depth taken from 'corrdptcm' (corrected depth), read as cm per AC.
    'odp1014a-tab.txt': dict(core='ODP 1014A', proxy='UK37', depth=2, depth_unit='cm',
                             age=1, age_unit=1.0, pval=5, sst=4,
                             method='Columns: depth, calyrBP, corrdptcm, d15N, sst-uk37, uk37. '
                                    'UK37 alkenone unsaturation index (Prahl et al. 1988 '
                                    'calibration) used as proxy value; SST from the sst-uk37 '
                                    'column. Depth reported here is "corrdptcm" (corrected '
                                    'depth), taken as centimetres per AC. Note the file also '
                                    'carries an uncorrected "depth" column; its unit label '
                                    '"(cm)" conflicts with the section/interval notes, so the '
                                    'corrected column is used instead.'),
    'm972141r-tab.txt': dict(core='MD97-2141', proxy='MgCa', depth=0, depth_unit='cm',
                             age=1, age_unit=1.0, pval=5, sst=6, d18o=2,
                             method='Columns: depth(cm), calyrBP, d18Og.rub-w212-250, dep.nvoid, '
                                    'd18Owater, Mg/Ca-g.rub-w, sst-mg/ca. SST inferred from Mg/Ca on '
                                    'Globigerinoides ruber (white), 212-250 um.'),
    'yamamoto2005.txt': dict(core='MD01-2421', proxy='UK37', depth=0, depth_unit='cm',
                             age=1, age_unit=1e3, pval=2, sst=3,
                             method='Columns: Depth(cm), Age(ka), UK37, SST(degC).'),
    'e49-23-tab.txt': dict(core='E49-23', proxy='Foram', depth=0, depth_unit='cm',
                           age=1, age_unit=1.0, d18o=13,
                           sst_aug=26, err_aug=27, sst_feb=28, err_feb=29,
                           method='Columns: depth(cm, col A), yrBP (col B), d18O (col N), '
                                  'sst-aug (col AA), sst-aug_err (col AB), sst-feb (col AC), '
                                  'sst-feb_err (col AD).'),
    'liu2022-u1488-alkenone.txt': dict(core='IODP U1488', proxy='UK37',
                                       depth_avg=(4, 5), depth_unit='cm',
                                       age=7, age_unit=1e3, pval=8, sst=9,
                                       method='Columns E,F,H,I,J = cm_top, cm_bottom, age_kaBP, '
                                              "UK'37, SST. Depth is the midpoint of cm_top and cm_bottom."),
    'liu2022-u1488-tex86.txt': dict(core='IODP U1488', proxy='TEX86',
                                    depth_avg=(4, 5), depth_unit='cm',
                                    age=7, age_unit=1e3, pval=15, sst=16,
                                    method='Columns E,F,H,P,Q = cm_top, cm_bottom, age_kaBP, '
                                           'TEX86_average, SST. Depth is the midpoint of cm_top and cm_bottom.'),
    'brennan2022_dsdp475.txt': dict(core='DSDP 475', proxy='UK37', depth=0, depth_unit='cm',
                                    age=1, age_unit=1e6, pval=2, err=3, err_is_1sd=True, sst=4,
                                    method='Columns: depth_cm, age_ma, Uk37, Uk37_err, sst_median. '
                                           'Age converted from Ma to years BP. sst_median used as SST mean.'),
    'ford2019-odp806.txt': dict(core='ODP 806', proxy='MgCa', age=0, age_unit=1e6,
                                pval=9, sst=10, no_depth=True,
                                method='Age (col A) in Ma. Proxy value from "Mg/Ca_seawater_adj" (col J); '
                                       'SST from "SST_Mgsw_adj" (col K), i.e. Mg/Ca corrected for '
                                       'seawater Mg/Ca. Dataset contains no depth series.'),
}

# ODP 668B: Anna supplied the values directly (age ka, d18O, Mg/Ca, SST)
LITERAL = {
    'ODP 668B': dict(
        core='ODP 668B', proxy='MgCa', file='hoenisch2009.txt',
        method='Values supplied directly by AC from table 2 of hoenisch2009.txt '
               '(Time ka, d18O, Mg/Ca mmol/mol, SST degC). Automated parsing of this '
               'table failed; the four in-window points were transcribed manually.',
        rows=[(123.2, -1.96, 4.00, 27.8), (128.1, -1.41, 3.72, 27.0),
              (136.8, -0.21, 2.84, 24.0), (138.6, 0.01, 3.00, 24.6)]),
}


def _cell(row, i):
    return P.num(row[i]) if i is not None and len(row) > i else None


def extract_manual(path, meta):
    """Return (series_list, error) mirroring parser2.extract for hand-mapped files."""
    fn = os.path.basename(path)
    if fn in MANUAL_REJECT:
        return [], MANUAL_REJECT[fn]
    spec = SPECS.get(fn)
    if not spec:
        return None, None                      # not hand-mapped; caller falls back
    L = P.read_lines(path)
    tables = P.find_all_tables(L)
    if not tables:
        return [], 'Could not locate a numeric data table'
    rows = max((t[1] for t in tables), key=len)
    hdr = P.header_text(L)

    seasonal = 'sst_aug' in spec
    recs, notes, method = [], [], [spec['method']]
    for r in rows:
        a = _cell(r, spec.get('age'))
        if a is None:
            continue
        a *= spec.get('age_unit', 1.0)
        if not (P.AGE_MIN <= a <= P.AGE_MAX):
            continue
        if seasonal:
            sa, sf = _cell(r, spec['sst_aug']), _cell(r, spec['sst_feb'])
            if sa is None or sf is None:
                continue
            sst = (sa + sf) / 2.0
            ea, ef = _cell(r, spec.get('err_aug')), _cell(r, spec.get('err_feb'))
            err = (ea + ef) / 2.0 if (ea is not None and ef is not None) else None
            if err is not None:
                err *= 2.0                      # 1sd -> 2sd
            extra = dict(aug=sa, feb=sf)
        else:
            sst = _cell(r, spec.get('sst'))
            if sst is None:
                continue
            err = _cell(r, spec.get('err'))
            if err is not None and spec.get('err_is_1sd'):
                err *= 2.0
            extra = {}
        if not (P.SST_LO <= sst <= P.SST_HI):
            continue
        if spec.get('no_depth'):
            d = None
        elif 'depth_avg' in spec:
            d1, d2 = _cell(r, spec['depth_avg'][0]), _cell(r, spec['depth_avg'][1])
            d = (d1 + d2) / 2.0 if (d1 is not None and d2 is not None) else None
        else:
            d = _cell(r, spec.get('depth'))
            if d is not None and spec.get('depth_unit') == 'm':
                d *= 100.0
        recs.append(dict(depth=d, pval=_cell(r, spec.get('pval')), sst=sst, age=a,
                         lo=None if err is None else sst - err,
                         hi=None if err is None else sst + err,
                         bd18o=_cell(r, spec.get('d18o')), **extra))
    if not recs:
        return [], 'Hand-mapped columns parsed, but no valid SST values fall inside 115-140 ka'
    recs.sort(key=lambda x: x['age'])
    if seasonal:
        notes.append('Seasonal Aug/Feb columns averaged to annual mean (errors averaged then doubled)')
        method.append('Annual SST computed as the mean of the August and February columns; '
                      'the two reported errors were averaged and doubled to give +/-2sd.')
    if len(recs) < 5:
        notes.append(f'SPARSE: only {len(recs)} entries within 115-140 ka')
    if all(r['lo'] is None for r in recs):
        notes.append('No SST uncertainty column in source; -2sd/+2sd blank')
    notes.append('Columns hand-mapped from spec supplied by AC (need_to_confirm section I)')
    return [dict(col=f'manual:{fn}', proxy=spec['proxy'], recs=recs, notes=notes,
                 method=' '.join(method), table_line=0, seasonal=seasonal)], None


def literal_series(key):
    L = LITERAL[key]
    recs = [dict(depth=None, pval=mg, sst=sst, age=t * 1000.0, lo=None, hi=None, bd18o=d18)
            for t, d18, mg, sst in L['rows']]
    recs.sort(key=lambda x: x['age'])
    return dict(col='manual-literal', proxy=L['proxy'], recs=recs,
                notes=[f'SPARSE: only {len(recs)} entries within 115-140 ka',
                       'Values transcribed manually by AC from hoenisch2009.txt table 2'],
                method=L['method'], table_line=0, seasonal=False)
