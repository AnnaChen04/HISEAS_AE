#!/usr/bin/env python3
"""
Shared sheet-writing helpers.

Extracted from write_outputs.py so that patch_tabs.py can regenerate a single tab
using exactly the same column layout and Method-column text as a full rebuild.
Importing write_outputs.py directly is not an option - it runs the whole pipeline
as a side effect of import.
"""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_mining_outputs as B
import decisions as D

# Col AA (27), added by AC 2026-07-21.
HEADERS_EXT = list(B.HEADERS) + ['summer_or_averaged_signal']
COL_SEASON = len(HEADERS_EXT)          # 27 = AA
COL_DUPE = COL_SEASON + 1              # 28 = AB

SEASON_NOTE = re.compile(r'SUMMER/SEASONAL SIGNAL CORE|SEASONAL SIGNAL', re.I)


def season_label(ser):
    if ser.get('_season'):                 # AC override from decisions.py
        return ser['_season']
    if ser.get('seasonal'):
        return 'aug-feb averaged'
    if any(SEASON_NOTE.search(n) for n in ser.get('notes', [])):
        return 'summer signal'
    return None


def apply_fixes(meta, ser):
    """Apply AC's confirmed per-core amendments (handoff section 6)."""
    fx = D.fix_for(meta['site'])
    add = []
    if fx.get('proxy'):
        ser['proxy'] = fx['proxy']
    if fx.get('season'):
        ser['_season'] = fx['season']
    if fx.get('err_abs') is not None:
        e = fx['err_abs']
        for rec in ser['recs']:
            rec['lo'], rec['hi'] = rec['sst'] - e, rec['sst'] + e
        ser['notes'] = [n for n in ser['notes']
                        if 'No SST uncertainty column in source' not in n]
    if fx.get('method_add'):
        add.append(fx['method_add'])
    if D.norm(meta['site']) in D.GEOB_ERR_FIX:
        add.append(D.GEOB_ERR_NOTE)
        ser.setdefault('notes', []).append('SIDE NOTE (AC 2026-07-21): ' + D.GEOB_ERR_FLAG)
    if fx.get('anomaly_baseline') and ser.get('_baseline') is not None:
        add.append(f'Absolute-SST baseline derived as (SST - SST_anom) = '
                   f'{ser["_baseline"]:.2f} degC; the reported uncertainty is on the anomaly '
                   f'and, the baseline being constant, transfers directly to absolute SST.')
    if add:
        ser['method'] = (ser.get('method', '') + ' ' + ' '.join(add)).strip()
    return ser


def write_series(ws, meta, ser, hdr):
    for c, h in enumerate(HEADERS_EXT, 1):
        ws.cell(row=1, column=c, value=h)
    core = re.sub(r'_(UK37|MgCa|TEX86|Foram|Diatom|Radiolaria|Cocc)$', '',
                  str(meta['site']), flags=re.I)
    ws.cell(row=2, column=1, value=core)
    ws.cell(row=2, column=2, value=meta['lon'])
    ws.cell(row=2, column=3, value=meta['lat'])
    ws.cell(row=2, column=4, value=ser['proxy'])
    ws.cell(row=2, column=22, value=(meta.get('cite') or '')[:1500])
    ws.cell(row=2, column=23, value=meta['url'])
    ws.cell(row=2, column=24, value=B.grab(hdr, ['chronolog', 'age model', 'age control',
                                                 'stratigraph', 'tie point', 'tuned']))
    src_method = B.grab(hdr, ['method', 'calibration', 'calculated', 'analy',
                              'proxy value', 'variables']) or ''
    method = ' '.join(x for x in (ser.get('method', ''), src_method) if x).strip()
    ws.cell(row=2, column=25, value=method[:2000] or None)
    ws.cell(row=2, column=26, value=B.elevation(meta, hdr))
    ws.cell(row=2, column=COL_SEASON, value=season_label(ser))
    for k, rec in enumerate(ser['recs']):
        r = 2 + k
        ws.cell(row=r, column=12, value=None)
        ws.cell(row=r, column=13, value=rec.get('bd18o'))
        ws.cell(row=r, column=14, value=rec['depth'])
        ws.cell(row=r, column=15, value=rec['pval'])
        ws.cell(row=r, column=16, value=rec['sst'])   # P  (== T, no interpolation)
        ws.cell(row=r, column=18, value=rec['age'])   # R
        ws.cell(row=r, column=19, value=rec['lo'])    # S
        ws.cell(row=r, column=20, value=rec['sst'])   # T
        ws.cell(row=r, column=21, value=rec['hi'])    # U
