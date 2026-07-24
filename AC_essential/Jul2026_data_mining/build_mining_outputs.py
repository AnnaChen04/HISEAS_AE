#!/usr/bin/env python3
"""
Build SST data-mining outputs from NCEI paleo txt files.

Inputs (all in AC_essential/):
  screened.json                       - screening decisions from ncei_search_full.json
  mined_txt/                          - downloaded NCEI data files
  SST_Data_Mining_corrected_2026-07-20.xlsx  - user's file + ka/longitude corrections

Outputs:
  SST_Data_Mining_corrected_2026-07-20.xlsx  (accepted cores appended)
  same_core_different_chronology.xlsx
  SST_Depth_Only_For_Jerry.xlsx
  flags.json / rejects.json  (consumed by the docx writer)
"""
import json, re, os, glob, math, unicodedata
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(BASE, 'mined_txt')
AGE_MIN, AGE_MAX = 115000.0, 140000.0

HEADERS = ['ID','Longitude','Latitude','Proxy','Tie Point Depth (cm)','Tie Point Age (yr)',
 'Tie Point Uncertainty (yr)','Hadley Bias Correction','Hadley uncertainty','TF Uncertainty',
 'SS Uncertainty','Benthic d18O Depth (cm)','Benthic d18O (per mil)','Depth (cm)','Proxy Value',
 'SST (ºC)','Biascorr','Age (yr)','SST -2sd','SST mean','SST +2sd','Publication','Data Link',
 'Chronology Info','Method','Elevation (m)']

# ---------------------------------------------------------------- file parsing
def read_lines(p):
    return open(p, encoding='utf-8', errors='replace').read().split('\n')

def split_row(s):
    return [t.strip() for t in re.split(r'\t|\s{2,}', s.strip())]

def is_num(t):
    return bool(re.fullmatch(r'-?\d+\.?\d*([eE][-+]?\d+)?', t.strip()))

def find_table(L):
    """Return (header_tokens, list_of_data_rows) for the numeric table."""
    for i, l in enumerate(L):
        s = l.strip()
        if not s or s.startswith('#'):
            continue
        toks = split_row(s)
        if len(toks) < 2:
            continue
        nxt = [x for x in L[i+1:i+8] if x.strip() and not x.strip().startswith('#')]
        if not nxt:
            continue
        good = 0
        for n in nxt:
            f = split_row(n)
            if sum(1 for t in f if is_num(t)) >= max(1, len(f)//2):
                good += 1
        if good >= max(1, len(nxt)//2):
            rows = []
            for n in L[i+1:]:
                if not n.strip() or n.strip().startswith('#'):
                    continue
                rows.append(split_row(n))
            return toks, rows
    return None, None

def header_text(L):
    return '\n'.join(x for x in L if x.strip().startswith('#'))

MISSING = {'-999', '-999.0', '-999.9', '-9999', 'nan', 'NaN', '', 'NA', '-99.9', '999'}
def num(t):
    t = (t or '').strip()
    if t in MISSING or not is_num(t):
        return None
    v = float(t)
    if v in (-999, -9999, -99.9, 999):
        return None
    return v

# ---------------------------------------------------------------- column roles
RE_AGE   = re.compile(r'^(cal)?(yr|year)s?[\s_-]?bp$|age|^time$|calyrbp|yrbp|kyr|kabp', re.I)
RE_KA    = re.compile(r'ka|kyr|kiloyear', re.I)
RE_MA    = re.compile(r'(?<![a-z])ma(?![a-z])|calma|mabp|million', re.I)
RE_DEPTH = re.compile(r'depth|^cm[_ ]?top$|mbsf|mcd', re.I)
# NB: underscore is a word char, so \bsst\b fails on SST_P / sst_uk_med. Use a
# letter-only left boundary instead so those are caught.
RE_SST   = re.compile(r'(?<![a-z])sst|sea[\s_-]?surface[\s_-]?temp|^temp(erature)?$', re.I)
RE_ANOM  = re.compile(r'anom|delta|resid', re.I)
RE_SD    = re.compile(r'(?<![a-z])sd(?![a-z])|std|stdev|error|(?<![a-z])err|uncert|\+/-|±|95%|conf|sig', re.I)
# CI-bound suffixes that pair with a median column (sst_uk_5 / sst_uk_med / sst_uk_95)
RE_LO    = re.compile(r'(_|-|\.)(5|05|low|lo|min|2\.5|025)$', re.I)
RE_HI    = re.compile(r'(_|-|\.)(95|high|hi|max|97\.5|975)$', re.I)
RE_MED   = re.compile(r'(_|-|\.)(med|median|50|mean|ave|avg)$', re.I)
RE_D18O  = re.compile(r'd18o|delta18o|δ18o', re.I)
RE_BENTH = re.compile(r'benth|cibicid|uvigerina|c\.?\s?wueller', re.I)
PROXY_PATS = [('UK37', r"uk'?\.?37|alkenone"), ('MgCa', r"mg/?ca"), ('TEX86', r"tex\.?86"),
              ('Diatom', r'diatom'), ('Radiolaria', r'radiolaria|radio\b'),
              ('Cocc', r'coccolith|nannofossil'), ('Foram', r'foramin|transfer function|modern analog|faunal|plank')]
SEASON = re.compile(r'aug|feb|summer|winter|jja|djf|jan|jul', re.I)

def classify(cols):
    role = {}
    for i, c in enumerate(cols):
        cl = c.lower().strip()
        if RE_SST.search(cl):
            role.setdefault('sst', []).append(i)
        if RE_AGE.search(cl) and not RE_DEPTH.search(cl):
            role.setdefault('age', []).append(i)
        if RE_DEPTH.search(cl):
            role.setdefault('depth', []).append(i)
        if RE_D18O.search(cl):
            role.setdefault('d18o', []).append(i)
    return role

def pick_proxy(text):
    for name, pat in PROXY_PATS:
        if re.search(pat, text, re.I):
            return name
    return ''

# ---------------------------------------------------------------- extraction
def extract(path, meta):
    """Return list of series dicts extracted from one txt file."""
    L = read_lines(path)
    if any('<html' in x.lower() for x in L[:40]):
        return [], 'HTML directory index, not a data file'
    cols, rows = find_table(L)
    if not cols:
        return [], 'Could not locate a numeric data table'
    hdr = header_text(L)
    role = classify(cols)
    if 'sst' not in role:
        return [], 'No SST column found in data table'
    if 'age' not in role:
        return [], 'No age column found in data table'

    agei = role['age'][0]
    depthi = role['depth'][0] if 'depth' in role else None

    # age unit: Ma > ka > yr, by column name first then by magnitude
    agevals = [num(r[agei]) for r in rows if len(r) > agei]
    agevals = [v for v in agevals if v is not None]
    if not agevals:
        return [], 'Age column present but contains no numeric values'
    an = cols[agei]
    if RE_MA.search(an):
        scale = 1e6
    elif RE_KA.search(an):
        scale = 1e3
    elif max(agevals) < 1000:
        scale = 1e3            # bare small numbers are almost always ka
    else:
        scale = 1.0

    # ---- choose real SST series, pairing off CI-bound columns -------------
    sst_cols = [i for i in role['sst'] if not RE_ANOM.search(cols[i]) and not RE_SD.search(cols[i])]
    lo_cols = [i for i in sst_cols if RE_LO.search(cols[i])]
    hi_cols = [i for i in sst_cols if RE_HI.search(cols[i])]
    series_cols = [i for i in sst_cols if i not in lo_cols and i not in hi_cols]

    def base(n):
        return RE_LO.sub('', RE_HI.sub('', RE_MED.sub('', n))).strip('_-. ')

    # Collapse alternative calibrations of the SAME proxy to one series, but keep
    # genuinely different proxies (e.g. UK37 vs TEX86) as separate series.
    def proxy_of(i):
        return (pick_proxy(cols[i]) or pick_proxy(' '.join(cols))
                or pick_proxy(meta.get('study_name', '') + ' ' + meta.get('tnotes', ''))
                or meta.get('proxy', '') or '?')
    bygroup = defaultdict(list)
    for i in series_cols:
        bygroup[proxy_of(i)].append(i)

    out = []
    for gproxy, idxs in bygroup.items():
        si = idxs[0]
        name = cols[si]
        notes = []
        if SEASON.search(name):
            notes.append(f'Seasonal SST column "{name}" (not an annual mean)')
        if len(idxs) > 1:
            notes.append('Source has %d alternative %s calibrations (%s); used "%s"'
                         % (len(idxs), gproxy, ', '.join(cols[j] for j in idxs), name))
        # (a) paired CI bounds sharing this column's base name
        lo_i = next((j for j in lo_cols if base(cols[j]) == base(name)), None)
        hi_i = next((j for j in hi_cols if base(cols[j]) == base(name)), None)
        sd_i, sd_kind = None, None
        if lo_i is not None and hi_i is not None:
            notes.append(f'Uncertainty from paired bounds "{cols[lo_i]}"/"{cols[hi_i]}"')
        else:
            # (b) fall back to a nearby explicit uncertainty column
            for j, c in enumerate(cols):
                if j == si or not RE_SD.search(c):
                    continue
                if RE_SST.search(c) or abs(j - si) <= 2:
                    sd_i = j
                    sd_kind = '2sd' if re.search(r'2\s?s|95%|two', c, re.I) else '1sd'
                    break
        proxy = gproxy if gproxy != '?' else ''
        # proxy value column
        pvi = None
        if proxy:
            pat = dict(PROXY_PATS)[proxy]
            for j, c in enumerate(cols):
                if j == si or RE_SST.search(c):
                    continue
                if re.search(pat, c, re.I):
                    pvi = j
                    break
        # benthic d18o
        bdi = None
        for j in role.get('d18o', []):
            if RE_BENTH.search(cols[j]) or RE_BENTH.search(hdr):
                bdi = j
                break

        recs = []
        for r in rows:
            if len(r) <= max(si, agei):
                continue
            a = num(r[agei])
            s = num(r[si])
            if a is None or s is None:
                continue
            a *= scale
            if not (AGE_MIN <= a <= AGE_MAX):
                continue
            lo = hi = None
            if lo_i is not None and hi_i is not None and len(r) > max(lo_i, hi_i):
                lo, hi = num(r[lo_i]), num(r[hi_i])
            sd = num(r[sd_i]) if sd_i is not None and len(r) > sd_i else None
            if sd is not None and sd_kind == '1sd':
                sd *= 2
            if lo is None and sd is not None:
                lo, hi = s - sd, s + sd
            recs.append(dict(
                depth=num(r[depthi]) if depthi is not None and len(r) > depthi else None,
                pval=num(r[pvi]) if pvi is not None and len(r) > pvi else None,
                sst=s, age=a, lo=lo, hi=hi,
                bd18o=num(r[bdi]) if bdi is not None and len(r) > bdi else None))
        if not recs:
            continue
        recs.sort(key=lambda x: x['age'])
        if sd_kind == '1sd':
            notes.append('Reported 1sd DOUBLED to 2sd')
        if lo_i is None and sd_i is None:
            notes.append('No uncertainty column in source; SST -2sd/+2sd left blank')
        if len(recs) < 5:
            notes.append(f'SPARSE: only {len(recs)} entries within 115-140 ka')
        out.append(dict(col=name, proxy=proxy, recs=recs, notes=notes))
    if not out:
        return [], 'SST column(s) found but no values inside 115-140 ka'
    return out, None

# ---------------------------------------------------------------- chronology / method text
def grab(hdr, keys, maxlen=900):
    lines = hdr.split('\n')
    hits = []
    for i, l in enumerate(lines):
        if any(k in l.lower() for k in keys):
            blk = [l]
            for n in lines[i+1:i+6]:
                t = n.lstrip('#').strip()
                if not t or re.match(r'^[A-Z][a-z]+.*:', t):
                    break
                blk.append(n)
            hits.append(' '.join(x.lstrip('#').strip() for x in blk))
    return ' | '.join(hits)[:maxlen]

def elevation(meta, hdr):
    try:
        e = float(meta.get('elev'))
        if e < 0:
            return e
    except (TypeError, ValueError):
        pass
    m = re.search(r'water\s*depth\s*\(?m?\)?\s*:?\s*(-?\d+\.?\d*)', hdr, re.I)
    if m:
        return -abs(float(m.group(1)))
    return None
