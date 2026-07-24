#!/usr/bin/env python3
"""
Parser v2 for NCEI paleo txt files.

Fixes over v1:
  * reads EVERY numeric table in a file, not just the first
  * error column must demonstrably belong to SST (never age_uncert / proxy err)
  * depth harmonised to cm
  * seasonal Aug/Feb SST averaged to annual, errors averaged then doubled
  * Method column records calibration used + alternatives, species, seasonality,
    and free-text uncertainty statements
"""
import re, os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(BASE, 'mined_txt')
AGE_MIN, AGE_MAX = 115000.0, 140000.0
SST_LO, SST_HI = -5.0, 40.0          # physical sanity bounds (Anna, 2026-07-20)

MISSING = {'-999', '-999.0', '-999.9', '-9999', '-99.9', '999', 'nan', 'NaN', '', 'NA', '.'}

def read_lines(p):
    return open(p, encoding='utf-8', errors='replace').read().split('\n')

def split_row(s):
    """Delimiter priority: tabs > 2+ spaces > commas. Splitting on commas
    unconditionally breaks headers like 'Depth,cm' and misaligns every column."""
    s = s.rstrip()
    if '\t' in s:
        return [t.strip() for t in s.split('\t')]
    if re.search(r'\s{2,}', s.strip()):
        return [t.strip() for t in re.split(r'\s{2,}', s.strip())]
    if s.count(',') >= 2:
        return [t.strip() for t in s.split(',')]
    return [t.strip() for t in s.split()]

LEGEND = re.compile(r'^\s*#?\s*Column\s+(\d+)\s*[:.]\s*(.+?)\s*$', re.I)

def legends(L):
    """Map table-start line -> {col_index: name} from 'Column N: ...' prose blocks."""
    blocks, cur, start = [], {}, None
    for i, l in enumerate(L):
        m = LEGEND.match(l)
        if m:
            if not cur:
                start = i
            cur[int(m.group(1)) - 1] = m.group(2).strip()
        elif cur:
            blocks.append((start, i, cur))
            cur, start = {}, None
    if cur:
        blocks.append((start, len(L), cur))
    return blocks

def legend_for(blocks, table_line):
    """Nearest legend block that ends before the table starts."""
    best = None
    for s, e, d in blocks:
        if e <= table_line and (best is None or e > best[0]):
            best = (e, d)
    return best[1] if best else None

def is_num(t):
    return bool(re.fullmatch(r'-?\d+\.?\d*([eE][-+]?\d+)?', (t or '').strip()))

def num(t):
    t = (t or '').strip().replace(',', '')
    if t in MISSING or not is_num(t):
        return None
    v = float(t)
    return None if v in (-999, -9999, -99.9, 999, -999.9) else v

# ------------------------------------------------------------------ tables
def find_all_tables(L):
    """Yield (header_tokens, rows, header_line_index) for EVERY numeric block."""
    out, i, n = [], 0, len(L)
    while i < n:
        s = L[i].strip()
        if not s or s.startswith('#'):
            i += 1
            continue
        toks = split_row(s)
        if len(toks) < 2 or sum(1 for t in toks if is_num(t)) > len(toks) * 0.6:
            i += 1
            continue
        # NOAA files often stack a multi-line header: names, then units, then
        # the calibration reference, e.g. mohtadi2010b.txt
        #     Depth   Cal.Age  Mg/CaG.r   SST G.r   UK'37   AlkSST
        #     [cm]    [kyrBP]  [mmol/mol] [degC]    [ppm]   [degC]
        #                                 Anand2003         Conte2006
        # Without this, the scan walks past the real names row and adopts the
        # calibration row as the header, losing every column name in the file.
        sub, k = [], i + 1
        while k < n and len(sub) < 2:
            s2 = L[k].strip()
            if not s2 or s2.startswith('#'):
                break
            f2 = split_row(s2)
            if len(f2) < 2 or sum(1 for x in f2 if is_num(x)) >= max(1, len(f2) * 0.4):
                break
            sub.append(f2)
            k += 1
        if sub:
            f3 = split_row(L[k].strip()) if k < n and L[k].strip() else []
            if f3 and sum(1 for x in f3 if is_num(x)) >= max(1, len(f3) * 0.4):
                # a numeric block really does follow - fold the sub-rows into the names
                merged = list(toks)
                for extra in sub:
                    # only fold in a sub-row that lines up one-to-one with the
                    # names row. A partial row (e.g. a calibration row naming
                    # only 3 of 11 columns) is left-packed by the splitter and
                    # would attach its labels to the wrong columns.
                    if len(extra) != len(merged):
                        continue
                    for idx, tok in enumerate(extra):
                        if tok and tok not in merged[idx]:
                            merged[idx] = f'{merged[idx]} {tok}'.strip()
                toks = merged
                i = k - 1                      # resume scanning at the numeric block

        j, rows = i + 1, []
        while j < n:
            t = L[j].strip()
            if not t or t.startswith('#'):
                j += 1
                if rows and (j - i) > 3 and all((not L[k].strip() or L[k].strip().startswith('#'))
                                                for k in range(j, min(j + 3, n))):
                    break
                continue
            f = split_row(t)
            if sum(1 for x in f if is_num(x)) >= max(1, len(f) * 0.4):
                rows.append(f)
                j += 1
            else:
                break
        if len(rows) >= 2:
            out.append((toks, rows, i))
            i = j
        else:
            i += 1
    return out

def header_text(L):
    return '\n'.join(x for x in L if x.strip().startswith('#') or not re.search(r'\d\s+\d', x))

# ------------------------------------------------------------------ roles
RE_AGE   = re.compile(r'^(cal)?(yr|year)s?[\s_-]?bp$|age|^time$|calyrbp|yrbp|kyr|kabp|calka', re.I)
RE_MA    = re.compile(r'(?<![a-z])ma(?![a-z])|calma|mabp|million', re.I)
RE_KA    = re.compile(r'(?<![a-z])ka(?![a-z])|kyr|kiloyear|calka', re.I)
RE_DEPTH = re.compile(r'depth|mbsf|mcd|ambsf|^cm[_ ]?top$', re.I)
RE_METRE = re.compile(r'_m$|\(m\)|mbsf|mcd|ambsf|meter|metre|depth_m', re.I)
# 'sst' must be either the exact uppercase token SST (so FebSST, AlkSST, TEX_SST,
# SST_P all still match) or a lowercase sst that does not sit inside a longer word
# (so sst_uk_med, sst-mg/ca, sstsn_bay match). Matching plain 'sst' case-insensitively
# - as the old rule did - also matches "MassStandardAdded", whose 'ssSt' spans a word
# join; that column, a GDGT standard mass in micrograms, was being written into the
# SST column of ODP 1146 as a constant 0.13.
# Only the bare 'sst' branch is case-sensitive; the spelled-out branches stay
# case-insensitive via scoped (?i:...), otherwise 'Alkenone temp (deg.C)' in
# harada2006.txt (MD01-2412) stops matching and the core is lost entirely.
RE_SST   = re.compile(r'SST|(?<![A-Za-z])sst'
                      r'|(?i:sea[\s_-]?surface[\s_-]?temp|alkenone\s*temp|^temp(erature)?$)')
# Subsurface / thermocline temperature is NOT sea-surface temperature. u939-tab.txt
# offers both 'sst-sub.uk37' ("Subsurface Sea Temperature") and 'sst-warm.uk37'
# ("Sea Surface Temperature ... warm season"); the subsurface column was winning.
RE_SUBSURF = re.compile(r'sub[\s_.-]?surf|subsurface|\bsubt\b|sub[\s_.-]?t_|thermocline|\bsst-sub|_sub\b', re.I)

# Credible/confidence-interval bound columns. These are NOT the central estimate:
# tran2025 (IODP U1482), windler2019 (MD98-2152) and windler2022 (VM19-193) all
# publish sst_*_low / sst_*_med / sst_*_high (or _5/_med/_95) triplets, and the
# bound sorted first was being taken as the SST.
# Token sets, not substring regexes: windler2019 names its triplet sst_uk_5 /
# sst_uk_med / sst_uk_95, where the bound marker is a bare number in a suffix
# position. A regex with a letter-boundary prefix cannot see '_5' after 'uk', and
# a looser one makes '95' match the low set as well as the high set.
BOUND_LO_TOK = {'low', 'lower', 'lo', 'min', 'lwr', 'l95', 'p05', 'p5', '5', '2.5', '025',
                'cil', 'lb', 'lower95', 'minus'}
BOUND_HI_TOK = {'high', 'higher', 'hi', 'max', 'upr', 'upper', 'u95', 'p95', '95', '97.5',
                '975', 'ciu', 'ub', 'upper95', 'plus'}
CENTRAL_TOK = {'med', 'median', 'mean', 'avg', 'average', 'best', 'mid', '50', 'est', 'p50'}


def _toks(name):
    return set(re.split(r'[^a-z0-9.]+', (name or '').lower())) - {''}


def is_lo(c):
    return bool(_toks(c) & BOUND_LO_TOK)


def is_hi(c):
    return bool(_toks(c) & BOUND_HI_TOK)


def is_central(c):
    return bool(_toks(c) & CENTRAL_TOK)
# 90% credible interval -> 2sd. Half-width/1.645 = 1sd, x2 = 2sd, i.e. x1.2158.
CI90_TO_2SD = 2.0 / 1.6449
RE_ANOM  = re.compile(r'anom|resid|\bback\b', re.I)
RE_ERR   = re.compile(r'(?<![a-z])sd(?![a-z])|std|stdev|error|(?<![a-z])err|uncert|\+/-|±|95%|conf|sig|precision', re.I)
RE_AGEERR = re.compile(r'age.?(uncert|err|sd)|(uncert|err|sd).?age|age_uncert', re.I)
SEASON_AUG = re.compile(r'aug|warm|summer|jja|jas|jul', re.I)
# 'jfm' added 2026-07-21: MD04-2845 publishes SST.for-JAS alongside SST.for-JFM
# (January-February-March). Without it the winter column went unrecognised and the
# core was wrongly recorded as summer-signal-only.
SEASON_FEB = re.compile(r'feb|cold|winter|djf|jan|jfm', re.I)
PROXY_PATS = [('UK37', r"uk'?\.?37|alkenone"), ('MgCa', r"mg/?ca"), ('TEX86', r"tex\.?86"),
              ('Diatom', r'diatom'), ('Radiolaria', r'radiolaria|radio\b'),
              ('Cocc', r'coccolith|nannofossil'),
              ('Foram', r'foramin|transfer function|modern analog|faunal|plank|g\.\s?ruber|sacculifer')]
SPECIES = re.compile(r'(G\.|Globigerinoides|Globorotalia|Neogloboquadrina|Uvigerina|Cibicidoides)'
                     r'\.?\s*[a-z]+(\s+(white|pink|sensu stricto|dextral|sinistral))?', re.I)
FREE_ERR = re.compile(r'[^.\n]*(error|uncertaint|precision|\+/-|±)[^.\n]*(degree|deg\.?\s*c|°c)[^.\n]*\.', re.I)

def pick_proxy(text):
    for name, pat in PROXY_PATS:
        if re.search(pat, text, re.I):
            return name
    return ''

def depth_to_cm(val, colname, allvals):
    """Return (value_in_cm, note or None)."""
    if val is None:
        return None, None
    if RE_METRE.search(colname or ''):
        return val * 100.0, 'm->cm'
    mx = max([v for v in allvals if v is not None] or [0])
    if mx < 60:                                  # cm range would be implausibly short
        return val * 100.0, 'm->cm (by magnitude)'
    return val, None

# ------------------------------------------------------------------ extract
def extract(path, meta):
    L = read_lines(path)
    if any('<html' in x.lower() for x in L[:40]):
        return [], 'HTML directory index, not a data file'
    tables = find_all_tables(L)
    if not tables:
        return [], 'Could not locate a numeric data table'
    hdr = header_text(L)
    LEG = legends(L)
    best = []
    for cols, rows, li in tables:
        # width actually present in the data
        widths = {}
        for r in rows[:40]:
            widths[len(r)] = widths.get(len(r), 0) + 1
        w = max(widths, key=widths.get) if widths else len(cols)
        leg = legend_for(LEG, li)
        # If the detected header does not line up with the data, prefer the
        # 'Column N:' legend; NOAA files often have no usable header row.
        if leg and (len(cols) != w or sum(1 for c in cols if is_num(c)) > 0):
            cols2 = [leg.get(k, cols[k] if k < len(cols) else f'col{k+1}') for k in range(w)]
            # the 'header' we consumed was really the first data row - put it back
            if len(cols) == w and sum(1 for c in cols if is_num(c)) >= w * 0.4:
                rows = [cols] + rows
        elif len(cols) != w:
            cols2 = (cols + [f'col{k+1}' for k in range(len(cols), w)])[:w]
        else:
            cols2 = cols
        cands = [cols2]
        # A leading label column can span several whitespace-separated fields
        # (e.g. '167 1018C  1H  1'), shifting every named column rightwards.
        src = leg and [leg[k] for k in sorted(leg)] or cols
        if len(src) < w:
            pad = w - len(src)
            cands.append([f'lbl{k+1}' for k in range(pad)] + list(src))
        for cc in cands:
            res = _one_table(cc, rows, hdr, meta, li)
            if res:
                best.extend(res)
                break
    if not best:
        return [], 'No table in this file pairs an age variable with an SST variable inside 115-140 ka'
    # keep richest series per proxy
    bypx = defaultdict(list)
    for s in best:
        bypx[s['proxy']].append(s)
    return [max(v, key=lambda s: len(s['recs'])) for v in bypx.values()], None

def _one_table(cols, rows, hdr, meta, li):
    idx_sst = [i for i, c in enumerate(cols)
               if RE_SST.search(c) and not RE_ANOM.search(c) and not RE_ERR.search(c)]
    # Per-core override from decisions.py: AC has named the exact SST column to
    # use (e.g. ODP 847 must use 'sst-mg/ca.adj', not the unadjusted 'sst-mg/ca').
    force = (meta or {}).get('_force_sst_col')
    if force:
        pick = [i for i in idx_sst if force.lower() in cols[i].lower()]
        if pick:
            idx_sst = pick

    # ---- credible-interval triplets -------------------------------------
    # Group low/med/high siblings so the CENTRAL estimate becomes the SST and the
    # two bounds become the uncertainty, instead of whichever bound sorted first.
    def _fam(name):
        return ''.join(t for t in re.split(r'[^a-z0-9.]+', (name or '').lower())
                       if t and t not in BOUND_LO_TOK and t not in BOUND_HI_TOK
                       and t not in CENTRAL_TOK)

    ci_bounds = {}                      # central col index -> (lo idx, hi idx)
    if len(idx_sst) > 1:
        fams = defaultdict(list)
        for i in idx_sst:
            fams[_fam(cols[i])].append(i)
        drop = set()
        for members in fams.values():
            if len(members) < 2:
                continue
            cen = [i for i in members if is_central(cols[i])]
            lo = [i for i in members if is_lo(cols[i]) and i not in cen]
            hi = [i for i in members if is_hi(cols[i]) and i not in cen]
            if cen and (lo or hi):
                ci_bounds[cen[0]] = (lo[0] if lo else None, hi[0] if hi else None)
                drop |= set(lo) | set(hi)
        if drop:
            idx_sst = [i for i in idx_sst if i not in drop]

    # ---- subsurface columns ---------------------------------------------
    # Only fall back to a subsurface/thermocline column if there is no true
    # surface column in the table.
    surf = [i for i in idx_sst if not RE_SUBSURF.search(cols[i])]
    if surf and len(surf) < len(idx_sst):
        idx_sst = surf
    idx_age = [i for i, c in enumerate(cols) if RE_AGE.search(c) and not RE_DEPTH.search(c)
               and not RE_ERR.search(c)]
    if not idx_sst or not idx_age:
        return []
    agei = idx_age[0]
    an = cols[agei]
    av = [num(r[agei]) for r in rows if len(r) > agei]
    av = [v for v in av if v is not None]
    if not av:
        return []
    # Unit inference. Key insight: to reach our 115-140 ka window a column expressed
    # in YEARS must have max >= 115000. Anything smaller cannot be years.
    if RE_MA.search(an):
        scale = 1e6
    elif RE_KA.search(an):
        scale = 1e3
    elif max(av) >= 115000:
        scale = 1.0
    elif max(av) >= 115:
        scale = 1e3
    else:
        scale = 1e6
    depthi = next((i for i, c in enumerate(cols) if RE_DEPTH.search(c)), None)
    dvals = [num(r[depthi]) for r in rows if depthi is not None and len(r) > depthi] if depthi is not None else []

    # ---- seasonal pairing -------------------------------------------------
    aug = [i for i in idx_sst if SEASON_AUG.search(cols[i])]
    feb = [i for i in idx_sst if SEASON_FEB.search(cols[i])]
    groups = []
    if aug and feb:
        groups.append(dict(kind='seasonal', si=aug[0], si2=feb[0],
                           name=f'{cols[aug[0]]}+{cols[feb[0]]}'))
        used = {aug[0], feb[0]}
    else:
        used = set()
    single_season = {}
    for i in idx_sst:
        if i in used:
            continue
        if SEASON_AUG.search(cols[i]):
            single_season[i] = 'warm/summer'
        elif SEASON_FEB.search(cols[i]):
            single_season[i] = 'cold/winter (February = austral summer in the Southern Ocean)'
    for i in idx_sst:
        if i not in used:
            groups.append(dict(kind='plain', si=i, si2=None, name=cols[i]))

    out = []
    for g in groups:
        si = g['si']
        notes, method = [], []
        # Proxy attribution. Falling straight through to the whole-table name
        # soup mislabels multi-proxy tables: mohtadi2010b.txt carries both
        # "Mg/CaG.r"/"SST G.r" and "UK'37"/"AlkSST", and PROXY_PATS tries UK37
        # first, so the Mg/Ca series was being recorded as UK37. So: try the SST
        # column's own name, then only those columns that share a distinguishing
        # token with it, and only then the whole table.
        def _kin(name):
            """Columns naming the same carrier as this SST column. Strip the SST
            wording and any unit bracket, and use what is left as a substring
            key: 'SST G.r [degC]' -> 'g.r', which is contained in both
            'd18OG.r' and 'Mg/CaG.r' but not in 'Mg/CaN.d'. Substring, not
            token, matching - 'Mg/CaG.r' tokenises as mg + cag.r, so an exact
            token test would miss it."""
            # Tokenise first, then drop stopwords. Doing it with word-boundary
            # regexes fails on 'sst_tex_med', because '_' is a word character so
            # \bsst\b never fires and the key comes out as 'ssttex', matching
            # nothing. The series was then labelled UK37 like its sibling and the
            # two collapsed into one tab (IODP U1482).
            STOP = {'sst', 'temp', 'temperature', 'sea', 'surface', 'degc', 'deg', 'c',
                    'degree', 'celsius'}
            key = ''.join(t for t in re.split(r'[^a-z0-9.]+', re.sub(r'\[.*?\]', ' ', (name or '').lower()))
                          if t and t not in STOP and t not in BOUND_LO_TOK
                          and t not in BOUND_HI_TOK and t not in CENTRAL_TOK)
            if len(key) < 2:
                return ''
            return ' '.join(cols[j] for j in range(len(cols))
                            if j != si and not RE_SST.search(cols[j]) and not RE_ERR.search(cols[j])
                            and key in re.sub(r'[^a-z0-9.]', '', cols[j].lower()))

        proxy = (pick_proxy(cols[si]) or pick_proxy(_kin(cols[si])) or pick_proxy(' '.join(cols))
                 or pick_proxy(meta.get('study_name', '') + ' ' + meta.get('tnotes', ''))
                 or meta.get('proxy', '') or '')
        # --- error column: must belong to SST, never to age or proxy -------
        def err_for(k):
            base = re.sub(r'[^a-z0-9]', '', cols[k].lower())
            cands = []
            for j, c in enumerate(cols):
                if j == k or not RE_ERR.search(c) or RE_AGEERR.search(c):
                    continue
                cl = re.sub(r'[^a-z0-9]', '', c.lower())
                if RE_SST.search(c) or cl.startswith(base[:6]):
                    cands.append(j)
            return cands[0] if cands else None
        ei = err_for(si)
        ei2 = err_for(g['si2']) if g['si2'] is not None else None
        one_sd = ei is not None and not re.search(r'2\s?s|95%|two', cols[ei], re.I)
        ci = ci_bounds.get(si)          # (lo idx, hi idx) from a low/med/high triplet

        # --- Method text ---------------------------------------------------
        if g['kind'] == 'seasonal':
            method.append(f'Annual SST computed as the mean of "{cols[si]}" (warm/August) and '
                          f'"{cols[g["si2"]]}" (cold/February) reported in the source file.')
            notes.append(f'Seasonal columns "{cols[si]}"/"{cols[g["si2"]]}" averaged to annual mean')
        elif si in single_season:
            method.append(f'SEASONAL SIGNAL: source reports only "{cols[si]}" '
                          f'({single_season[si]}); no counter-season column available, '
                          f'so no annual mean could be formed.')
            notes.append(f'SUMMER/SEASONAL SIGNAL CORE - only "{cols[si]}" ({single_season[si]}) available')
        alt = [cols[i] for i in idx_sst if i != si and i != g['si2']]
        if alt:
            method.append(f'Source reports alternative SST column(s): {", ".join(alt)}. '
                          f'Value used here is "{cols[si]}".')
            notes.append(f'Alternative SST column(s) present ({", ".join(alt)}); used "{cols[si]}"')
        sp = SPECIES.search(hdr)
        if sp:
            method.append(f'Proxy carrier: {sp.group(0)}.')
        fe = FREE_ERR.search(hdr)
        if fe:
            method.append('Stated uncertainty in source: ' + ' '.join(fe.group(0).split()))
        if ei is not None:
            method.append(f'Uncertainty column used: "{cols[ei]}"'
                          + (' (1sd, doubled to 2sd).' if one_sd else ' (already 2sd).'))
        if ci:
            bn = [cols[i] for i in ci if i is not None]
            method.append(
                f'SST is the CENTRAL estimate "{cols[si]}"; the source also publishes the '
                f'credible-interval bounds {", ".join(bn)}, which are NOT the mean and must '
                f'not be read as such. Those bounds are a 90% interval, so each side has been '
                f'rescaled about the central value by 2/1.645 = {CI90_TO_2SD:.4f} to express '
                f'it as +/-2sd for columns S and U (AC, 2026-07-21). The rescaling is applied '
                f'to each side independently, so an asymmetric interval remains asymmetric.')
            notes.append(f'Credible-interval triplet: central "{cols[si]}" used as SST; '
                         f'bounds {", ".join(bn)} rescaled from 90% CI to 2sd')

        recs = []
        for r in rows:
            if len(r) <= max(si, agei):
                continue
            a, s1 = num(r[agei]), num(r[si])
            if a is None or s1 is None:
                continue
            a *= scale
            if not (AGE_MIN <= a <= AGE_MAX):
                continue
            e1 = num(r[ei]) if ei is not None and len(r) > ei else None
            if g['kind'] == 'seasonal':
                s2 = num(r[g['si2']]) if len(r) > g['si2'] else None
                if s2 is None:
                    continue
                e2 = num(r[ei2]) if ei2 is not None and len(r) > ei2 else None
                sst = (s1 + s2) / 2.0
                err = None
                if e1 is not None and e2 is not None:
                    err = (e1 + e2) / 2.0
                elif e1 is not None:
                    err = e1
                extra = dict(aug=s1, feb=s2)
            else:
                sst, err, extra = s1, e1, {}
            if err is not None and one_sd:
                err *= 2
            if not (SST_LO <= sst <= SST_HI):
                continue
            lo_v = hi_v = None
            if ci:
                # 90% credible interval -> 2sd, per AC 2026-07-21. Half-width /1.645
                # gives 1sd; doubling gives 2sd. Each side is scaled about the median
                # separately, so an asymmetric Bayesian interval stays asymmetric.
                bl = num(r[ci[0]]) if ci[0] is not None and len(r) > ci[0] else None
                bh = num(r[ci[1]]) if ci[1] is not None and len(r) > ci[1] else None
                if bl is not None:
                    lo_v = sst - (sst - bl) * CI90_TO_2SD
                if bh is not None:
                    hi_v = sst + (bh - sst) * CI90_TO_2SD
            d = num(r[depthi]) if depthi is not None and len(r) > depthi else None
            d, dn = depth_to_cm(d, cols[depthi] if depthi is not None else '', dvals)
            if lo_v is None and err is not None:
                lo_v = sst - err
            if hi_v is None and err is not None:
                hi_v = sst + err
            recs.append(dict(depth=d, pval=None, sst=sst, age=a,
                             lo=lo_v, hi=hi_v, **extra))
        if not recs:
            continue
        recs.sort(key=lambda x: x['age'])
        if dvals and any(r['depth'] is not None for r in recs):
            _, dn = depth_to_cm(1.0, cols[depthi], dvals)
            if dn:
                notes.append(f'Depth converted to cm ({dn}) from column "{cols[depthi]}"')
                method.append(f'Depth converted from metres to centimetres (source column "{cols[depthi]}").')
        if ei is None and not ci:
            notes.append('No SST uncertainty column in source; -2sd/+2sd blank')
        if len(recs) < 5:
            notes.append(f'SPARSE: only {len(recs)} entries within 115-140 ka')
        # proxy value column
        pvi = None
        if proxy:
            pat = dict(PROXY_PATS)[proxy]
            for j, c in enumerate(cols):
                if j == si or RE_SST.search(c) or RE_ERR.search(c):
                    continue
                if re.search(pat, c, re.I):
                    pvi = j
                    break
        if pvi is not None:
            k = 0
            for r in rows:
                if len(r) <= max(si, agei):
                    continue
                a, s1 = num(r[agei]), num(r[si])
                if a is None or s1 is None:
                    continue
                a *= scale
                if not (AGE_MIN <= a <= AGE_MAX):
                    continue
                if k < len(recs):
                    recs[k]['pval'] = num(r[pvi]) if len(r) > pvi else None
                    k += 1
        # If the file also carries an SST ANOMALY column matching the SST column
        # we used, derive the absolute-SST baseline (SST - anomaly). Needed for
        # e.g. IODP U1485, where the reported error is on the anomaly.
        baseline = None
        base_key = re.sub(r'[^a-z0-9]', '', cols[si].lower())
        for j, c in enumerate(cols):
            if not RE_ANOM.search(c) or RE_ERR.search(c):
                continue
            cl = re.sub(r'[^a-z0-9]', '', c.lower())
            if not cl.startswith(base_key[:5]):
                continue
            diffs = []
            for r in rows:
                if len(r) <= max(si, j):
                    continue
                a, b = num(r[si]), num(r[j])
                if a is not None and b is not None:
                    diffs.append(a - b)
            if diffs:
                baseline = sum(diffs) / len(diffs)
                break

        out.append(dict(col=g['name'], proxy=proxy, recs=recs, notes=notes,
                        method=' '.join(method), table_line=li,
                        seasonal=(g['kind'] == 'seasonal'), _baseline=baseline))
    return out
