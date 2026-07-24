#!/usr/bin/env python3
"""
Extra content for SST_Depth_Only_For_Jerry.xlsx.

Two sources, both from handoff.md:

  * section 6.3 - mined_txt/Ruddiman_Fossil_Plankton/, 31 files of SST vs depth
    with no age model. Filename is the core ID, the last two numeric columns are
    SSTwarm and SSTcold, and lon/lat are on line 1.
  * section 6.4 - PS1778-5, E45-29 (249-284 cm), E49-17, E49-18: cores that DO
    have an age model, but whose depth series is sampled more densely than the
    age series, so the depth-vs-SST view is worth giving Jerry separately.

Annual SST is the mean of SSTwarm and SSTcold, matching the treatment of paired
Aug/Feb columns in the main sheet.
"""
import os, re, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser2 as P

RUDDIMAN = os.path.join(P.TXT, 'Ruddiman_Fossil_Plankton')

# Filenames whose core ID must NOT be taken from the filename. Confirmed by AC
# 2026-07-21.
FILENAME_OVERRIDE = {
    # line 1 and all 155 row labels say K714-14; only the filename says k714_4.
    # Two in-file sources beat one filename, and K714-15 has its own file, so
    # there is no collision.
    'k714_4.sst.txt': ('K714-14',
                       'Core ID taken from the file header and row labels ("K714-14"), not from '
                       'the filename ("k714_4"). AC confirmed K714-14 on 2026-07-21.'),
    # v28_14.sst.txt originally held V29-177 (a byte-identical copy of
    # v29_177.sst.txt) followed by an embedded 1992 email header and then the
    # real V28-14 block. AC directed on 2026-07-21 that the V29-177 rows and the
    # email lines be deleted from the source file; the untouched original is
    # kept alongside as v28_14.sst.txt.orig_backup.
    'v28_14.sst.txt': ('V28-14',
                       'Source file originally concatenated V29-177 (duplicate of '
                       'v29_177.sst.txt) with V28-14, separated by an embedded 1992 email '
                       'header. The V29-177 and email lines were removed from the file on '
                       'AC instruction 2026-07-21; original preserved as '
                       'v28_14.sst.txt.orig_backup. Coordinates 64 47.0 N, 29 34.0 W '
                       'confirmed by AC and matching the file\'s own V28-14 header line.'),
}

# section 6.4 - cores whose depth series is denser than their age series.
# depth_range limits the rows written, where AC specified one.
DENSE_DEPTH = [
    dict(core='PS1778-5', file='ps1778-5-tab.txt', depth_range=None),
    dict(core='E45-29',   file='e45-29-tab.txt',   depth_range=(249.0, 284.0)),
    dict(core='E49-17',   file='e49-17-tab.txt',   depth_range=None),
    dict(core='E49-18',   file='e49-18-tab.txt',   depth_range=None),
]

DMS = re.compile(r'(\d+)\s+(\d+\.?\d*)\s*([NSEW])', re.I)


def _coords(line):
    """Line 1 is e.g. 'K708-1   50 00.0 N   23 45.0 W  4053 AOf13B-4CE'.
    Returns (lon, lat, elev_m_negative)."""
    lat = lon = elev = None
    for m in DMS.finditer(line):
        deg, mins, hemi = float(m.group(1)), float(m.group(2)), m.group(3).upper()
        val = deg + mins / 60.0
        if hemi in 'SW':
            val = -val
        if hemi in 'NS':
            lat = val
        else:
            lon = val
    tail = DMS.split(line)[-1] if DMS.search(line) else line
    m = re.search(r'(\d{3,5})', tail)
    if m:
        elev = -abs(float(m.group(1)))
    return lon, lat, elev


def _last3(tokens):
    """Last three numeric fields = (depth, SSTwarm, SSTcold). This holds for both
    layouts present: the ' K708  1  0.0 17.0 11.0' form, where a label spans two
    whitespace fields, and the '552AC4-1 0 1400.5 14.1 8.9' form, which carries an
    extra in-section depth column before the composite depth."""
    nums = [P.num(t) for t in tokens]
    nums = [(i, v) for i, v in enumerate(nums) if v is not None]
    if len(nums) < 3:
        return None
    return [v for _, v in nums[-3:]]


def _core_from_header(line, fn):
    """Core ID from the line-1 header, i.e. everything before the first
    coordinate group. AC's ruling on k714_4.sst.txt (2026-07-21) was that the
    file's own contents outrank the filename, so the same rule is applied to all
    31 files. It also yields cleaner IDs than the filenames do: '552A' rather
    than '552A-FUL', 'V23-81' rather than 'V23-81S', 'LK4' rather than 'LK-4'."""
    m = DMS.search(line)
    head = line[:m.start()] if m else line
    head = re.sub(r'^\s*HOLE\s+', '', head.strip(), flags=re.I)
    head = re.sub(r'\s+', ' ', head).strip()
    if not head:
        return re.sub(r'\.sst$', '', os.path.splitext(fn)[0]).upper().replace('_', '-')
    # 'CHK 9' -> 'CHK9'; leave already-hyphenated IDs like 'K708-1' alone
    return head.replace(' ', '') if re.fullmatch(r'[A-Za-z]+ \d+', head) else head


def ruddiman_cores():
    """Yield dicts ready for the Jerry writer."""
    out = []
    for path in sorted(glob.glob(os.path.join(RUDDIMAN, '*.txt'))):
        fn = os.path.basename(path)
        L = P.read_lines(path)
        if not L:
            continue
        if fn in FILENAME_OVERRIDE:
            core, note = FILENAME_OVERRIDE[fn]
        else:
            core, note = _core_from_header(L[0], fn), None
        lon, lat, elev = _coords(L[0])
        rows = []
        for line in L[1:]:
            if not line.strip():
                continue
            t = P.split_row(line)
            v = _last3(t)
            if v is None:
                continue
            depth, warm, cold = v
            if not (P.SST_LO <= warm <= P.SST_HI) or not (P.SST_LO <= cold <= P.SST_HI):
                continue
            rows.append(dict(depth=depth, sst=(warm + cold) / 2.0, warm=warm, cold=cold))
        if not rows:
            continue
        rows.sort(key=lambda x: (x['depth'] if x['depth'] is not None else 0))
        method = ('Faunal transfer-function SST from the Ruddiman fossil-plankton compilation. '
                  'The source reports SSTwarm and SSTcold; the value given here is their mean, '
                  'i.e. an annual estimate, with the two seasonal values retained in the '
                  'SSTwarm/SSTcold columns. Depth in cm. NO AGE MODEL accompanies this file, '
                  'which is why the record appears in this sheet rather than the main one.')
        if note:
            method += ' ' + note
        out.append(dict(core=core, lon=lon, lat=lat, elev=elev, proxy='Foram',
                        rows=rows, src=f'Ruddiman_Fossil_Plankton/{fn}',
                        method=method, season='warm-cold averaged'))
    return out


def dense_depth_cores(meta_by_file):
    """section 6.4 cores, written as depth-vs-SST regardless of their age model."""
    out = []
    for spec in DENSE_DEPTH:
        path = os.path.join(P.TXT, spec['file'])
        if not os.path.exists(path):
            continue
        L = P.read_lines(path)
        tables = P.find_all_tables(L)
        if not tables:
            continue
        cols, rows_raw, li = max(tables, key=lambda t: len(t[1]))
        di = next((i for i, c in enumerate(cols) if P.RE_DEPTH.search(c)), None)
        sst_idx = [i for i, c in enumerate(cols)
                   if P.RE_SST.search(c) and not P.RE_ANOM.search(c) and not P.RE_ERR.search(c)]
        if di is None or not sst_idx:
            continue
        aug = [i for i in sst_idx if P.SEASON_AUG.search(cols[i])]
        feb = [i for i in sst_idx if P.SEASON_FEB.search(cols[i])]
        paired = bool(aug and feb)
        rows = []
        for r in rows_raw:
            if len(r) <= di:
                continue
            d = P.num(r[di])
            if d is None:
                continue
            if spec['depth_range'] and not (spec['depth_range'][0] <= d <= spec['depth_range'][1]):
                continue
            if paired:
                w = P.num(r[aug[0]]) if len(r) > aug[0] else None
                c = P.num(r[feb[0]]) if len(r) > feb[0] else None
                if w is None or c is None:
                    continue
                s = (w + c) / 2.0
            else:
                w = c = None
                s = P.num(r[sst_idx[0]]) if len(r) > sst_idx[0] else None
                if s is None:
                    continue
            if not (P.SST_LO <= s <= P.SST_HI):
                continue
            rows.append(dict(depth=d, sst=s, warm=w, cold=c))
        if not rows:
            continue
        rows.sort(key=lambda x: x['depth'])
        m = meta_by_file.get(spec['file'], {})
        rng = (f" Restricted to depths {spec['depth_range'][0]:.0f}-{spec['depth_range'][1]:.0f} cm "
               f"per AC (section 6.4)." if spec['depth_range'] else '')
        method = (f'Depth-vs-SST view of {spec["core"]}. This core DOES have an age model and also '
                  f'appears in the main spreadsheet; it is repeated here because its depth series is '
                  f'sampled more densely than its age series, so more of the record is visible '
                  f'against depth (section 6.4).{rng}'
                  + (' Annual SST is the mean of the source August and February columns.'
                     if paired else ''))
        out.append(dict(core=spec['core'], lon=m.get('lon'), lat=m.get('lat'),
                        elev=m.get('elev'), proxy=m.get('proxy') or 'Foram', rows=rows,
                        src=spec['file'], method=method,
                        season='aug-feb averaged' if paired else None))
    return out
