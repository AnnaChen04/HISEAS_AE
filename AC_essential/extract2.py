#!/usr/bin/env python3
"""
Extraction router.

Single entry point used by write_outputs.py. Dispatch order:

  1. special_formats.py  - files needing bespoke handling (herbert2016 stacks,
                           ikehara2000, the hand-edited tripati xlsx, ...)
  2. manual_specs.py     - hand-mapped column specs from Anna's section I,
                           plus her explicit MANUAL_REJECT list
  3. parser2.py          - the general multi-table parser

Returns (series_list, error) exactly like parser2.extract, so it is a drop-in
replacement for the old build_mining_outputs.extract.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser2 as P
import manual_specs as M

try:
    import special_formats as SP
except ImportError:                     # not written yet
    SP = None

BASE = P.BASE
TXT = P.TXT

# re-export the helpers write_outputs needs so it can import from one place
read_lines = P.read_lines
header_text = P.header_text
num = P.num
find_all_tables = P.find_all_tables


def route(path, meta):
    """(series_list, error). series_list may hold >1 series (multi-proxy, or a
    stack file that splits into several cores - those carry ser['core_override'])."""
    fn = os.path.basename(path)

    if SP is not None and fn in getattr(SP, 'SPECIAL', {}):
        return SP.extract_special(path, meta)

    if fn in M.MANUAL_REJECT:
        return [], M.MANUAL_REJECT[fn]

    out, err = M.extract_manual(path, meta)
    if out is not None:                 # None means "not hand-mapped, fall through"
        return out, err

    return P.extract(path, meta)


# ---------------------------------------------------------------- provenance
def parser_used(path):
    """Which branch handled this file - written into the flags report so Anna
    can see at a glance what was automatic and what was hand-specified."""
    fn = os.path.basename(path)
    if SP is not None and fn in getattr(SP, 'SPECIAL', {}):
        return 'special'
    if fn in M.MANUAL_REJECT:
        return 'manual-reject'
    if fn in M.SPECS:
        return 'manual-spec'
    return 'parser2'
