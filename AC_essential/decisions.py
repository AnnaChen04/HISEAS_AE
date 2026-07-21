#!/usr/bin/env python3
"""
Anna's confirmed per-core decisions, transcribed from handoff.md section 6.

Keeping these in one file means the screening pipeline never re-litigates a
judgement call that has already been made, and every exclusion carries the
reason Anna gave for it into rejected_with_reason.docx.

Keys are matched on a normalised core ID (lowercase, alphanumerics only), so
'ODP 847', 'odp847' and 'ODP-847' all resolve to the same entry.
"""
import re

PSUF = r'_(UK37|MgCa|TEX86|Foram|Diatom|Radiolaria|Cocc)$'


def norm(s):
    return re.sub(r'[^a-z0-9]', '', re.sub(PSUF, '', str(s), flags=re.I).lower())


# --------------------------------------------------------------- exclusions
_LONG_SPAN = ('Record spans an interval far longer than the study window (of order 10-20 Myr), '
              'so the depth-age model is too poorly constrained over the Last Interglacial to '
              'place individual samples reliably within 115-140 ka.')

EXCLUDE = {norm(k): v for k, v in {
    'DSDP 588': _LONG_SPAN,
    'ODP 1021': _LONG_SPAN,
    'ODP 1088': _LONG_SPAN,
    'ODP 1208': _LONG_SPAN,
    'ODP 1010': _LONG_SPAN,
    'ODP 883': _LONG_SPAN + ' No samples fall within 115-140 ka (record is 2705-11345 ka).',
    'ODP 884': _LONG_SPAN + ' No samples fall within 115-140 ka (record is 6546-7863 ka).',
    'E49-23': 'Only one sample falls within 115-140 ka.',
    'MD97-2141': 'Has 255 rows inside 115-140 ka but the SST field is blank in every one of them.',
    'IODP U1488': 'No rows fall within 115-140 ka.',
    'DSDP 475': 'No rows fall within 115-140 ka.',
    'ODP 806': 'Two conflicting age models are reported for the same samples (Medina-Elizalde '
               '& Lea 2005 and LR04), and only two samples fall within 115-140 ka under either '
               'model; the record is too sparsely sampled across the study window to constrain '
               'LIG SST.',
    'ODP 980': 'Header of the source file (oppo2006.txt) is not interpretable, even manually; '
               'column definitions must be recovered from Oppo et al. 2006 before the data can '
               'be used. Deferred.',
}.items()}

# Cores kept despite being sparse. AC's reasoning: the AGES are well sampled,
# it is only that some samples lack a proxy-to-SST conversion. Do not re-reject
# these on point count.
KEEP_SPARSE = {norm(k) for k in (
    'Q200', 'R657', 'U938', 'ODP 847', 'PS1778-5', 'RS147-GC14',
    'E45-29', 'E49-17', 'E49-18', 'E49-21', 'DSDP 593',
)}

# Tabs to delete from same_core_different_chronology.xlsx: AC confirmed the
# versions already in her own spreadsheet are the correct ones.
DROP_FROM_CHRONO = {norm(k) for k in ('RC13-110', 'NH22P')}

# 'Lingtai' was previously filed under "not marine". It is the ODP 1146
# deep-ocean dataset and must not appear there.
NOT_MARINE_REMOVE = {norm('Lingtai')}


# ------------------------------------------------------- per-core amendments
# Applied after extraction. Recognised keys:
#   sst_col       - substring of the SST column name that MUST be used
#   err_abs       - absolute +/-2sd to apply to every point (already 2sd)
#   proxy         - override the detected proxy
#   season        - value for col AA
#   method_add    - text appended to col Y
#   elev_nan_ok   - extract even though elevation is unknown
#   anomaly_baseline - reported error is on the anomaly; derive SST error from
#                      (SST - SST_anom)
FIXES = {norm(k): v for k, v in {
    'ODP 847': dict(
        sst_col='sst-mg/ca.adj',
        method_add='SST taken from the "sst-mg/ca.adj" column, which is adjusted for variation '
                   'in seawater Mg/Ca, in preference to the unadjusted Mg/Ca SST column. '
                   'Confirmed by AC.'),
    'RS147-GC14': dict(
        err_abs=1.5,
        method_add='Source states "SSTs from Uk\'37 have an error of +/-1.5 degC"; this free-text '
                   'value is used for the -2sd and +2sd columns as stated (not doubled). '
                   'Confirmed by AC.'),
    'LPAZ-21P': dict(
        err_abs=0.3,
        method_add='SAME CORE as Hoffman et al. (2017) "LAPAZ21". Source states a precision of '
                   '0.15 degC, which gives +/-0.3 degC at 2sd. Confirmed by AC.'),
    'PS1778-5': dict(
        proxy='Radiolaria',
        season='summer signal',
        method_add='Mean sea surface temperature, summer, Dec-March, 10 m water depth, calculated '
                   'from radiolaria, using Transfer function (Imbrie & Kipp, 1971, in Turekian, '
                   'Yale Univ Press).'),
    'SO136-111': dict(
        season='summer signal',
        method_add='Source reports February SST. In the Southern Ocean February is austral summer, '
                   'so this is a SUMMER-SIGNAL core, not a winter one. Confirmed by AC.'),
    'MD99-2331': dict(
        season='summer signal',
        method_add='SST column "SST.for-JAS" is a July-August-September (boreal summer) estimate, '
                   'so this is a SUMMER-SIGNAL core. The accompanying "-i" and "-s" columns are '
                   'probably inferior and superior bounds but are not defined in the source, so '
                   'they are deliberately not used as uncertainties.'),
    'ODP 820': dict(
        elev_nan_ok=True,
        method_add='Elevation is not given by the source and is left as NaN; AC confirmed the data '
                   'should still be extracted.'),
    'IODP U1485': dict(
        anomaly_baseline=True,
        method_add='NOT an interval-average product: the source reports a per-sample age for every '
                   'SST value. The reported uncertainty applies to the SST ANOMALY, so the '
                   'baseline was derived as (SST - SST_anom) and used to place the error on '
                   'absolute SST. Confirmed by AC.'),
    # AC 2026-07-21: this core has BOTH seasons (SST.for-JAS and SST.for-JFM), so
    # section 4's rule applies and the annual mean is formed. It is NOT a
    # summer-signal core - unlike MD99-2331, where only JAS exists.
    'MD04-2845': dict(
        method_add='The source reports both SST.for-JAS (July-August-September) and '
                   'SST.for-JFM (January-February-March); the annual value used here is '
                   'their mean, per AC 2026-07-21. The "-i"/"-s" columns are probably '
                   'inferior and superior bounds but are not defined in the source, so they '
                   'are deliberately not used as uncertainties.'),
    'MD01-2421': dict(
        sst_col='SST Corr.',
        method_add='SST taken from the "SST Corr." column, not the raw "SST(degC)" column. '
                   '"SST Corr." applies the published correction for bottom-water temperature '
                   '(the study assumes the LGM bottom temperature was 2.5 degC below present, '
                   'recovering to the present value with eustatic sea-level rise; the '
                   '"SST-2.5" column is that correction term, not a temperature). '
                   'Confirmed by AC 2026-07-21.'),
    'ODP 806B': dict(
        sst_col='sst-mg/ca.adj',
        method_add='SST taken from the "sst-mg/ca.adj" column, adjusted for variation in '
                   'seawater Mg/Ca, in preference to the unadjusted "sst-mg/ca". This applies '
                   'the same ruling AC made for ODP 847 consistently across this file family. '
                   'Confirmed by AC 2026-07-21.'),
}.items()}

# The GeoB error-column correction (section 7).
GEOB_ERR_FIX = {norm(k) for k in ('GeoB10083', 'GeoB10163', 'GeoB10285')}
GEOB_ERR_NOTE = (
    'Uncertainty taken from the "sst_err" column (a constant 0.3 degC), NOT from '
    '"d18Og.rub250-350_err" (~0.08 degC). The latter is an error on the oxygen-isotope '
    'measurement, not on SST, and 0.08 degC is not a physically achievable SST precision. '
    'The source does not state whether sst_err is 1sd or 2sd and does not define the column '
    'in its variable list; per AC (2026-07-21) it is read as 1sd and doubled, giving a 2sd '
    'range of +/-0.6 degC, which is consistent with the ~0.5-0.6 degC uncertainty usually '
    'quoted for the Prahl et al. (1988) UK\'37 calibration used by this study.')

# Raised as a side note in need_to_confirm.docx at AC's request (2026-07-21).
GEOB_ERR_FLAG = (
    'sst_err is a constant 0.3 degC and is not defined in the source variable list, so its '
    'sd multiple is not stated anywhere in the file. It has been read as 1sd and doubled to '
    'a 2sd range of +/-0.6 degC. Worth confirming against Mohtadi et al. if these three cores '
    'end up carrying weight in the reconstruction.')


def excluded(site):
    return EXCLUDE.get(norm(site))


def fix_for(site):
    return FIXES.get(norm(site), {})
