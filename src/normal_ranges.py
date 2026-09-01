"""
normal_ranges.py — WHO Reference Ranges & Anemia Diagnostic Engine.

Compares blood cell parameters against WHO guidelines, calculates estimated
Hemoglobin (Hb) & Hematocrit (HCT), grades Anemia severity, and provides
anemia subtype differential suggestions.
"""

from dataclasses import dataclass, field
from enum import Enum
from src.utils import get_logger

log = get_logger(__name__)


# ── Reference Ranges ──────────────────────────────────────────────────────────────

NORMAL_RANGES: dict[str, dict] = {
    "rbc": {
        "label": "Red Blood Cells (RBC)",
        "min": 4_200_000,
        "max": 5_800_000,
        "unit": "cells/µL",
        "low_condition":  "Anemia Detected (Decreased Oxygen Carrying Capacity)",
        "high_condition": "Polycythemia (Erythrocytosis / Blood Hyperviscosity Risk)",
    },
    "wbc": {
        "label": "White Blood Cells (WBC)",
        "min": 4_500,
        "max": 11_000,
        "unit": "cells/µL",
        "low_condition":  "Leukopenia (Immunodeficiency / Bone Marrow Suppression)",
        "high_condition": "Leukocytosis (Infection / Inflammation)",
    },
    "platelet": {
        "label": "Platelets",
        "min": 150_000,
        "max": 400_000,
        "unit": "cells/µL",
        "low_condition":  "Thrombocytopenia (Bleeding Risk / Platelet Consumption)",
        "high_condition": "Thrombocytosis (Reactive / Clotting Risk)",
    },
}

WBC_DIFF_RANGES: dict[str, dict] = {
    "Neutrophil": {
        "min": 50, "max": 70,
        "low_condition":  "Neutropenia",
        "high_condition": "Neutrophilia (Bacterial Infection / Inflammation)",
    },
    "Lymphocyte": {
        "min": 20, "max": 40,
        "low_condition":  "Lymphocytopenia",
        "high_condition": "Lymphocytosis (Viral Infection)",
    },
    "Monocyte": {
        "min": 2, "max": 8,
        "low_condition":  "Monocytopenia",
        "high_condition": "Monocytosis (Chronic Inflammation)",
    },
    "Eosinophil": {
        "min": 1, "max": 4,
        "low_condition":  "Eosinopenia",
        "high_condition": "Eosinophilia (Allergy / Parasitic Helminth Infection)",
    },
    "Basophil": {
        "min": 0, "max": 1,
        "low_condition":  "Basopenia",
        "high_condition": "Basophilia (Myeloproliferative Disorder)",
    },
}


# ── Data Classes ─────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    NORMAL  = "Normal"
    LOW     = "Low"
    HIGH    = "High"
    UNKNOWN = "Undetermined"


class AnemiaGrade(str, Enum):
    NONE     = "No Anemia (Normal RBC)"
    MILD     = "Mild Anemia"
    MODERATE = "Moderate Anemia"
    SEVERE   = "Severe Anemia"
    POLY     = "Polycythemia (High RBC)"


@dataclass
class CellFlag:
    cell_type:  str
    label:      str
    value:      int | float
    unit:       str
    severity:   Severity
    condition:  str
    reference:  str


@dataclass
class AnalysisFlags:
    cell_flags:            list[CellFlag]  = field(default_factory=list)
    diff_flags:            list[CellFlag]  = field(default_factory=list)
    overall:               str             = "Normal"    # "Normal" | "Abnormal"
    summary:               list[str]       = field(default_factory=list)
    
    # Anemia Specific Diagnostics
    anemia_detected:       bool            = False
    anemia_grade:          AnemiaGrade     = AnemiaGrade.NONE
    estimated_hb:          float           = 15.0        # g/dL
    estimated_hct:         float           = 45.0        # %
    anemia_type_hint:      str             = "Normal Erythrogram"


# ── Anemia Profiling Engine ───────────────────────────────────────────────────────

def evaluate_anemia(calibrated_counts: dict[str, int], wbc_diff: dict[str, float]) -> tuple[bool, AnemiaGrade, float, float, str]:
    """
    Calculate estimated Hemoglobin & Hematocrit, grade Anemia severity,
    and determine differential anemia subtype indications.
    """
    rbc = calibrated_counts.get("rbc")
    wbc = calibrated_counts.get("wbc")
    plt = calibrated_counts.get("platelet")

    # If any essential cell type is undetermined (None), return an indeterminate
    # anemia result so upstream code can mark the analysis as needing re-scan.
    if rbc is None or wbc is None or plt is None:
        return False, AnemiaGrade.NONE, 0.0, 0.0, "Undetermined — insufficient sampling; consider re-scan"

    # Physiological Estimation Formulas based on standard hematological indices
    # Hb (g/dL) ~ (RBC_count / 5.0M) * 15.0 g/dL
    est_hb = round((rbc / 5_000_000.0) * 15.0, 1)
    est_hct = round(est_hb * 3.0, 1)

    # Anemia Severity Classification (WHO Standards)
    if rbc < 2_500_000:
        detected = True
        grade = AnemiaGrade.SEVERE
    elif rbc < 3_500_000:
        detected = True
        grade = AnemiaGrade.MODERATE
    elif rbc < 4_200_000:
        detected = True
        grade = AnemiaGrade.MILD
    elif rbc > 5_800_000:
        detected = False
        grade = AnemiaGrade.POLY
    else:
        detected = False
        grade = AnemiaGrade.NONE

    # Differential Anemia Subtype Hints
    if not detected:
        if grade == AnemiaGrade.POLY:
            type_hint = "Polycythemia / Relative Hemoconcentration"
        else:
            type_hint = "Normal RBC Count (No Anemia Detected)"
    else:
        # Check pancytopenia (RBC low + WBC low + Platelet low) -> Aplastic Anemia
        if wbc < 4_500 and plt < 150_000:
            type_hint = "Aplastic Anemia / Bone Marrow Hypoplasia Risk (Pancytopenia Pattern)"
        # Check infection-induced anemia (RBC low + WBC high)
        elif wbc > 11_000:
            type_hint = "Anemia of Chronic Disease / Infection-Induced Anemia"
        # Check eosinophilia (Allergic / Helminth parasite induced iron loss)
        elif wbc_diff.get("Eosinophil", 0) > 4.0:
            type_hint = "Microcytic Anemia Risk (Helminth / Parasitic Blood Loss)"
        # Default iron deficiency / normocytic profile
        elif grade in (AnemiaGrade.SEVERE, AnemiaGrade.MODERATE):
            type_hint = "Iron Deficiency Anemia / Microcytic-Hypochromic Pattern (Primary Suspect)"
        else:
            type_hint = "Mild Iron Deficiency or Early Normocytic Anemia"

    return detected, grade, est_hb, est_hct, type_hint


# ── Flagging Logic ────────────────────────────────────────────────────────────────

def flag_cell_counts(calibrated_counts: dict[str, int]) -> list[CellFlag]:
    flags: list[CellFlag] = []
    for cell_type, count in calibrated_counts.items():
        ref = NORMAL_RANGES.get(cell_type)
        if ref is None:
            continue

        lo, hi = ref["min"], ref["max"]
        ref_str = f"{lo:,} – {hi:,} {ref['unit']}"

        if count is None:
            severity = Severity.UNKNOWN
            condition = "Undetermined — insufficient sampling; consider re-scan"
        elif count < lo:
            severity  = Severity.LOW
            condition = ref["low_condition"]
        elif count > hi:
            severity  = Severity.HIGH
            condition = ref["high_condition"]
        else:
            severity  = Severity.NORMAL
            condition = "Within normal limits"

        flags.append(CellFlag(
            cell_type = cell_type,
            label     = ref["label"],
            value     = count,
            unit      = ref["unit"],
            severity  = severity,
            condition = condition,
            reference = ref_str,
        ))
    return flags


def flag_wbc_differential(diff: dict[str, float]) -> list[CellFlag]:
    flags: list[CellFlag] = []
    for subtype, pct in diff.items():
        ref = WBC_DIFF_RANGES.get(subtype)
        if ref is None:
            continue

        lo, hi = ref["min"], ref["max"]
        ref_str = f"{lo} – {hi}%"

        if pct < lo:
            severity  = Severity.LOW
            condition = ref["low_condition"]
        elif pct > hi:
            severity  = Severity.HIGH
            condition = ref["high_condition"]
        else:
            severity  = Severity.NORMAL
            condition = "Within normal limits"

        flags.append(CellFlag(
            cell_type = subtype,
            label     = f"WBC Differential — {subtype}",
            value     = round(pct, 1),
            unit      = "%",
            severity  = severity,
            condition = condition,
            reference = ref_str,
        ))
    return flags


def run_full_analysis(
    calibrated_counts: dict[str, int],
    wbc_diff: dict[str, float],
) -> AnalysisFlags:
    """Run full blood analysis focusing on Anemia classification."""
    cell_flags = flag_cell_counts(calibrated_counts)
    diff_flags = flag_wbc_differential(wbc_diff)

    detected, grade, hb, hct, type_hint = evaluate_anemia(calibrated_counts, wbc_diff)

    all_flags = cell_flags + diff_flags
    abnormal_flags = [f for f in all_flags if f.severity != Severity.NORMAL]

    overall = "Abnormal" if (abnormal_flags or detected) else "Normal"

    summary: list[str] = []
    if detected:
        summary.append(f"🩸 ANEMIA DIAGNOSIS: {grade.value} (Est. Hb: {hb} g/dL, HCT: {hct}%)")
        summary.append(f"📋 Suspected Pattern: {type_hint}")

    if not abnormal_flags and not detected:
        summary.append("✅ Normal RBC count & Blood parameters. No Anemia detected.")
    else:
        for f in abnormal_flags:
            # Safely format numeric values; allow `None` for undetermined counts
            val_str = f"{f.value:,}" if f.value is not None else "Undetermined"
            summary.append(
                f"⚠️ {f.label}: {val_str} {f.unit} "
                f"({f.severity.value}) — {f.condition}"
            )

    return AnalysisFlags(
        cell_flags        = cell_flags,
        diff_flags        = diff_flags,
        overall           = overall,
        summary           = summary,
        anemia_detected   = detected,
        anemia_grade      = grade,
        estimated_hb      = hb,
        estimated_hct     = hct,
        anemia_type_hint  = type_hint,
    )
