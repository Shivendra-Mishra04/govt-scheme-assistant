"""Rule-based eligibility checker for PM-USP CSSS scholarship.
All rules come from data/schemes.json, which is derived from the official guidelines.
"""
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from src.config import SCHEMES_JSON


def load_scheme_data():
    """Load scheme configuration from schemes.json."""
    with open(SCHEMES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def check_eligibility(profile):
    """Check eligibility for PM-USP CSSS scholarship.

    Args:
        profile: dict with fields:
            - class12_percentile (float)
            - course_mode (str: "regular" | "distance" | "correspondence")
            - course_type (str: "diploma" | "undergraduate" | "postgraduate" | "professional_5yr" | "btech_beng")
            - year_of_study (int)
            - institution_recognized (bool)
            - availing_other_scholarship_or_fee_waiver (bool)
            - family_income_per_year (number)
            - is_renewal (bool, default false)

            For renewal only:
            - annual_exam_percent (float)
            - attendance_percent (float)
            - disciplinary_or_criminal_complaint (bool)

            Optional informational:
            - gender (str)
            - category (str)

    Interpretation used by this checker (Section 4 of the guidelines):
        - Fresh application: clauses i to v (percentile, regular mode, not
          diploma, recognized institution, no other scholarship, income).
        - Renewal: the ongoing conditions (clauses ii, iii, iv) plus the
          renewal criteria in clause viii (50% marks, 75% attendance, no
          complaint). Percentile (clause i) is an entry criterion and is not
          re-checked. Income is not re-checked because clause v requires the
          income certificate only for fresh applicants; the guidelines do not
          say whether the limit is re-applied at renewal.

    Returns:
        dict with keys:
            - eligible (bool)
            - checks (list of dicts: {rule, passed, detail, guideline_section})
            - annual_amount (int or None)
            - notes (list of str)
    """
    scheme = load_scheme_data()

    # Validation
    errors = []
    required_fields = [
        "class12_percentile", "course_mode", "course_type",
        "year_of_study", "institution_recognized",
        "availing_other_scholarship_or_fee_waiver", "family_income_per_year"
    ]

    for field in required_fields:
        if field not in profile:
            errors.append(f"Missing required field: {field}")

    is_renewal = profile.get("is_renewal", False)
    if is_renewal:
        renewal_fields = ["annual_exam_percent", "attendance_percent", "disciplinary_or_criminal_complaint"]
        for field in renewal_fields:
            if field not in profile:
                errors.append(f"Missing required renewal field: {field}")

    if errors:
        return {
            "eligible": False,
            "checks": [],
            "annual_amount": None,
            "notes": [],
            "error": "; ".join(errors)
        }

    checks = []

    # Entry criterion: fresh applications only
    if not is_renewal:
        # Percentile > 80 (Section 4, clause i)
        percentile_threshold = scheme["eligibility_criteria"]["percentile_threshold"]
        percentile_passed = profile["class12_percentile"] > percentile_threshold
        checks.append({
            "rule": f"Class XII percentile above {percentile_threshold}th among successful candidates",
            "passed": percentile_passed,
            "detail": f"Your percentile: {profile['class12_percentile']}",
            "guideline_section": "Section 4, clause i"
        })

    # Conditions checked for BOTH fresh applications and renewals

    # Regular course mode (clause ii)
    mode_passed = profile["course_mode"] == "regular"
    checks.append({
        "rule": "Course must be regular (not distance or correspondence)",
        "passed": mode_passed,
        "detail": f"Your mode: {profile['course_mode']}",
        "guideline_section": "Section 4, clause ii"
    })

    # Not diploma (clause ii)
    type_passed = profile["course_type"] != "diploma"
    checks.append({
        "rule": "Course type must not be diploma",
        "passed": type_passed,
        "detail": f"Your course type: {profile['course_type']}",
        "guideline_section": "Section 4, clause ii"
    })

    # Institution recognized (clause iii)
    institution_passed = profile["institution_recognized"]
    checks.append({
        "rule": "Institution must be recognized by AICTE or the respective regulatory body",
        "passed": institution_passed,
        "detail": "Recognized" if institution_passed else "Not recognized",
        "guideline_section": "Section 4, clause iii"
    })

    # No other scholarship or fee waiver (clause iv)
    other_scholarship_passed = not profile["availing_other_scholarship_or_fee_waiver"]
    checks.append({
        "rule": "Not availing any other merit scholarship, state scholarship, or fee waiver",
        "passed": other_scholarship_passed,
        "detail": "Availing other support" if not other_scholarship_passed else "Not availing other support",
        "guideline_section": "Section 4, clause iv"
    })

    # Family income threshold: fresh applications only (clause v)
    if not is_renewal:
        income_threshold = scheme["eligibility_criteria"]["income_threshold"]
        income_passed = profile["family_income_per_year"] <= income_threshold
        checks.append({
            "rule": f"Family income not exceeding ₹{income_threshold:,} per annum",
            "passed": income_passed,
            "detail": f"Your family income: ₹{profile['family_income_per_year']:,}",
            "guideline_section": "Section 4, clause v"
        })

    # Renewal criteria (clause viii)
    if is_renewal:
        # Annual exam >= 50%
        exam_threshold = scheme["renewal_criteria"]["annual_exam_threshold"]
        exam_passed = profile["annual_exam_percent"] >= exam_threshold
        checks.append({
            "rule": f"Annual examination result at least {exam_threshold}%",
            "passed": exam_passed,
            "detail": f"Your exam result: {profile['annual_exam_percent']}%",
            "guideline_section": "Section 4, clause viii"
        })

        # Attendance >= 75%
        attendance_threshold = scheme["renewal_criteria"]["attendance_threshold"]
        attendance_passed = profile["attendance_percent"] >= attendance_threshold
        checks.append({
            "rule": f"Attendance at least {attendance_threshold}%",
            "passed": attendance_passed,
            "detail": f"Your attendance: {profile['attendance_percent']}%",
            "guideline_section": "Section 4, clause viii"
        })

        # No disciplinary or criminal complaint
        discipline_passed = not profile["disciplinary_or_criminal_complaint"]
        checks.append({
            "rule": "No disciplinary or criminal complaint",
            "passed": discipline_passed,
            "detail": "Complaint exists" if not discipline_passed else "No complaints",
            "guideline_section": "Section 4, clause viii"
        })

    # Determine eligibility
    all_passed = all(check["passed"] for check in checks)
    eligible = all_passed

    # Calculate annual amount
    annual_amount = None
    amount_notes = []

    if eligible:
        course_type = profile["course_type"]
        year = profile["year_of_study"]
        rates = scheme["scholarship_rates"]

        if course_type == "undergraduate":
            if 1 <= year <= 3:
                annual_amount = rates["undergraduate_years_1_3"]
            else:
                amount_notes.append(f"Undergraduate course: year {year} is outside the typical 3-year range")

        elif course_type == "postgraduate":
            if year <= 2:
                annual_amount = rates["postgraduate"]
            else:
                amount_notes.append(f"Postgraduate course: year {year} is outside the typical 2-year range")

        elif course_type == "professional_5yr":
            if 1 <= year <= 3:
                annual_amount = rates["professional_5yr_years_1_3"]
            elif 4 <= year <= 5:
                annual_amount = rates["professional_5yr_years_4_5"]
            else:
                amount_notes.append(f"Professional 5-year course: year {year} is outside the 5-year range")

        elif course_type == "btech_beng":
            if 1 <= year <= 3:
                annual_amount = rates["btech_beng_years_1_3"]
            elif year == 4:
                annual_amount = rates["btech_beng_year_4"]
            else:
                amount_notes.append(f"B.Tech/B.Eng course: year {year} is outside the 4-year range")

        elif course_type == "diploma":
            amount_notes.append("Diploma courses are not covered under this scheme")

        else:
            amount_notes.append(f"Unknown course type: {course_type}")

    # Standard notes
    notes = [
        "Meeting these criteria does not guarantee selection; scholarships are limited to 82,000 fresh slots a year and are awarded on merit, with 50% earmarked for girls and central reservation (15% SC, 7.5% ST, 27% OBC, 5% horizontal for benchmark disabilities).",
        "Apply only online on the National Scholarships Portal (www.scholarships.gov.in)."
    ]
    if is_renewal:
        notes.append(
            "For renewals, the Class XII percentile is not re-checked, and the family income limit is "
            "not re-checked because the guidelines (Section 4, clause v) require the income certificate "
            "only for fresh applicants."
        )
    notes.extend(amount_notes)

    return {
        "eligible": eligible,
        "checks": checks,
        "annual_amount": annual_amount,
        "notes": notes
    }