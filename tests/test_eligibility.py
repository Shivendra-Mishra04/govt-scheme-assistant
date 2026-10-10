"""Tests for eligibility checker."""
import pytest
from src.eligibility import check_eligibility


def test_eligible_fresh_undergraduate():
    """Test eligible fresh undergraduate student."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 2,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is True
    assert result["annual_amount"] == 12000
    assert all(check["passed"] for check in result["checks"])


def test_percentile_exactly_80_fails():
    """Test that exactly 80th percentile fails (must be above 80)."""
    profile = {
        "class12_percentile": 80.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    percentile_check = [c for c in result["checks"] if "percentile" in c["rule"].lower()][0]
    assert percentile_check["passed"] is False


def test_percentile_80_point_1_passes():
    """Test that 80.1 percentile passes."""
    profile = {
        "class12_percentile": 80.1,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    percentile_check = [c for c in result["checks"] if "percentile" in c["rule"].lower()][0]
    assert percentile_check["passed"] is True


def test_income_exactly_450000_passes():
    """Test that income exactly at 450000 passes (≤ threshold)."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 450000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    income_check = [c for c in result["checks"] if "income" in c["rule"].lower()][0]
    assert income_check["passed"] is True


def test_income_450001_fails():
    """Test that income of 450001 fails."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 450001,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    income_check = [c for c in result["checks"] if "income" in c["rule"].lower()][0]
    assert income_check["passed"] is False


def test_distance_mode_fails():
    """Test that distance education mode fails."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "distance",
        "course_type": "undergraduate",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    mode_check = [c for c in result["checks"] if "regular" in c["rule"].lower()][0]
    assert mode_check["passed"] is False


def test_diploma_fails():
    """Test that diploma courses fail."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "diploma",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    type_check = [c for c in result["checks"] if "diploma" in c["rule"].lower()][0]
    assert type_check["passed"] is False


def test_unrecognized_institution_fails():
    """Test that unrecognized institution fails."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 1,
        "institution_recognized": False,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    inst_check = [c for c in result["checks"] if "recognized" in c["rule"].lower()][0]
    assert inst_check["passed"] is False


def test_other_scholarship_fails():
    """Test that having another scholarship fails."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": True,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    other_check = [c for c in result["checks"] if "other" in c["rule"].lower()][0]
    assert other_check["passed"] is False


def test_renewal_49_percent_fails():
    """Test that 49% marks fail renewal."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 2,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": True,
        "annual_exam_percent": 49.0,
        "attendance_percent": 80.0,
        "disciplinary_or_criminal_complaint": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    exam_check = [c for c in result["checks"] if "exam" in c["rule"].lower()][0]
    assert exam_check["passed"] is False


def test_renewal_50_percent_passes():
    """Test that 50% marks pass renewal."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 2,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": True,
        "annual_exam_percent": 50.0,
        "attendance_percent": 80.0,
        "disciplinary_or_criminal_complaint": False
    }
    result = check_eligibility(profile)
    exam_check = [c for c in result["checks"] if "exam" in c["rule"].lower()][0]
    assert exam_check["passed"] is True


def test_renewal_74_attendance_fails():
    """Test that 74% attendance fails renewal."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 2,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": True,
        "annual_exam_percent": 60.0,
        "attendance_percent": 74.0,
        "disciplinary_or_criminal_complaint": False
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    att_check = [c for c in result["checks"] if "attendance" in c["rule"].lower()][0]
    assert att_check["passed"] is False


def test_renewal_75_attendance_passes():
    """Test that 75% attendance passes renewal."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 2,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": True,
        "annual_exam_percent": 60.0,
        "attendance_percent": 75.0,
        "disciplinary_or_criminal_complaint": False
    }
    result = check_eligibility(profile)
    att_check = [c for c in result["checks"] if "attendance" in c["rule"].lower()][0]
    assert att_check["passed"] is True


def test_renewal_disciplinary_complaint_fails():
    """Test that having a disciplinary complaint fails renewal."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 2,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": True,
        "annual_exam_percent": 60.0,
        "attendance_percent": 80.0,
        "disciplinary_or_criminal_complaint": True
    }
    result = check_eligibility(profile)
    assert result["eligible"] is False
    disc_check = [c for c in result["checks"] if "disciplinary" in c["rule"].lower()][0]
    assert disc_check["passed"] is False


def test_amount_ug_year_2():
    """Test amount for undergraduate year 2."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "undergraduate",
        "year_of_study": 2,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["annual_amount"] == 12000


def test_amount_postgraduate():
    """Test amount for postgraduate."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "postgraduate",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["annual_amount"] == 20000


def test_amount_professional_5yr_year_4():
    """Test amount for professional 5-year course, year 4."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "professional_5yr",
        "year_of_study": 4,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["annual_amount"] == 20000


def test_amount_professional_5yr_year_5():
    """Test amount for professional 5-year course, year 5."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "professional_5yr",
        "year_of_study": 5,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["annual_amount"] == 20000


def test_amount_btech_year_3():
    """Test amount for B.Tech year 3."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "btech_beng",
        "year_of_study": 3,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["annual_amount"] == 12000


def test_amount_btech_year_4():
    """Test amount for B.Tech year 4 (graduation level)."""
    profile = {
        "class12_percentile": 85.0,
        "course_mode": "regular",
        "course_type": "btech_beng",
        "year_of_study": 4,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000,
        "is_renewal": False
    }
    result = check_eligibility(profile)
    assert result["annual_amount"] == 20000


def test_invalid_input_missing_field():
    """Test that missing required field returns error."""
    profile = {
        "class12_percentile": 85.0,
        # missing course_mode
        "course_type": "undergraduate",
        "year_of_study": 1,
        "institution_recognized": True,
        "availing_other_scholarship_or_fee_waiver": False,
        "family_income_per_year": 300000
    }
    result = check_eligibility(profile)
    assert "error" in result
    assert "course_mode" in result["error"]
