"""Tests for the quantum_prof package."""
import math
import pytest

from quantum_prof.physics import (
    HBAR, EV, A0, RYDBERG_EV,
    infinite_square_well,
    harmonic_oscillator,
    hydrogen_level,
    hydrogen_transition,
    de_broglie,
    photon_energy,
    uncertainty_bound,
    qubit_state,
    apply_gate,
    measure_probabilities,
    bloch_vector,
    qubit_report,
    solve,
    SOLVERS,
    M_E,
    H,
)
from quantum_prof.library import TOPICS, BOOKS, learning_path, topic as get_topic
from quantum_prof.problems import PROBLEMS, problems_for_topic
from quantum_prof.serialize import topic_to_dict, book_to_dict, problem_to_dict


# ── Constants ──────────────────────────────────────────────────────────────

def test_physics_constants():
    assert abs(HBAR - 1.0545718176461565e-34) < 1e-45
    assert abs(EV - 1.602176634e-19) < 1e-29
    assert abs(RYDBERG_EV - 13.605693122994) < 1e-6
    assert abs(A0 - 5.29177210903e-11) < 1e-20


# ── Physics functions ──────────────────────────────────────────────────────

def test_infinite_square_well_ground_state():
    L = 1e-9
    e1 = infinite_square_well(1, L)
    e2 = infinite_square_well(2, L)
    assert e1 > 0
    assert abs(e2 / e1 - 4.0) < 1e-9
    e1_expected = (math.pi * HBAR) ** 2 / (2 * M_E * L ** 2) / EV
    assert abs(e1 - e1_expected) < 1e-10


def test_infinite_square_well_invalid():
    with pytest.raises(ValueError):
        infinite_square_well(0, 1e-9)


def test_harmonic_oscillator():
    omega = 1e14
    e0 = harmonic_oscillator(0, omega)
    e1 = harmonic_oscillator(1, omega)
    assert abs(e1 - e0 - HBAR * omega) < 1e-40


def test_hydrogen_level_ground():
    e1 = hydrogen_level(1)
    assert abs(e1 - (-RYDBERG_EV)) < 1e-9
    assert abs(e1 + 13.605693122994) < 1e-6


def test_hydrogen_level_scaling():
    e1 = hydrogen_level(1)
    e2 = hydrogen_level(2)
    assert abs(e2 / e1 - 0.25) < 1e-9


def test_hydrogen_level_invalid():
    with pytest.raises(ValueError):
        hydrogen_level(0)


def test_hydrogen_transition_lyman_alpha():
    result = hydrogen_transition(2, 1)
    assert result["emission"] is True
    wl = result["wavelength_nm"]
    assert abs(wl - 121.567) < 0.5


def test_hydrogen_transition_balmer_alpha():
    result = hydrogen_transition(3, 2)
    wl = result["wavelength_nm"]
    assert abs(wl - 656.3) < 1.0
    assert "visible" in result["region"].lower()


def test_de_broglie():
    wl = de_broglie(M_E, 1e6)
    expected = H / (M_E * 1e6)
    assert abs(wl - expected) < 1e-40


def test_de_broglie_invalid():
    with pytest.raises(ValueError):
        de_broglie(0, 1e6)
    with pytest.raises(ValueError):
        de_broglie(M_E, 0)


def test_photon_energy_visible():
    e = photon_energy(500)
    assert abs(e - 2.48) < 0.02


def test_photon_energy_invalid():
    with pytest.raises(ValueError):
        photon_energy(0)


def test_uncertainty_bound():
    dx = 1e-10
    dp = uncertainty_bound(dx)
    assert abs(dp - HBAR / (2 * dx)) < 1e-50


def test_uncertainty_invalid():
    with pytest.raises(ValueError):
        uncertainty_bound(0)


# ── Qubit / gates ──────────────────────────────────────────────────────────

def test_qubit_state_normalization():
    a, b = qubit_state(3 + 0j, 4 + 0j)
    p0, p1 = measure_probabilities((a, b))
    assert abs(p0 + p1 - 1.0) < 1e-10


def test_qubit_x_gate_flips():
    state = qubit_state(1 + 0j, 0j)
    new_state = apply_gate("X", state)
    p0, p1 = measure_probabilities(new_state)
    assert abs(p0) < 1e-10
    assert abs(p1 - 1.0) < 1e-10


def test_qubit_h_gate_superposition():
    state = qubit_state(1 + 0j, 0j)
    new_state = apply_gate("H", state)
    p0, p1 = measure_probabilities(new_state)
    assert abs(p0 - 0.5) < 1e-9
    assert abs(p1 - 0.5) < 1e-9


def test_qubit_z_gate():
    state = qubit_state(0j, 1 + 0j)
    new_state = apply_gate("Z", state)
    p0, p1 = measure_probabilities(new_state)
    assert abs(p0) < 1e-10
    assert abs(p1 - 1.0) < 1e-10


def test_bloch_vector_poles():
    state0 = qubit_state(1 + 0j, 0j)
    bv = bloch_vector(state0)
    assert abs(bv[2] - 1.0) < 1e-9

    state1 = qubit_state(0j, 1 + 0j)
    bv1 = bloch_vector(state1)
    assert abs(bv1[2] + 1.0) < 1e-9


def test_invalid_gate():
    state = qubit_state(1 + 0j, 0j)
    with pytest.raises(ValueError):
        apply_gate("CNOT", state)


def test_qubit_report_keys():
    state = qubit_state(1 + 0j, 0j)
    report = qubit_report(state)
    assert "alpha" in report
    assert "beta" in report
    assert "P0" in report
    assert "P1" in report
    assert "bloch" in report


# ── SOLVERS ────────────────────────────────────────────────────────────────

def test_solvers_keys():
    expected = {"particle-in-a-box", "harmonic-oscillator", "hydrogen-atom", "uncertainty-principle"}
    for key in expected:
        assert key in SOLVERS


def test_solve_particle_in_a_box():
    result = solve("particle-in-a-box", n=1, L=1e-9)
    assert "energy_eV" in result
    assert result["energy_eV"] > 0


def test_solve_unknown():
    with pytest.raises(ValueError):
        solve("no-such-solver")


# ── Library ────────────────────────────────────────────────────────────────

def test_library_topic_count():
    assert len(TOPICS) >= 32


def test_library_book_count():
    assert len(BOOKS) >= 13


def test_library_all_levels():
    levels = {t.level for t in TOPICS.values()}
    assert "basics" in levels
    assert "intermediate" in levels
    assert "advanced" in levels
    assert "advanced_topics" in levels


def test_library_topic_prerequisites_exist():
    for t in TOPICS.values():
        for prereq in t.prerequisites:
            assert prereq in TOPICS, f"Topic {t.id!r} has missing prereq {prereq!r}"


def test_library_books_reference_topics():
    for b in BOOKS.values():
        for topic_ref in b.topics:
            assert topic_ref in TOPICS, f"Book {b.id!r} references unknown topic {topic_ref!r}"


def test_get_topic():
    t = get_topic("particle-in-a-box")
    assert t.id == "particle-in-a-box"


def test_get_topic_invalid():
    with pytest.raises(KeyError):
        get_topic("no-such-topic")


# ── Learning path ──────────────────────────────────────────────────────────

def test_learning_path_no_cycles():
    for topic_id in TOPICS:
        path = learning_path(topic_id)
        assert len(path) == len(set(path)), f"Cycle detected in path to {topic_id!r}"


def test_learning_path_includes_target():
    path = learning_path("harmonic-oscillator")
    assert "harmonic-oscillator" in path


def test_learning_path_prerequisites_before_target():
    path = learning_path("hydrogen-atom")
    h_idx = path.index("hydrogen-atom")
    for prereq in TOPICS["hydrogen-atom"].prerequisites:
        assert path.index(prereq) < h_idx


# ── Problems ───────────────────────────────────────────────────────────────

def test_problems_not_empty():
    assert len(PROBLEMS) >= 10


def test_problem_fields():
    for p in PROBLEMS.values():
        assert p.id
        assert p.topic_id
        assert p.difficulty in ("easy", "medium", "hard")
        assert p.stem
        assert p.hint
        assert p.answer_latex


def test_problems_for_topic():
    probs = problems_for_topic("particle-in-a-box")
    assert len(probs) >= 1
    for p in probs:
        assert p.topic_id == "particle-in-a-box"


def test_problems_topic_ids_exist():
    for p in PROBLEMS.values():
        assert p.topic_id in TOPICS, f"Problem {p.id!r} references unknown topic {p.topic_id!r}"


# ── Serialization ──────────────────────────────────────────────────────────

def test_topic_to_dict():
    t = TOPICS["particle-in-a-box"]
    d = topic_to_dict(t)
    assert d["id"] == "particle-in-a-box"
    assert isinstance(d["key_concepts"], list)
    assert isinstance(d["prerequisites"], list)


def test_book_to_dict():
    b = list(BOOKS.values())[0]
    d = book_to_dict(b)
    assert "id" in d
    assert "authors" in d
    assert isinstance(d["authors"], list)


def test_problem_to_dict():
    p = list(PROBLEMS.values())[0]
    d = problem_to_dict(p)
    assert "id" in d
    assert "stem" in d
    assert "answer_latex" in d
