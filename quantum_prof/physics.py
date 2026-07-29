"""Quantum mechanics physics engine — pure stdlib, no external dependencies."""
from __future__ import annotations

import cmath
import math
from typing import Dict, Tuple

HBAR: float = 1.0545718176461565e-34
H: float = 6.62607015e-34
C: float = 299_792_458.0
M_E: float = 9.1093837015e-31
EV: float = 1.602176634e-19
A0: float = 5.29177210903e-11
RYDBERG_EV: float = 13.605693122994

_SQRT2_INV = 1.0 / math.sqrt(2)

GATES: Dict[str, Tuple[Tuple[complex, complex], Tuple[complex, complex]]] = {
    "I": ((1 + 0j, 0j), (0j, 1 + 0j)),
    "X": ((0j, 1 + 0j), (1 + 0j, 0j)),
    "Y": ((0j, -1j), (1j, 0j)),
    "Z": ((1 + 0j, 0j), (0j, -1 + 0j)),
    "H": ((_SQRT2_INV + 0j, _SQRT2_INV + 0j), (_SQRT2_INV + 0j, -_SQRT2_INV + 0j)),
    "S": ((1 + 0j, 0j), (0j, 1j)),
    "T": ((1 + 0j, 0j), (0j, cmath.exp(1j * math.pi / 4))),
}


def infinite_square_well(n: int, L: float, m: float = M_E) -> float:
    """Energy of level n in an infinite square well of width L (metres). Returns eV."""
    if n < 1:
        raise ValueError("n must be >= 1")
    energy_j = (n * math.pi * HBAR) ** 2 / (2 * m * L ** 2)
    return energy_j / EV


def harmonic_oscillator(n: int, omega: float) -> float:
    """Energy of level n for angular frequency omega. Returns Joules."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return HBAR * omega * (n + 0.5)


def hydrogen_level(n: int) -> float:
    """Energy of hydrogen level n. Returns eV (negative = bound)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return -RYDBERG_EV / n ** 2


def hydrogen_transition(n_i: int, n_f: int) -> Dict[str, object]:
    """Photon emitted/absorbed by hydrogen transition n_i -> n_f."""
    delta_e_ev = hydrogen_level(n_f) - hydrogen_level(n_i)
    delta_e_j = abs(delta_e_ev) * EV
    wavelength_m = H * C / delta_e_j
    wavelength_nm = wavelength_m * 1e9

    if wavelength_nm < 10:
        region = "gamma-ray"
    elif wavelength_nm < 122:
        region = "UV (Lyman series)"
    elif wavelength_nm < 400:
        region = "UV"
    elif wavelength_nm < 700:
        region = "visible"
    elif wavelength_nm < 2500:
        region = "near-IR (Paschen series)"
    else:
        region = "IR"

    return {
        "n_i": n_i,
        "n_f": n_f,
        "delta_energy_eV": round(delta_e_ev, 6),
        "wavelength_nm": round(wavelength_nm, 4),
        "region": region,
        "emission": delta_e_ev < 0,
    }


def de_broglie(mass_kg: float, speed_m_s: float) -> float:
    """de Broglie wavelength in metres."""
    if mass_kg <= 0 or speed_m_s <= 0:
        raise ValueError("mass and speed must be positive")
    return H / (mass_kg * speed_m_s)


def photon_energy(wavelength_nm: float) -> float:
    """Energy of a photon with given wavelength (nm). Returns eV."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    return (H * C) / (wavelength_nm * 1e-9 * EV)


def uncertainty_bound(delta_x_m: float) -> float:
    """Minimum momentum uncertainty given position uncertainty delta_x (m). Returns kg·m/s."""
    if delta_x_m <= 0:
        raise ValueError("delta_x must be positive")
    return HBAR / (2 * delta_x_m)


def qubit_state(alpha: complex, beta: complex) -> Tuple[complex, complex]:
    """Return normalised qubit state (alpha|0> + beta|1>)."""
    norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    if norm < 1e-15:
        raise ValueError("qubit state cannot be zero vector")
    return (alpha / norm, beta / norm)


def apply_gate(gate_name: str, state: Tuple[complex, complex]) -> Tuple[complex, complex]:
    """Apply named single-qubit gate to state."""
    gate_name = gate_name.upper()
    if gate_name not in GATES:
        raise ValueError(f"Unknown gate '{gate_name}'. Choose from {sorted(GATES)}")
    g = GATES[gate_name]
    a, b = state
    new_a = g[0][0] * a + g[0][1] * b
    new_b = g[1][0] * a + g[1][1] * b
    return (new_a, new_b)


def measure_probabilities(state: Tuple[complex, complex]) -> Tuple[float, float]:
    """Return (P(|0>), P(|1>)) for a qubit state."""
    a, b = state
    p0 = abs(a) ** 2
    p1 = abs(b) ** 2
    return (round(p0, 10), round(p1, 10))


def bloch_vector(state: Tuple[complex, complex]) -> Tuple[float, float, float]:
    """Return Bloch sphere (x, y, z) for a pure qubit state."""
    a, b = state
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = abs(a) ** 2 - abs(b) ** 2
    return (round(x, 10), round(y, 10), round(z, 10))


def qubit_report(state: Tuple[complex, complex]) -> dict:
    """Full report for a qubit state."""
    p0, p1 = measure_probabilities(state)
    bx, by, bz = bloch_vector(state)
    a, b = state
    return {
        "alpha": {"re": round(a.real, 8), "im": round(a.imag, 8)},
        "beta": {"re": round(b.real, 8), "im": round(b.imag, 8)},
        "P0": p0,
        "P1": p1,
        "bloch": {"x": bx, "y": by, "z": bz},
    }


def _solve_infinite_well(n: int = 1, L: float = 1e-9, m: float = M_E) -> dict:
    energy_ev = infinite_square_well(n, L, m)
    return {
        "topic": "particle-in-a-box",
        "n": n,
        "L_nm": L * 1e9,
        "energy_eV": round(energy_ev, 6),
        "energy_J": round(energy_ev * EV, 30),
        "formula": "E_n = n²π²ℏ² / (2mL²)",
    }


def _solve_qho(n: int = 0, omega: float = 1e14) -> dict:
    energy_j = harmonic_oscillator(n, omega)
    return {
        "topic": "harmonic-oscillator",
        "n": n,
        "omega_rad_s": omega,
        "energy_J": round(energy_j, 30),
        "energy_eV": round(energy_j / EV, 6),
        "formula": "E_n = ℏω(n + ½)",
    }


def _solve_hydrogen(n: int = 1) -> dict:
    energy_ev = hydrogen_level(n)
    return {
        "topic": "hydrogen-atom",
        "n": n,
        "energy_eV": round(energy_ev, 6),
        "formula": "E_n = -13.606 / n² eV",
    }


def _solve_hydrogen_transition(n_i: int = 3, n_f: int = 2) -> dict:
    return hydrogen_transition(n_i, n_f)


def _solve_de_broglie(mass_kg: float = M_E, speed_m_s: float = 1e6) -> dict:
    wavelength = de_broglie(mass_kg, speed_m_s)
    return {
        "topic": "de-broglie",
        "mass_kg": mass_kg,
        "speed_m_s": speed_m_s,
        "wavelength_m": wavelength,
        "wavelength_pm": round(wavelength * 1e12, 4),
        "formula": "λ = h / (mv)",
    }


def _solve_photon(wavelength_nm: float = 500.0) -> dict:
    energy_ev = photon_energy(wavelength_nm)
    return {
        "topic": "blackbody-radiation",
        "wavelength_nm": wavelength_nm,
        "energy_eV": round(energy_ev, 6),
        "formula": "E = hf = hc/λ",
    }


def _solve_uncertainty(delta_x_m: float = 1e-10) -> dict:
    min_dp = uncertainty_bound(delta_x_m)
    return {
        "topic": "uncertainty-principle",
        "delta_x_m": delta_x_m,
        "min_delta_p_kg_m_s": min_dp,
        "formula": "Δx·Δp ≥ ℏ/2",
    }


def _solve_qubit(gate: str = "H", alpha: float = 1.0, beta: float = 0.0) -> dict:
    state = qubit_state(complex(alpha), complex(beta))
    new_state = apply_gate(gate, state)
    return {
        "topic": "quantum-information",
        "gate": gate.upper(),
        "input_state": qubit_report(state),
        "output_state": qubit_report(new_state),
    }


SOLVERS: Dict[str, object] = {
    "particle-in-a-box": _solve_infinite_well,
    "harmonic-oscillator": _solve_qho,
    "hydrogen-atom": _solve_hydrogen,
    "hydrogen-transition": _solve_hydrogen_transition,
    "de-broglie": _solve_de_broglie,
    "blackbody-radiation": _solve_photon,
    "photon-energy": _solve_photon,
    "uncertainty-principle": _solve_uncertainty,
    "quantum-information": _solve_qubit,
    "qubit": _solve_qubit,
}


def solve(topic_id: str, **kwargs) -> dict:
    """Run the physics solver for the given topic_id."""
    solver = SOLVERS.get(topic_id)
    if solver is None:
        available = sorted(SOLVERS.keys())
        raise ValueError(
            f"No solver for '{topic_id}'. Available: {available}"
        )
    return solver(**kwargs)  # type: ignore[operator]
