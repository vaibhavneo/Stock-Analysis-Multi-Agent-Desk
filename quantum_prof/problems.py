"""Practice problems for the quantum mechanics curriculum."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Problem:
    id: str
    topic_id: str
    difficulty: str  # "easy" | "medium" | "hard"
    stem: str
    hint: str
    answer_latex: str


_PROBLEMS_LIST: List[Problem] = [
    # wave-particle-duality
    Problem(
        id="wpd-1",
        topic_id="wave-particle-duality",
        difficulty="easy",
        stem=(
            "An electron gun fires electrons toward a narrow slit. "
            "A diffraction pattern forms on a screen. "
            "What does this demonstrate about the electron?"
        ),
        hint="Think about which classical objects produce diffraction patterns.",
        answer_latex=r"Diffraction is a wave phenomenon. The electron has a de Broglie wavelength \lambda = h/p, causing wave-like diffraction.",
    ),
    Problem(
        id="wpd-2",
        topic_id="wave-particle-duality",
        difficulty="medium",
        stem=(
            "Calculate the de Broglie wavelength of a proton "
            "(m = 1.673 × 10⁻²⁷ kg) moving at 1% of the speed of light."
        ),
        hint="Use λ = h/p = h/(mv).",
        answer_latex=r"\lambda = \frac{h}{mv} = \frac{6.626\times10^{-34}}{1.673\times10^{-27}\times3\times10^{6}} \approx 1.32 \times 10^{-13}\,\text{m}",
    ),
    # photoelectric-effect
    Problem(
        id="pe-1",
        topic_id="photoelectric-effect",
        difficulty="easy",
        stem=(
            "Light of frequency 8.0 × 10¹⁴ Hz strikes a metal surface "
            "with work function 2.3 eV. Find the maximum kinetic energy "
            "of the emitted electrons in eV."
        ),
        hint="KE_max = hf - φ. Convert hf to eV using 1 eV = 1.602×10⁻¹⁹ J.",
        answer_latex=r"KE_{\max} = hf - \phi = (6.626\times10^{-34}\times8\times10^{14})/1.602\times10^{-19} - 2.3 \approx 3.31 - 2.3 = 1.01\,\text{eV}",
    ),
    Problem(
        id="pe-2",
        topic_id="photoelectric-effect",
        difficulty="medium",
        stem=(
            "No electrons are emitted from a metal when illuminated with "
            "600 nm light, but they are emitted at 400 nm. "
            "Estimate the work function range."
        ),
        hint="E_photon = hc/λ. At threshold, KE = 0 so hf = φ.",
        answer_latex=r"E_{600} \approx 2.07\,\text{eV},\; E_{400} \approx 3.10\,\text{eV}. \text{ So } 2.07\,\text{eV} < \phi < 3.10\,\text{eV}.",
    ),
    # de-broglie
    Problem(
        id="db-1",
        topic_id="de-broglie",
        difficulty="easy",
        stem="A baseball (0.145 kg) travels at 40 m/s. What is its de Broglie wavelength?",
        hint="λ = h/p. Compare with atomic sizes (~10⁻¹⁰ m).",
        answer_latex=r"\lambda = \frac{6.626\times10^{-34}}{0.145\times40} \approx 1.14\times10^{-34}\,\text{m} \text{ (undetectably small)}",
    ),
    Problem(
        id="db-2",
        topic_id="de-broglie",
        difficulty="medium",
        stem=(
            "What speed must an electron have so that its de Broglie "
            "wavelength equals 0.1 nm (roughly the Bohr radius)?"
        ),
        hint="v = h/(mλ).",
        answer_latex=r"v = \frac{h}{m_e\lambda} = \frac{6.626\times10^{-34}}{9.109\times10^{-31}\times10^{-10}} \approx 7.27\times10^6\,\text{m/s}",
    ),
    # double-slit
    Problem(
        id="ds-1",
        topic_id="double-slit",
        difficulty="easy",
        stem=(
            "In a double-slit experiment with electrons, you place a "
            "detector at one slit to determine which slit each electron "
            "passes through. What happens to the interference pattern?"
        ),
        hint="Measurement disturbs the quantum state.",
        answer_latex=r"\text{The interference pattern disappears. Obtaining which-path information destroys the superposition.}",
    ),
    Problem(
        id="ds-2",
        topic_id="double-slit",
        difficulty="medium",
        stem=(
            "Electrons with λ = 0.2 nm pass through two slits separated "
            "by d = 10 nm. Find the angle θ of the first bright fringe."
        ),
        hint="For bright fringes: d sin θ = mλ, with m = 1.",
        answer_latex=r"\sin\theta = \frac{\lambda}{d} = \frac{0.2}{10} = 0.02 \Rightarrow \theta \approx 1.15°",
    ),
    # blackbody-radiation
    Problem(
        id="bb-1",
        topic_id="blackbody-radiation",
        difficulty="easy",
        stem="What is the energy of a photon with wavelength 550 nm (green light)?",
        hint="E = hc/λ.",
        answer_latex=r"E = \frac{hc}{\lambda} = \frac{6.626\times10^{-34}\times3\times10^8}{550\times10^{-9}} \approx 2.25\,\text{eV}",
    ),
    Problem(
        id="bb-2",
        topic_id="blackbody-radiation",
        difficulty="medium",
        stem=(
            "The sun's surface temperature is ~5778 K. "
            "Using Wien's law (λ_max T = 2.898×10⁻³ m·K), "
            "find the peak wavelength of solar emission."
        ),
        hint="λ_max = b/T.",
        answer_latex=r"\lambda_{\max} = \frac{2.898\times10^{-3}}{5778} \approx 501\,\text{nm (green-yellow)}",
    ),
    # bohr-model
    Problem(
        id="bohr-1",
        topic_id="bohr-model",
        difficulty="easy",
        stem="Calculate the energy of the n=3 level of hydrogen using the Bohr model.",
        hint="E_n = -13.6 / n² eV.",
        answer_latex=r"E_3 = \frac{-13.606}{9} \approx -1.51\,\text{eV}",
    ),
    Problem(
        id="bohr-2",
        topic_id="bohr-model",
        difficulty="medium",
        stem=(
            "Find the wavelength of light emitted when hydrogen transitions "
            "from n=3 to n=2 (Hα line)."
        ),
        hint="ΔE = E_3 − E_2; λ = hc/|ΔE|.",
        answer_latex=r"\Delta E = -1.51 - (-3.40) = 1.89\,\text{eV} \Rightarrow \lambda = \frac{hc}{1.89\,\text{eV}} \approx 656\,\text{nm (red)}",
    ),
    # uncertainty-principle
    Problem(
        id="up-1",
        topic_id="uncertainty-principle",
        difficulty="easy",
        stem=(
            "An electron is confined to a region Δx = 0.1 nm. "
            "What is the minimum uncertainty in its momentum?"
        ),
        hint="Δp ≥ ℏ/(2Δx).",
        answer_latex=r"\Delta p_{\min} = \frac{\hbar}{2\Delta x} = \frac{1.055\times10^{-34}}{2\times10^{-10}} \approx 5.27\times10^{-25}\,\text{kg·m/s}",
    ),
    Problem(
        id="up-2",
        topic_id="uncertainty-principle",
        difficulty="medium",
        stem=(
            "A particle has energy uncertainty ΔE = 1 eV. "
            "What is the minimum lifetime of the energy state (ΔEΔt ≥ ℏ/2)?"
        ),
        hint="Δt ≥ ℏ/(2ΔE).",
        answer_latex=r"\Delta t \geq \frac{\hbar}{2\Delta E} = \frac{1.055\times10^{-34}}{2\times1.602\times10^{-19}} \approx 3.3\times10^{-16}\,\text{s}",
    ),
    # wavefunction-born-rule
    Problem(
        id="wbr-1",
        topic_id="wavefunction-born-rule",
        difficulty="easy",
        stem=(
            "A particle has ψ(x) = A for 0 ≤ x ≤ L and 0 elsewhere. "
            "Find A so that ψ is normalised."
        ),
        hint="Integrate |ψ|² over all space and set equal to 1.",
        answer_latex=r"\int_0^L A^2\,dx = A^2 L = 1 \Rightarrow A = \frac{1}{\sqrt{L}}",
    ),
    Problem(
        id="wbr-2",
        topic_id="wavefunction-born-rule",
        difficulty="medium",
        stem=(
            "The wavefunction ψ = A e^{-x²/(2σ²)} (Gaussian). "
            "Find A for normalisation."
        ),
        hint="Use ∫_{-∞}^{∞} e^{-x²/σ²} dx = σ√π.",
        answer_latex=r"A = \left(\pi\sigma^2\right)^{-1/4}",
    ),
    # schrodinger-equation
    Problem(
        id="se-1",
        topic_id="schrodinger-equation",
        difficulty="easy",
        stem=(
            "Write the time-independent Schrödinger equation for a particle "
            "of mass m in a 1D potential V(x)."
        ),
        hint="Think about operator Ĥψ = Eψ.",
        answer_latex=r"-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi",
    ),
    Problem(
        id="se-2",
        topic_id="schrodinger-equation",
        difficulty="medium",
        stem=(
            "For a free particle (V=0), show that ψ = e^{ikx} solves "
            "the TISE and find the energy eigenvalue."
        ),
        hint="Compute -ℏ²/(2m) d²/dx² e^{ikx}.",
        answer_latex=r"-\frac{\hbar^2}{2m}(ik)^2 e^{ikx} = \frac{\hbar^2 k^2}{2m}e^{ikx} = Ee^{ikx} \Rightarrow E = \frac{\hbar^2 k^2}{2m}",
    ),
    # particle-in-a-box
    Problem(
        id="pib-1",
        topic_id="particle-in-a-box",
        difficulty="easy",
        stem=(
            "An electron is in an infinite square well of width L = 0.5 nm. "
            "Calculate the ground-state energy in eV."
        ),
        hint="E_1 = π²ℏ²/(2mL²).",
        answer_latex=r"E_1 = \frac{\pi^2(1.055\times10^{-34})^2}{2(9.109\times10^{-31})(5\times10^{-10})^2} \approx 1.51\,\text{eV}",
    ),
    Problem(
        id="pib-2",
        topic_id="particle-in-a-box",
        difficulty="medium",
        stem=(
            "For the infinite square well, find the probability of finding "
            "the particle in the left quarter (0 to L/4) for the n=1 state."
        ),
        hint="P = ∫₀^{L/4} |ψ₁(x)|² dx where ψ₁ = √(2/L) sin(πx/L).",
        answer_latex=r"P = \frac{2}{L}\int_0^{L/4}\sin^2\!\!\left(\frac{\pi x}{L}\right)dx = \frac{1}{4} - \frac{1}{2\pi} \approx 0.0908",
    ),
]

PROBLEMS: Dict[str, Problem] = {p.id: p for p in _PROBLEMS_LIST}


def problems_for_topic(topic_id: str) -> List[Problem]:
    return [p for p in _PROBLEMS_LIST if p.topic_id == topic_id]


def problem(problem_id: str) -> Problem:
    try:
        return PROBLEMS[problem_id]
    except KeyError:
        raise KeyError(f"Unknown problem id: {problem_id!r}. Available: {sorted(PROBLEMS)}")
